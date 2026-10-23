"""One nbhost game process (one battle) behind the shared-memory protocol (proto.py).

    inst = Instance(0); inst.start(); inst.wait_ready()
    r = inst.reset(seed=1)            # -> Resp (ctypes view; copy what you keep)
    r = inst.step(actions={0: (move_dir, fire, aim_deg, super_dist)}, ticks=4, flags=F_STATE)
Every game process runs as `unshare -rn qemu-aarch64 nbhost ...` (no network namespace)."""
import ctypes as C, mmap, os, select, subprocess, time
import numpy as np
from . import proto as P

WORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Everything is relative to the repository (nbrl-env/): game host loader/nbhost, renderer glhost/nbgl, emulator
# qemu/bin/qemu-aarch64-nb, the extracted APK apk_extract/ (made by loader/setup_apk.sh from apk/*.apk).
# NBHOST_BIN / NBGL_BIN / NBHOST_QEMU / NBHOST_APK override them.
NBHOST = os.environ.get("NBHOST_BIN", os.path.join(WORK, "loader", "nbhost"))
NBGL = os.environ.get("NBGL_BIN", os.path.join(WORK, "glhost", "nbgl"))
PAYLOADS = os.path.join(WORK, "loader", "payloads")
APK = os.environ.get("NBHOST_APK", os.path.join(WORK, "apk_extract"))


class InstanceDead(RuntimeError):
    pass


def set_div(d, div):
    """fill a proto.Div from {"location", "theme", "chr", "skin"} (chr / skin: {player index: id} or a list by index)"""
    div = div or {}
    d.location = int(div.get("location") or 0); d.theme = int(div.get("theme") or 0)
    for f in ("chr", "skin", "gadget", "star"):
        v = div.get(f) or {}
        items = v.items() if isinstance(v, dict) else enumerate(v)
        arr = getattr(d, f)
        for i in range(P.MAXP): arr[i] = 0
        for i, x in items:
            if 0 <= int(i) < P.MAXP: arr[int(i)] = int(x or 0)
    d.setup_super_mask = int(div.get("setup_super_mask") or 0) & 0xffffffff
    d.setup_unprotect_mask = int(div.get("setup_unprotect_mask") or 0) & 0xffffffff
    for i in range(2): d.reserved[i] = 0


QEMU_NB = os.path.join(WORK, "qemu", "bin", "qemu-aarch64-nb")


def qemu_cmd_env():
    """The emulator: work/qemu/bin/qemu-aarch64-nb (QEMU 8.2.2 + work/qemu/patches: cross-page chaining, inline jump-cache
    probe, 16-bit jump cache; ~2x faster, same game state bit for bit) when it exists, else the system qemu-aarch64.
    NBHOST_QEMU=<path> picks another binary; NBHOST_QEMU=system the system one. The patched binary's switches are
    turned on here (QEMU_NB_XPAGE, QEMU_NB_INLINE) together with the cheap IMPDEF pointer-authentication algorithm
    (QEMU_CPU=max,pauth-impdef=on); a value already in the environment wins."""
    q = os.environ.get("NBHOST_QEMU", "")
    if q == "system" or (not q and not os.path.exists(QEMU_NB)):
        q = "qemu-aarch64"
    elif not q:
        q = QEMU_NB
    env = dict(os.environ)
    if os.path.basename(q).startswith("qemu-aarch64-nb"):
        env.setdefault("QEMU_NB_XPAGE", "1"); env.setdefault("QEMU_NB_INLINE", "1")
        env.setdefault("QEMU_CPU", "max,pauth-impdef=on")
    return q, env


class Instance:
    def __init__(self, idx, tag=None, log_dir=None, steps_per_frame=4, cores=2, extra_args=(), gl_shm=None, gl=False, size="2400x1080",
                 bell=None):
        """gl=True: client mode (--env-client) with its own headless nbgl renderer (pixels)"""
        self.idx, self.spf, self.cores, self.extra = idx, steps_per_frame, cores, list(extra_args)
        self.gl, self.size, self.glproc = gl, size, None
        self.tag = tag or f"{os.getpid()}"
        self.path = f"/dev/shm/nbenv_{self.tag}_{idx}"
        self.log_dir = log_dir or os.path.join(WORK, "runs", "instances")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"game_{self.tag}_{idx}.log")
        self.proc = None; self.mm = None; self.seq = 0; self.gl_shm = gl_shm
        self.starts = 0
        # bell pipes (NBENV_BELL=1): one byte per command / response so that both sides block instead of spinning
        self.bell = (os.environ.get("NBENV_BELL", "0") == "1") if bell is None else bool(bell); self.cmd_w = self.resp_r = -1
        self.bell_spin_us = 0   # sent with every command (cmd reserved[3]): how long nbhost spins before blocking (0 = 200 us)
        # "native + lead" aim (work/loader lead_aim; every command carries it): bit i = player i's native AI leads its shots
        self.lead_mask, self.lead_dt, self.lead_speed, self.lead_frac = 0, 0.0, 197.0, 1.0

    # ------------------------------------------------------------------ process
    def start(self):
        with open(self.path, "wb") as f:
            f.truncate(P.TOTAL)
        fd = os.open(self.path, os.O_RDWR)
        self.mm = mmap.mmap(fd, P.TOTAL, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        self.hdr = P.Hdr.from_buffer(self.mm, 0)
        self.cmd = P.Cmd.from_buffer(self.mm, P.OFF_CMD)
        self.resp = P.Resp.from_buffer(self.mm, P.OFF_RESP)
        self.state_i32 = np.frombuffer(self.mm, np.int32, P.STATE_CAP, P.OFF_STATE)
        self.map_i32 = np.frombuffer(self.mm, np.int32, P.MAP_CAP, P.OFF_MAP)
        self.pix_u8 = np.frombuffer(self.mm, np.uint8, P.PIX_CAP, P.OFF_PIX)
        self.obs_f32 = np.frombuffer(self.mm, np.float32, P.MAXP * (P.OBS_VEC + P.OBS_GRID), P.OFF_OBS)
        self.seq = 0
        if self.gl and not self.gl_shm:
            self.gl_shm = f"/dev/shm/nbgl_{self.tag}_{self.idx}"
        if self.gl:
            self.gllog = open(self.log_path.replace("game_", "gl_"), "ab", buffering=0)
            _nice = ["nice", "-n", os.environ.get("NBENV_NICE", "10")] if os.environ.get("NBENV_NICE", "10") != "0" else []
            self.glproc = subprocess.Popen(_nice + [NBGL, "--shm", self.gl_shm, "--size", self.size, "--hidden"], stdout=self.gllog,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            t0 = time.time()
            while not os.path.exists(self.gl_shm) or os.path.getsize(self.gl_shm) < 1 << 20:
                if time.time() - t0 > 20: raise InstanceDead("nbgl did not start")
                time.sleep(0.05)
            time.sleep(0.3)
        # NBHOST_WRAP: a command prefix (e.g. the host sampler); the emulator and its environment: qemu_cmd_env()
        qemu, env = qemu_cmd_env()
        # pctrain: game processes run at a lower CPU priority (NBENV_NICE, default 10) so the desktop stays responsive
        nice = ["nice", "-n", os.environ.get("NBENV_NICE", "10")] if os.environ.get("NBENV_NICE", "10") != "0" else []
        args = os.environ.get("NBHOST_WRAP", "").split() + nice + ["unshare", "-rn", qemu, NBHOST, "--libs", os.path.join(APK, "lib", "arm64-v8a"), "--assets", APK,
                "--payloads", PAYLOADS, "--stage", "boot", "--dpi", "280", "--xdpi", "409.432", "--ydpi", "406.4", "--safe-margins", "0,0,88,0", "--cores", str(self.cores), "--fast",
                "--autoplay", "3", "--run-seconds", "100000000", "--env", self.path, "--steps-per-frame", str(self.spf)]
        if self.gl_shm:
            args += ["--gl", self.gl_shm, "--env-client"]
        args += self.extra
        fds = ()
        if self.bell:
            self._close_bell()
            cmd_r, self.cmd_w = os.pipe(); self.resp_r, resp_w = os.pipe()
            os.set_blocking(self.resp_r, False); fds = (cmd_r, resp_w)
            args += ["--cmd-fd", str(cmd_r), "--resp-fd", str(resp_w)]
        self.logf = open(self.log_path, "ab", buffering=0)
        self.logf.write(f"\n==== start {time.strftime('%F %T')} {' '.join(args)}\n".encode())
        self.proc = subprocess.Popen(args, stdout=self.logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, pass_fds=fds)
        for fd in fds: os.close(fd)
        self.t_start = time.time(); self.starts += 1
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def ready(self):
        ok = self.hdr.magic == P.NBE_MAGIC and self.hdr.status in (P.ST_READY, P.ST_RUNNING)
        if ok and self.hdr.pix_cap != P.PIX_CAP:   # expertbot round: frame slot of the binary vs NB_PIX_CAP
            raise InstanceDead(f"nbhost frame slot {self.hdr.pix_cap} != NB_PIX_CAP {P.PIX_CAP} (binary {NBHOST})")
        return ok

    def wait_ready(self, timeout=300):
        t0 = time.time()
        while not self.ready():
            if not self.alive():
                raise InstanceDead(f"instance {self.idx} exited during boot (rc {self.proc.returncode}); log {self.log_path}")
            if time.time() - t0 > timeout:
                raise InstanceDead(f"instance {self.idx} not ready after {timeout} s; log {self.log_path}")
            time.sleep(0.2)
        return time.time() - self.t_start

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try: self.proc.wait(5)
            except Exception: pass
        self.proc = None
        if self.glproc is not None:
            if self.glproc.poll() is None:
                self.glproc.terminate()
                try: self.glproc.wait(5)
                except Exception: self.glproc.kill()
            self.glproc = None
            try: os.unlink(self.gl_shm)
            except Exception: pass
        try: self.logf.close()
        except Exception: pass

    def close(self):
        if self.proc is not None and self.proc.poll() is None and self.mm is not None and self.ready():
            try:
                self.send(P.CMD_QUIT); self.proc.wait(3)
            except Exception:
                pass
        self.kill(); self._close_bell()
        for o in ("hdr", "cmd", "resp", "state_i32", "map_i32", "pix_u8", "obs_f32"):
            if hasattr(self, o): delattr(self, o)
        if self.mm is not None:
            try: self.mm.close()
            except BufferError: pass
            self.mm = None
        try: os.unlink(self.path)
        except FileNotFoundError: pass

    # ------------------------------------------------------------------ protocol
    def send(self, ctype, ticks=4, flags=0, seed=0, ctl_mask=0, hold_mask=0, native_mask=0xfffffffe, obs_mask=0, actions=None,
             act_tick=200, gas=None, pix=(0, 0), storm_phase=None, intro_ai=False, pix_obs=False, div=None):
        """gas: RESET storm on (1, default) / off (0); STEP: None/0 = storm clock model, k+1 = force latched storm step k (replay).
        storm_phase (RESET): the storm latch phase 0..7 (None = derived from the seed).
        div (RESET, diversity round): {"location": id, "theme": id, "chr": {player: id} or [ids], "skin": {player: id} or [ids]}
        (global data ids, nbpc.diversity); None / missing keys = the default battle (written as zeros on every RESET)"""
        if gas is None:
            gas = 1 if ctype == P.CMD_RESET else 0
        c = self.cmd
        c.reserved[0] = (int(storm_phase) + 1) if (ctype == P.CMD_RESET and storm_phase is not None) else 0
        c.reserved[1] = 1 if (ctype == P.CMD_RESET and intro_ai) else 0
        c.reserved[2] = 1 if pix_obs else 0   # pixels as the policy observation (RGB top-left, PIL BOX in nbgl)
        c.reserved[3] = int(self.bell_spin_us)
        c.type, c.ticks, c.flags, c.seed = ctype, ticks, flags, seed
        c.ctl_mask, c.hold_mask, c.native_mask, c.obs_mask = ctl_mask, hold_mask, native_mask & 0xffffffff, obs_mask
        c.act_tick, c.gas, c.pix_w, c.pix_h = act_tick, gas, pix[0], pix[1]
        c.lead_mask, c.lead_dt, c.lead_speed, c.lead_frac = int(self.lead_mask) & 0xffffffff, float(self.lead_dt), float(self.lead_speed), float(self.lead_frac)
        if ctype == P.CMD_RESET:   # diversity round: the battle setup (zeros = default; an old game host ignores the field)
            set_div(c.div, div)
        if actions:
            for i, a in actions.items():
                c.act[i].move_dir, c.act[i].fire, c.act[i].aim_deg, c.act[i].super_dist = int(a[0]), int(a[1]), float(a[2]), float(a[3])
        self.seq = (self.seq + 1) & 0xffffffff
        self.hdr.cmd_seq = self.seq          # ctypes write = plain store; nbhost reads with acquire
        if self.cmd_w >= 0:
            try: os.write(self.cmd_w, b"\1")
            except OSError: pass             # the process is gone; alive() reports it
        return self.seq

    def _close_bell(self):
        for a in ("cmd_w", "resp_r"):
            fd = getattr(self, a, -1)
            if fd >= 0:
                try: os.close(fd)
                except OSError: pass
            setattr(self, a, -1)

    def drain_bell(self):
        if self.resp_r >= 0:
            try:
                while os.read(self.resp_r, 4096): pass
            except (BlockingIOError, OSError):
                pass

    def done_seq(self):
        return self.hdr.resp_seq == self.seq

    def wait(self, timeout=60.0):
        t0 = time.time(); n = 0
        while self.hdr.resp_seq != self.seq:
            n += 1
            if self.resp_r >= 0:                 # bell: block until nbhost rings; liveness every ~50 ms
                select.select([self.resp_r], [], [], 0.05); self.drain_bell()
                if self.hdr.resp_seq == self.seq: break
                if not self.alive():
                    raise InstanceDead(f"instance {self.idx} died (rc {self.proc.returncode}, crashed {self.hdr.crashed}); log {self.log_path}")
                if time.time() - t0 > timeout:
                    raise InstanceDead(f"instance {self.idx}: no response within {timeout} s; log {self.log_path}")
                continue
            if n > 200:
                time.sleep(0.0002)
                if n % 500 == 0:
                    if not self.alive():
                        raise InstanceDead(f"instance {self.idx} died (rc {self.proc.returncode}, crashed {self.hdr.crashed}); log {self.log_path}")
                    if time.time() - t0 > timeout:
                        raise InstanceDead(f"instance {self.idx}: no response within {timeout} s; log {self.log_path}")
        return self.resp

    def reset(self, seed, act_tick=200, gas=1, flags=P.F_STATE, native_mask=0xfffffffe, hold_mask=0, obs_mask=0x3ff, timeout=60.0,
              storm_phase=None, pix=(0, 0), intro_ai=False, pix_obs=False, div=None):
        self.send(P.CMD_RESET, flags=flags | P.F_MAP, seed=seed, act_tick=act_tick, gas=gas, native_mask=native_mask, hold_mask=hold_mask,
                  obs_mask=obs_mask, storm_phase=storm_phase, pix=pix, intro_ai=intro_ai, pix_obs=pix_obs, div=div)
        return self.wait(timeout)

    def supports_div(self):
        """the game host understands the diversity fields (Hdr.reserved[0] == 'DIV1')"""
        return self.hdr.reserved[0] == P.DIV_MAGIC

    def battle_info(self):
        """diversity round: the battle of the last RESET response as the game built it (location, theme, client location,
        variation, per-player character / skin data ids, rejected requests) + its tile map (int32 (H*W,), the format of
        data/map_showdown.npy), read from the response; None when the game host does not publish it"""
        r = self.resp
        if not r.div_valid:
            return None
        return {"location": int(r.div_location), "theme": int(r.div_theme), "cl_location": int(r.div_cl_location),
                "variation": int(r.div_variation), "err": int(r.div_err) & 0xffffffff,
                "chr": [int(x) for x in r.pl_char], "skin": [int(x) for x in r.pl_skin],
                "map_w": int(r.map_w), "map_h": int(r.map_h), "map_hash": int(r.map_hash), "cards_flag": int(r.div_reserved[0]),
                "gadget_mask": int(r.div_reserved[1]), "star_mask": int(r.div_reserved[2])}

    def step(self, actions=None, ticks=4, flags=P.F_STATE, ctl_mask=None, native_mask=0xfffffffe, hold_mask=0, obs_mask=0, timeout=60.0,
             storm=None, pix=(0, 0), pix_obs=False):
        """one env step; note the RESET's gas setting stays for the episode; storm = force the latched storm step (replay)"""
        if ctl_mask is None:
            ctl_mask = sum(1 << i for i in (actions or {}))
        self.send(P.CMD_STEP, ticks=ticks, flags=flags, ctl_mask=ctl_mask, native_mask=native_mask, hold_mask=hold_mask,
                  obs_mask=obs_mask, actions=actions, gas=(storm + 1) if storm is not None else 0, pix=pix, pix_obs=pix_obs)
        return self.wait(timeout)

    def setup(self, positions=None, ai_off=(), super_fill=(), unprotect=()):
        """phone test harness (debug_js_tick): teleport/stop/heal players {i: (x, y)}, clear the AI flag of players ai_off;
        super_fill (diversity round): fill these players' super charge; applied inside the next tick (send steps afterwards)"""
        c = self.cmd; c.set_mask = 0; c.ai_off_mask = 0
        set_div(c.div, {"setup_super_mask": sum(1 << int(i) for i in super_fill), "setup_unprotect_mask": sum(1 << int(i) for i in unprotect)})
        for i, (x, y) in (positions or {}).items():
            c.set_mask |= 1 << i; c.set_x[i] = int(x); c.set_y[i] = int(y)
        for i in ai_off:
            c.ai_off_mask |= 1 << i
        self.send(P.CMD_SETUP)
        return self.wait()

    def state_bytes(self):
        return self.state_i32[: self.resp.state_ints].tobytes()

    def pixels_obs(self):
        """the policy observation frame of the last response (cmd pix_obs): (pix_h, pix_w, 3) uint8 RGB top-left (view)"""
        r = self.resp
        return self.pix_u8[: r.pix_bytes].reshape(r.pix_h, r.pix_w, 3) if (r.pix_bytes and r.reserved[4] == 1) else None

    def pixels_rgba(self):
        """the captured frame of the last response: (pix_h, pix_w, 4) uint8, bottom-left origin (GL readback)"""
        r = self.resp
        return self.pix_u8[: r.pix_bytes].reshape(r.pix_h, r.pix_w, 4) if r.pix_bytes else None

    def map_tiles(self):
        return self.map_i32[: self.resp.map_w * self.resp.map_h].copy()
