"""Vectorised Null's Brawl environment on the PC: N nbhost processes (one battle each) behind shared memory.

    from nbpc.vecenv import NBVecEnvPC, NBEnvPC
    venv = NBVecEnvPC(10)                       # boots 10 instances (staggered), ~1.5-5 min
    obs, info = venv.reset(seeds=range(10))     # obs = {"vec": (N, 436), "grid": (N, 6, 25, 25)} float32
    obs, rew, term, trunc, info = venv.step(actions)    # actions (N, 4) = action space v3 per slot
    info["valid"]  # (N,) bool: False = no transition for this slot this call (its next episode / a restarted process
                   #            is still being prepared); info["new_episode"] (N,) marks the first obs of an episode.

Semantics per slot = inputs/host_code/nb_env.py (NBEnv): Solo Showdown, local Piper (player 0) vs 9 native bot Pipers,
4 ticks per step, the first observation after an empty 1-tick step at tick 200, reward = weighted sum of the logged
terms (REWARD_WEIGHTS, same names), termination on the native end or the local player's elimination, truncation
after max_episode_ticks. Differences (work/NOTES.md): resets are asynchronous (a slot is masked while its next battle is
built, ~0.3 s, instead of stalling the batch); crashed processes are restarted (counted in .stats) and the episode in
flight ends truncated with info["session_restart"] = True.
NBEnvPC wraps one slot as a plain gymnasium.Env with NBEnv's constructor arguments and spaces."""
import os, sys, time, json, select
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from . import proto as P
from .instance import Instance, InstanceDead

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import state2 as S2  # noqa: E402
import adb_driver as AD  # noqa: E402  (the phone's capture -> observation pipeline, used unchanged)

REWARD_WEIGHTS = {"damage_dealt": 1.0, "damage_taken": -0.5, "box_damage": 0.1, "cubes": 0.5, "kills": 2.0, "gas_ticks": -0.02,
                  "placement": 2.0, "win": 3.0, "step": 0.0, "gas_shaping": 0.0}
SHAPING_GAMMA = 0.995
HUD_MASK = [(1015, 25, 1160, 95)]      # nb_env.HUD_MASK (screen boxes on 2400x1080, blacked out after the resize)
CAP_W, CAP_H = 300, 135                # GPU capture = 2400x1080 / 8 (nbgl's three 2:1 blits), like the phone agent
VN, GS = S2.flat_size()


def action_v3_to_native(a, enable_super=True):
    m, fire, aim, dist = (int(x) for x in a)
    if fire == 2 and not enable_super:
        fire = 0
    return (m - 1 if m > 0 else -1, fire, float(aim), (dist + 1) / 8.0)


def action_v2_to_native(a, enable_super=True):
    """v2 [move 17, attack 18, super 18, gadget 2]; the auto-aimed taps (attack 1 / super 1) and the gadget have no
    in-loop equivalent (the client's aim assist) and are dropped (counted in stats['v2_dropped'])."""
    m, at, su, g = (int(x) for x in a)
    move = m - 1 if m > 0 else -1
    if enable_super and su >= 2:
        return (move, 2, (su - 2) * 22.5, 1.0), None
    if enable_super and su == 1:
        return (move, 0, 0.0, 1.0), "super_quick"
    if at >= 2:
        return (move, 1, (at - 2) * 22.5, 1.0), None
    if at == 1:
        return (move, 0, 0.0, 1.0), "attack_quick"
    return (move, 0, 0.0, 1.0), ("gadget" if g else None)


class _Slot:
    IDLE, BOOT, RESETTING, RUNNING, DEAD = range(5)

    def __init__(self, inst):
        self.inst, self.state = inst, _Slot.BOOT
        self.ep = None; self.phi = None; self.last_obs = None; self.seed = None; self.boot_t0 = time.time()
        self.local = 0; self.map_wh = (60, 60)


class NBVecEnvPC:
    def __init__(self, n=10, ticks_per_step=4, max_episode_ticks=2400, act_tick=200, reward_weights=None, gas=True,
                 local_ai=False, n_active_bots=9, obs_mode="state", action_version=3, enable_super=True, tag=None,
                 max_concurrent_boots=6, log_dir=None, seed0=0, state_bytes=False, verbose=False, extra_args=(), autoreset=True,
                 obs_size=(256, 115), frame_stack=4, hud_mask=HUD_MASK, local_intro_ai=None, async_frac=None):
        assert obs_mode in ("state", "pixels", "both"), obs_mode
        self.want_px = obs_mode != "state"; self.want_vec = obs_mode != "pixels"
        self.obs_size, self.k, self.mask = tuple(obs_size), int(frame_stack), hud_mask
        # the observation frame is made in nbgl (glhost/pilbox.h: the phone's 300x135 capture -> PIL-exact BOX -> RGB top-left,
        # verified bit-exact by test_pilbox.py); only the HUD mask is applied here. pixel_path="python" = the old path.
        self.pixel_path = "nbgl"
        W_, H_ = self.obs_size
        self._mask_sl = [(slice(int(np.floor(y0 * H_ / 1080.0)), int(np.ceil(y1 * H_ / 1080.0))),
                          slice(int(np.floor(x0 * W_ / 2400.0)), int(np.ceil(x1 * W_ / 2400.0)))) for (x0, y0, x1, y1) in (hud_mask or [])]
        # local_ai: on the phone the agent's local_ai switch persists across resets, so from a session's 2nd episode on the
        # local AI also plays the intro (T6) -> default on with local_ai
        self.local_intro_ai = bool(local_ai) if local_intro_ai is None else bool(local_intro_ai)
        # async_frac (None = synchronous): step() returns as soon as this fraction of the in-flight slots finished; slots
        # still computing stay in flight (valid False) and do not take the next call's actions (info["sent"] marks the
        # slots that received this call's actions; a transition of slot i completes in the call where valid[i]).
        self.async_frac = async_frac
        self.bell_spin_s = float(os.environ.get("NBENV_BELL_SPIN_US", "200")) * 1e-6   # bell pipes: spin before blocking
        self.async_spin_us = int(os.environ.get("NBENV_ASYNC_SPIN_US", "5000"))
        self.py_block_async = os.environ.get("NBENV_PY_BLOCK_ASYNC", "0") == "1"   # with async_frac: Python blocks on the bell too   # with async_frac: how long instances spin for the next command
        # a reset that does not finish is a hang (pixel mode: a rare loading deadlock in the game, ~1 per 600-900 resets,
        # runs/t5_crash/): the process is replaced. Normal resets: 0.3 s (state), 2-4 s (pixels, loaded machine).
        self.reset_timeout = 30.0 if self.want_px else 60.0
        self.n, self.ticks, self.max_ticks, self.act_tick = n, ticks_per_step, max_episode_ticks, act_tick
        self.w = dict(REWARD_WEIGHTS); self.w.update(reward_weights or {})
        self.gas, self.local_ai, self.n_active, self.obs_mode = gas, local_ai, int(n_active_bots), obs_mode
        self.action_version, self.enable_super, self.verbose, self.want_state = int(action_version), enable_super, verbose, state_bytes
        self.tag = tag or f"v{os.getpid()}"; self.autoreset = autoreset
        self.max_boots = max_concurrent_boots
        st = {"vec": spaces.Box(-np.inf, np.inf, (VN,), np.float32), "grid": spaces.Box(0, 255, GS, np.float32)}
        pix = spaces.Box(0, 255, (self.obs_size[1], self.obs_size[0], 3 * self.k), np.uint8)
        self.single_observation_space = spaces.Dict(st) if obs_mode == "state" else (pix if obs_mode == "pixels" else spaces.Dict({"pixels": pix, **st}))
        self.single_action_space = spaces.MultiDiscrete([17, 3, 360, 8] if self.action_version == 3 else [17, 18, 18, 2])
        self.stats = {"restarts": 0, "episodes": 0, "steps": 0, "resets_s": [], "boot_s": [], "v2_dropped": {}, "crash_log": []}
        self._rng = np.random.default_rng(seed0)
        self._seed_queue = []
        # bell pipes on every instance (NBENV_BELL=0: off). Synchronous steps block on them (+18 % at 10 instances); with
        # async_frac the instances spin 5 ms and Python spins (blocking costs latency there): runs/opt2/sweep_*.json
        bell = os.environ.get("NBENV_BELL", "1") == "1"
        self.slots = [_Slot(Instance(i, tag=self.tag, log_dir=log_dir, extra_args=extra_args, gl=self.want_px, bell=bell)) for i in range(n)]
        self.obs_vec = np.zeros((n, VN), np.float32); self.obs_grid = np.zeros((n,) + GS, np.float32)
        self.obs_pix = np.zeros((n, self.obs_size[1], self.obs_size[0], 3 * self.k), np.uint8)
        self.stats["no_pixels"] = 0; self.stats["t_pre"] = 0.0
        self._boot_all()

    # ------------------------------------------------------------------ processes
    def _boot_all(self):
        pending = list(range(self.n)); booting = []
        t0 = time.time()
        while pending or booting:
            while pending and len(booting) < self.max_boots:
                i = pending.pop(0); s = self.slots[i]; s.inst.start(); s.boot_t0 = time.time(); booting.append(i)
            for i in list(booting):
                s = self.slots[i]
                if s.inst.ready():
                    booting.remove(i); s.state = _Slot.IDLE; self.stats["boot_s"].append(round(time.time() - s.boot_t0, 1))
                elif not s.inst.alive() or time.time() - s.boot_t0 > 400:
                    booting.remove(i); self._log_crash(i, "boot failed"); s.inst.kill(); pending.append(i)
            time.sleep(0.2)
        if self.verbose:
            print(f"[vecenv] {self.n} instances ready in {time.time() - t0:.0f} s (boot {self.stats['boot_s']})", flush=True)

    def _log_crash(self, i, why):
        s = self.slots[i]; c = s.inst.hdr.crashed if hasattr(s.inst, "hdr") else None
        row = {"t": time.strftime("%F %T"), "slot": i, "why": why, "signal": c, "log": s.inst.log_path, "episode_steps": s.ep["len"] if s.ep else None}
        self.stats["crash_log"].append(row)
        with open(os.path.join(s.inst.log_dir, f"crashes_{self.tag}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        if self.verbose:
            print("[vecenv] instance problem:", row, flush=True)

    def _restart(self, i):
        s = self.slots[i]; self.stats["restarts"] += 1
        try:   # a hung (alive) process: ask nbhost to log every thread's stack first (SIGUSR2, main.c on_dump)
            if s.inst.alive():
                import signal as _sg; os.kill(s.inst.proc.pid, _sg.SIGUSR2); time.sleep(1.0)
        except Exception:
            pass
        try: s.inst.close()
        except Exception: pass
        s.inst.start(); s.boot_t0 = time.time(); s.state = _Slot.DEAD

    def close(self):
        for s in self.slots:
            try: s.inst.close()
            except Exception: pass

    # ------------------------------------------------------------------ helpers
    def _masks(self):
        native = 0xfffffffe | (1 if self.local_ai else 0)
        hold = 0
        for i in range(1, 10):
            if i > self.n_active:
                hold |= 1 << i; native &= ~(1 << i)
        return native, hold

    def _next_seed(self):
        return self._seed_queue.pop(0) if self._seed_queue else int(self._rng.integers(0, 2 ** 31 - 1))

    def _send_reset(self, i, seed=None):
        s = self.slots[i]; native, hold = self._masks()
        s.seed = self._next_seed() if seed is None else int(seed) & 0x7fffffff
        # diversity round: optional battle setup per reset (self.div_fn(seed, slot) -> nbpc.diversity div dict or None)
        div = self.div_fn(s.seed, i) if getattr(self, "div_fn", None) else None
        s.inst.send(P.CMD_RESET, flags=self._flags() | P.F_MAP, seed=s.seed, act_tick=self.act_tick, gas=1 if self.gas else 0,
                    native_mask=native, hold_mask=hold, obs_mask=1, intro_ai=self.local_intro_ai, div=div, **self._pix_args())
        s.state = _Slot.RESETTING; s.reset_t0 = time.time()

    def _pix_args(self):
        if self.pixel_path == "nbgl":
            return {"pix": self.obs_size, "pix_obs": True}
        return {"pix": (CAP_W, CAP_H)}

    def _flags(self):
        return (P.F_OBS if self.want_vec else 0) | (P.F_STATE if self.want_state else 0) | (P.F_PIXELS if self.want_px else 0)

    def _obs_out(self):
        st = {"vec": self.obs_vec.copy(), "grid": self.obs_grid.copy()}
        if self.obs_mode == "state":
            return st
        return self.obs_pix.copy() if self.obs_mode == "pixels" else {"pixels": self.obs_pix.copy(), **st}

    def _gas_phi(self, r, prev):
        li = r.local_index; me = r.pl[li]
        if not me.present or me.hp <= 0:
            return prev
        W, H = r.map_w or 60, r.map_h or 60
        tx, ty = int(me.x) // S2.TILE, int(me.y) // S2.TILE
        d = min(tx, ty, W - 1 - tx, H - 1 - ty)
        return -float(max(0, S2.gas_step(r.tick) + 2 - d))

    def _read_obs(self, i, r, first=False):
        s = self.slots[i]; off = 0
        if self.want_px:
            t0 = time.perf_counter(); c = 3 * self.k
            if r.reserved[4] == 1 and r.pix_w == self.obs_size[0] and r.pix_h == self.obs_size[1]:
                f = s.inst.pixels_obs().copy()
                for sl in self._mask_sl:
                    f[sl] = 0
            elif r.pix_bytes == CAP_W * CAP_H * 4:
                f = AD.preprocess_train(s.inst.pix_u8[:r.pix_bytes].tobytes(), self.obs_size, self.mask)
            else:   # no frame for this step (should not happen; counted)
                self.stats["no_pixels"] += 1; f = self.obs_pix[i, :, :, c - 3:c]
            if first:
                self.obs_pix[i] = np.tile(f, (1, 1, self.k))
            else:
                self.obs_pix[i, :, :, :c - 3] = self.obs_pix[i, :, :, 3:].copy(); self.obs_pix[i, :, :, c - 3:] = f
            self.stats["t_pre"] += time.perf_counter() - t0
        if not self.want_vec:
            return
        if (r.obs_written >> r.local_index) & 1:
            off = r.local_index * (P.OBS_VEC + P.OBS_GRID)
            self.obs_vec[i] = s.inst.obs_f32[off:off + P.OBS_VEC]
            self.obs_grid[i] = s.inst.obs_f32[off + P.OBS_VEC: off + P.OBS_VEC + P.OBS_GRID].reshape(GS)
        else:
            self.obs_vec[i] = 0; self.obs_grid[i] = 0

    def _start_episode(self, i, r):
        s = self.slots[i]
        s.ep = {"start_tick": r.tick, "seed": s.seed, "ret": 0.0, "len": 0, "terms": {}}
        s.phi = None; s.phi = self._gas_phi(r, None)
        s.state = _Slot.RUNNING; s.local = r.local_index; s.map_wh = (r.map_w, r.map_h)
        s.battle_info = s.inst.battle_info()   # diversity round: location / theme / roster as built (None: old game host)
        self.stats["episodes"] += 1; self.stats["resets_s"].append(round(time.time() - s.reset_t0, 3))
        self._read_obs(i, r, first=True)

    # ------------------------------------------------------------------ gym-like API
    def reset(self, seeds=None):
        """reset every slot; blocks until all have their first observation"""
        if seeds is not None:
            self._seed_queue = [int(x) for x in seeds]
        for i, s in enumerate(self.slots):   # async mode: let steps in flight finish (one command per instance)
            if getattr(s, "in_flight", False):
                t0 = time.time()
                while not s.inst.done_seq() and s.inst.alive() and time.time() - t0 < 60:
                    time.sleep(0.0005)
                s.in_flight = False
        for i in range(self.n):
            if self.slots[i].state in (_Slot.IDLE, _Slot.RUNNING):
                self._send_reset(i)
        infos = {"valid": np.zeros(self.n, bool), "seed": np.zeros(self.n, np.int64)}
        while True:
            self._poll_background(block=True)
            ok = all(s.state == _Slot.RUNNING for s in self.slots)
            if ok:
                break
        for i, s in enumerate(self.slots):
            infos["valid"][i] = True; infos["seed"][i] = s.seed
        return self._obs_out(), infos

    def _poll_background(self, block=False):
        """advance RESETTING / DEAD slots; returns the slots whose new episode started now"""
        started = []
        for i, s in enumerate(self.slots):
            if s.state == _Slot.RESETTING:
                if s.inst.done_seq():
                    r = s.inst.resp
                    if r.status != 0:
                        self._log_crash(i, f"reset status {r.status} {r.err!r}"); self._restart(i); continue
                    self._start_episode(i, r); started.append(i)
                elif not s.inst.alive():
                    self._log_crash(i, "died during reset"); self._restart(i)
                elif time.time() - s.reset_t0 > self.reset_timeout:
                    self._log_crash(i, "reset timeout"); self._restart(i)
            elif s.state == _Slot.DEAD:
                if s.inst.ready():
                    self.stats["boot_s"].append(round(time.time() - s.boot_t0, 1)); self._send_reset(i)
                elif not s.inst.alive() or time.time() - s.boot_t0 > 400:
                    self._log_crash(i, "reboot failed"); self._restart(i)
        if block and not started:
            time.sleep(0.0005)
        return started

    def step(self, actions):
        actions = np.asarray(actions)
        rew = np.zeros(self.n, np.float32); term = np.zeros(self.n, bool); trunc = np.zeros(self.n, bool)
        valid = np.zeros(self.n, bool); new_ep = np.zeros(self.n, bool)
        infos = {"valid": valid, "new_episode": new_ep, "episode": [None] * self.n, "terms": [None] * self.n,
                 "session_restart": np.zeros(self.n, bool), "tick": np.zeros(self.n, np.int32), "rank": np.zeros(self.n, np.int32),
                 "local_xy_hp": np.zeros((self.n, 3), np.int32)}
        native, hold = self._masks()
        sent = []; infos["sent"] = np.zeros(self.n, bool)
        for i, s in enumerate(self.slots):
            if s.state != _Slot.RUNNING or getattr(s, "in_flight", False):
                continue
            if self.local_ai:
                acts, ctl = None, 0
            else:
                if self.action_version == 3:
                    na = action_v3_to_native(actions[i], self.enable_super)
                else:
                    na, drop = action_v2_to_native(actions[i], self.enable_super)
                    if drop: self.stats["v2_dropped"][drop] = self.stats["v2_dropped"].get(drop, 0) + 1
                acts, ctl = {s.local: na}, 1 << s.local
            s.inst.bell_spin_us = 0 if self.async_frac is None else self.async_spin_us
            s.inst.send(P.CMD_STEP, ticks=self.ticks, flags=self._flags(), ctl_mask=ctl,
                        native_mask=native, hold_mask=hold, obs_mask=1 << s.local, actions=acts, **self._pix_args())
            s.t_sent = time.time(); sent.append(i); s.in_flight = True; infos["sent"][i] = True
        pending = set(i for i, s in enumerate(self.slots) if getattr(s, "in_flight", False))
        need = len(pending) if self.async_frac is None else max(1, int(np.ceil(self.async_frac * len(pending))))
        done_now = 0; spins = 0; t_wait0 = time.time()
        # profile (runs/bench/prof_vecenv.txt): checking alive() (= waitpid) on every spin was most of the Python time;
        # process death / timeouts and the background slots are now checked every 256 spins
        while pending and done_now < need:
            spins += 1; slow = (spins & 255) == 0; got = False
            for i in list(pending):
                s = self.slots[i]
                if s.inst.done_seq():
                    got = True
                    pending.discard(i); s.in_flight = False; done_now += 1; self._finish_step(i, rew, term, trunc, valid, infos)
                elif slow and (not s.inst.alive() or time.time() - s.t_sent > 60):
                    pending.discard(i); s.in_flight = False; done_now += 1
                    self._log_crash(i, "died during step" if not s.inst.alive() else "step timeout")
                    trunc[i] = True; valid[i] = True; infos["session_restart"][i] = True
                    infos["episode"][i] = self._episode_summary(s, None); self._restart(i)
            if pending and slow:
                self._poll_background()
            if got:
                t_wait0 = time.time()
            if not got:
                fds = [self.slots[i].inst.resp_r for i in pending if self.slots[i].inst.resp_r >= 0]
                spin_s = self.bell_spin_s if (self.async_frac is None or self.py_block_async) else 1e9
                if fds and time.time() - t_wait0 < spin_s:
                    os.sched_yield()   # bell: spin a little first (a response is often just about to come)
                elif fds:   # bell pipes: sleep until a pending slot rings (or 2 ms, for the slow checks)
                    rd = select.select(fds, [], [], 0.002)[0]
                    for i in pending:
                        if self.slots[i].inst.resp_r in rd: self.slots[i].inst.drain_bell()
                else:
                    os.sched_yield()   # leave the hardware thread to the game processes (8 cores / 16 threads)
        for i in self._poll_background():
            new_ep[i] = True; valid[i] = True
        self.stats["steps"] += int(valid.sum() - new_ep.sum())
        return self._obs_out(), rew, term, trunc, infos

    def _episode_summary(self, s, r):
        ep = s.ep or {}
        return {"r": ep.get("ret"), "l": ep.get("len"), "rank": ep.get("rank"), "terms": dict(ep.get("terms", {})),
                "ticks": (r.tick - ep["start_tick"]) if (r is not None and ep) else None, "seed": ep.get("seed")}

    def _finish_step(self, i, rew, term, trunc, valid, infos):
        s = self.slots[i]; r = s.inst.resp
        if r.status != 0:
            self._log_crash(i, f"step status {r.status} {r.err!r}")
            trunc[i] = True; valid[i] = True; infos["episode"][i] = self._episode_summary(s, r)
            if self.autoreset: self._send_reset(i)
            else: s.state = _Slot.IDLE
            return
        li = r.local_index; me = r.pl[li]
        self._read_obs(i, r)
        dead = (not me.alive) or bool(me.eliminated)
        terminated = bool(r.done or dead)
        truncated = bool(not terminated and r.tick - s.ep["start_tick"] >= self.max_ticks)
        phi1 = self._gas_phi(r, s.phi)
        shaping = (SHAPING_GAMMA * phi1 - s.phi) if (phi1 is not None and s.phi is not None) else 0.0
        s.phi = phi1
        t = {"damage_dealt": me.damage_dealt / 1000.0, "damage_taken": me.damage_taken / 1000.0, "box_damage": me.box_damage / 1000.0,
             "cubes": float(me.cubes), "kills": float(me.kills), "gas_ticks": float(me.gas_ticks), "placement": 0.0, "win": 0.0,
             "step": 1.0, "gas_shaping": float(shaping)}
        if terminated:
            rank = r.local_rank if r.local_rank > 0 else (r.result if r.result > 0 else (1 + int(r.alive or 0) if dead else 1))
            t["placement"] = (10 - rank) / 9.0; t["win"] = 1.0 if rank == 1 else 0.0; s.ep["rank"] = rank
            infos["rank"][i] = rank
        reward = float(sum(self.w[k] * v for k, v in t.items()))
        s.ep["ret"] += reward; s.ep["len"] += 1
        for k, v in t.items():
            s.ep["terms"][k] = s.ep["terms"].get(k, 0.0) + v
        rew[i] = reward; term[i] = terminated; trunc[i] = truncated; valid[i] = True
        infos["terms"][i] = t; infos["tick"][i] = r.tick; infos["local_xy_hp"][i] = (me.x, me.y, me.hp)
        if terminated or truncated:
            infos["episode"][i] = self._episode_summary(s, r)
            if self.autoreset: self._send_reset(i)
            else: s.state = _Slot.IDLE


class NBEnvPC(gym.Env):
    """one slot of NBVecEnvPC as a gymnasium.Env (NBEnv-compatible constructor, spaces, info keys)"""
    metadata = {"render_modes": []}

    def __init__(self, ticks_per_step=4, max_episode_ticks=2400, act_tick=200, reward_weights=None, gas=True, obs_mode="state",
                 local_ai=False, n_active_bots=9, action_version=3, enable_super=True, tag=None, verbose=False, obs_size=(256, 115),
                 frame_stack=4, hud_mask=HUD_MASK, **_ignored):
        super().__init__()
        self.v = NBVecEnvPC(1, ticks_per_step, max_episode_ticks, act_tick, reward_weights, gas, local_ai, n_active_bots, obs_mode,
                            action_version, enable_super, tag=tag, verbose=verbose, autoreset=False, obs_size=obs_size,
                            frame_stack=frame_stack, hud_mask=hud_mask)
        self.observation_space, self.action_space = self.v.single_observation_space, self.v.single_action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        o, inf = self.v.reset(seeds=[seed] if seed is not None else None)
        return self._one(o), {"seed": int(inf["seed"][0])}

    @staticmethod
    def _one(o):
        return o[0] if isinstance(o, np.ndarray) else {k: v[0] for k, v in o.items()}

    def step(self, action):
        o, r, te, tr, inf = self.v.step(np.asarray(action)[None])
        info = {"terms": inf["terms"][0], "tick": int(inf["tick"][0]), "rank": int(inf["rank"][0]) or None, "episode": inf["episode"][0],
                "session_restart": bool(inf["session_restart"][0])}
        return self._one(o), float(r[0]), bool(te[0]), bool(tr[0]), info

    def close(self):
        self.v.close()
