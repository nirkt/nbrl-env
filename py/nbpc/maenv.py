"""Asynchronous multi-agent environment (pctrain round): N nbhost instances (one Solo Showdown battle each); in every battle a
set of players is policy-controlled ("agents"), the others are driven by the native AI (optionally with "native + lead" aim,
work/loader lead_aim). Instances step independently: poll() returns the agents whose instance finished its step (their new
observations + the finished transitions), send() gives them actions; GPU inference for one group overlaps with the other
instances' simulation (no batch barrier).

    env = MAEnv(12, battle_fn=..., reward_w=REWARD)
    obs, ids, trans = env.poll()          # obs {"vec": (B, 436) f32, "grid": (B, 6, 25, 25) u8}, ids (B,) agent ids
    env.send(ids, actions)                # actions (B, 4) v3: move 17, fire 3, aim 360, super-distance 8
battle_fn(rng, inst) -> dict(seed, ctl [player indices], watch [native players whose episodes are logged too], lead_mask,
  end: "agents" | "player0" | "done", tag (anything))
  end "agents": the battle is reset once every controlled player is eliminated (their ranks are final; the natives' outcome
  does not matter); "player0": once player 0 is eliminated (the 1-vs-9 protocol, also with player 0 native); "done": at
  the battle's end.
Agent ids: unique per (battle, player): (battle counter << 4) | player.
Rewards per agent and step: sum of REWARD_W[k] * term[k], terms (logged): dealt (damage dealt to brawlers + leap bombs,
/1000), taken (/1000, storm included), hp (HP change /1000: damage, regen, cubes), box (/1000), cubes, kills, hits, shots
(ammo drops), gas (storm ticks), place ((10 - rank) / 9 at elimination / end), win.
Observations: the C encoder's state2 vector + grid of each agent, made fair (nbpc.fair) unless fair=False.
Episode rows (env.episodes, appended when an agent's episode ends): seed, player, rank, win, ticks alive, steps, return,
summed terms, storm death, truncated, tag."""
import os, time, json, select
import numpy as np
from . import proto as P
from .instance import Instance, InstanceDead
from . import fair as FAIR

NV, NG = P.OBS_VEC, P.OBS_GRID
GS = (6, 25, 25)
REWARD_W = {"dealt": 0.5, "taken": 0.0, "hp": 1.0, "box": 0.05, "cubes": 0.5, "kills": 1.0, "hits": 0.0, "shots": 0.0, "gas": -0.02,
            "place": 5.0, "win": 3.0}
TERMS = list(REWARD_W)


def action_to_native(a):
    """v3 action -> nbhost NbeAction (move_dir -1 = release, 0..15; fire 0/1/2; aim degrees; leap fraction). The super
    distance bin k executes the bin CENTRE (k + 0.5) / 8 of 2600 (research F1.5: the build round's (k + 1) / 8 = the upper
    edge made converted leaps ~162 units too long)."""
    m, f, aim, d = (int(x) for x in a)
    return (m - 1 if m > 0 else -1, f, float(aim), (d + 0.5) / 8.0)


def mem_available_mb():
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"): return int(l.split()[1]) // 1024
    except Exception:
        pass
    return 1 << 30


def vram_used_mb():
    try:
        return int(open("/sys/class/drm/card1/device/mem_info_vram_used").read()) // (1 << 20)
    except Exception:
        return 0


class MemoryGuard(RuntimeError):
    """raised by MAEnv.poll when the machine runs out of RAM / VRAM (the trainers checkpoint and close the env)"""


# pctrain (2026-10-05, a PC freeze: 10 client-mode instances ~15 GB VRAM + a 6 GB GPU replay spilled into RAM): limits
MIN_AVAIL_MB_BOOT = int(os.environ.get("NBENV_MIN_AVAIL_MB_BOOT", "3000"))   # do not start another instance below this
MIN_AVAIL_MB_RUN = int(os.environ.get("NBENV_MIN_AVAIL_MB_RUN", "1200"))     # abort the run (checkpoint + close) below this
MAX_VRAM_MB_RUN = int(os.environ.get("NBENV_MAX_VRAM_MB", "15700"))


class _Inst:
    def __init__(self, inst):
        self.inst = inst; self.state = "boot"; self.cfg = None; self.agents = {}; self.battle = -1; self.t_sent = 0.0
        self.prev = {}; self.ep = {}; self.acts = {}; self.boot_t0 = time.time(); self.start_tick = 0; self.W = 60; self.H = 60


class MAEnv:
    def __init__(self, n, battle_fn, reward_w=None, max_ticks=4000, fair=True, tag=None, extra_args=(), verbose=False,
                 lead_dt=0.0, lead_speed=197.0, log_dir=None, seed=0, max_boots=6, bell_spin_us=100, obs_players=None, want_full=False,
                 pixels=False, frame_stack=4, want_state=False, ticks=4, obs_size=(256, 115)):
        self.n, self.battle_fn, self.max_ticks, self.fair, self.verbose = n, battle_fn, int(max_ticks), bool(fair), verbose
        self.want_full = bool(want_full)   # also keep the unfair observation (asymmetric critic): env.last_full after poll()
        self.w = dict(REWARD_W); self.w.update(reward_w or {})
        self.tag = tag or f"ma{os.getpid()}"; self.rng = np.random.default_rng(seed)
        # pixels=True: client mode (real client + nbgl renderer per instance); the LOCAL player 0's 256x115 view (the phone's
        # preprocess, HUD mask applied) is returned as a frame stack in obs["pixels"] (B, 115, 256, 3k) for agents that are
        # player 0; want_state: the state2 bytes + camera of every gathered instance (labels.py) in env.last_state
        self.pixels, self.k, self.want_state = bool(pixels), int(frame_stack), bool(want_state) or bool(pixels)
        # ticks per env step (4 = 5 decisions/s; 1 = 20/s, the game's tick rate). obs_size (W, H): (256, 115) = the phone
        # pipeline's observation (300x135 capture -> PIL BOX in nbgl, bit-exact to the phone); any other size = the GPU
        # box-downscale of the full 2400x1080 frame straight to (W, H) (raw RGBA readback, flipped here), e.g. 512x230
        self.ticks, self.obs_size = int(ticks), (int(obs_size[0]), int(obs_size[1]))
        self.phone_obs = self.obs_size == (256, 115)
        self.I = [_Inst(Instance(i, tag=self.tag, log_dir=log_dir, extra_args=extra_args, bell=True, gl=self.pixels)) for i in range(n)]
        if self.pixels:
            from .vecenv import HUD_MASK
            W_, H_ = self.obs_size
            self._mask_sl = [(slice(int(np.floor(y0 * H_ / 1080.0)), int(np.ceil(y1 * H_ / 1080.0))),
                              slice(int(np.floor(x0 * W_ / 2400.0)), int(np.ceil(x1 * W_ / 2400.0)))) for (x0, y0, x1, y1) in HUD_MASK]
            for s in self.I: s.stack = np.zeros((H_, W_, 3 * self.k), np.uint8)
        for s in self.I:
            s.inst.lead_dt, s.inst.lead_speed = float(lead_dt), float(lead_speed); s.inst.bell_spin_us = int(bell_spin_us)
        self.battles = 0; self.episodes = []; self.stats = {"restarts": 0, "battles": 0, "agent_steps": 0, "env_steps": 0, "boot_s": []}
        self.ready = []          # (inst index) with a finished step whose agents wait for actions
        self._trans = []; self._waiting = {}; self.dropped = []
        self._boot(max_boots)

    # ------------------------------------------------------------------ processes
    def _boot(self, max_boots):
        pending, booting, t0 = list(range(self.n)), [], time.time()
        while pending or booting:
            while pending and len(booting) < max_boots:
                if mem_available_mb() < MIN_AVAIL_MB_BOOT:
                    if not booting:
                        raise MemoryGuard(f"only {mem_available_mb()} MB RAM available: not booting instance {pending[0]} "
                                          f"({self.n - len(pending)} of {self.n} started)")
                    break
                i = pending.pop(0); self.I[i].inst.start(); self.I[i].boot_t0 = time.time(); booting.append(i)
            for i in list(booting):
                s = self.I[i]
                if s.inst.ready():
                    booting.remove(i); self.stats["boot_s"].append(round(time.time() - s.boot_t0, 1)); self._reset(i)
                elif not s.inst.alive() or time.time() - s.boot_t0 > 400:
                    booting.remove(i); s.inst.kill(); pending.append(i)
            time.sleep(0.1)
        if self.verbose:
            print(f"[maenv] {self.n} instances ready in {time.time() - t0:.0f} s", flush=True)

    def _restart(self, i, why):
        s = self.I[i]; self.stats["restarts"] += 1
        if self.verbose:
            print(f"[maenv] instance {i}: {why}; restarting", flush=True)
        for p in list(s.agents):   # episodes in flight end truncated (no transition reward)
            self._end_agent(i, p, None, truncated=True, restart=True)
        try: s.inst.close()
        except Exception: pass
        s.inst.start(); s.boot_t0 = time.time(); s.state = "boot"

    def close(self):
        for s in self.I:
            try: s.inst.close()
            except Exception: pass

    # ------------------------------------------------------------------ battles
    def _reset(self, i):
        s = self.I[i]; cfg = self.battle_fn(self.rng, i)
        if cfg is None:   # nothing left to run on this instance
            s.state = "idle"; s.agents = {}; s.ep = {}; s.prev = {}; return
        cfg = dict(cfg); s.cfg = cfg
        s.ctl = sorted(int(p) for p in cfg.get("ctl", []))
        s.watch = sorted(set(int(p) for p in cfg.get("watch", [])) - set(s.ctl))
        s.ctl_mask = sum(1 << p for p in s.ctl); s.native = 0xffffffff & ~s.ctl_mask
        s.send_mask = s.ctl_mask
        if cfg.get("native_ctl"):   # observe the agents but let their native AI play (recording the native AI)
            s.native = 0xffffffff; s.send_mask = 0
        s.inst.lead_mask = int(cfg.get("lead_mask", 0)) & ~s.ctl_mask
        if "lead_dt" in cfg: s.inst.lead_dt = float(cfg["lead_dt"])
        if "lead_frac" in cfg: s.inst.lead_frac = float(cfg["lead_frac"])
        s.obs_mask = s.ctl_mask
        # diversity round: cfg["div"] = the battle setup (nbpc.diversity.make_div / random_div; None = the default battle)
        s.inst.send(P.CMD_RESET, flags=self._flags() | P.F_MAP, seed=int(cfg["seed"]) & 0x7fffffff, act_tick=200, gas=1,
                    native_mask=s.native, obs_mask=s.obs_mask, intro_ai=True, div=cfg.get("div"), **self._pix())
        s.first_frame = True
        s.state = "resetting"; s.t_sent = time.time()
        self.battles += 1; s.battle = self.battles; s.agents = {}; s.prev = {}; s.ep = {}; s.acts = {}

    def _end_agent(self, i, p, r, truncated=False, restart=False, rank=None):
        s = self.I[i]; ep = s.ep.pop(p, None); aid = s.agents.pop(p, None); s.prev.pop(p, None)
        if aid is not None and (restart or (truncated and rank is None)):
            self.dropped.append(aid)   # its last action gets no transition (the trainer bootstraps it)
        if not s.ep: s.tracking_done = True
        if ep is None:
            return
        row = {"seed": int(s.cfg["seed"]), "player": p, "rank": rank, "win": int(rank == 1) if rank is not None else None,
               "ticks": (r.tick - s.start_tick) if r is not None else None, "steps": ep["steps"], "ret": ep["ret"],
               "terms": {k: round(v, 4) for k, v in ep["terms"].items()}, "storm_death": int(r.pl[p].elim_gas) if (r is not None and r.pl[p].eliminated) else 0,
               "truncated": bool(truncated), "restart": bool(restart), "tag": s.cfg.get("tag"), "ctl": s.ctl, "lead_mask": s.inst.lead_mask,
               "attack_steps": ep["attack_steps"], "super_steps": ep["super_steps"], "move_steps": ep["move_steps"]}
        self.episodes.append(row)

    def _start_battle(self, i, r):
        s = self.I[i]; s.state = "running"; s.start_tick = r.tick; s.W, s.H = (r.map_w or 60), (r.map_h or 60)
        s.battle_info = s.inst.battle_info()   # diversity round: what the game built (location, theme, roster, map hash)
        s.map_tiles = s.inst.map_tiles() if r.map_written else None
        if r.map_written and int(r.map_hash) != 3654183511 and not s.cfg.get("div"):   # default battles: the fixed map (data/map_showdown.npy)
            self.stats["map_mismatch"] = self.stats.get("map_mismatch", 0) + 1
            if self.verbose: print(f"[maenv] WARNING: map hash {r.map_hash} differs from the Showdown map", flush=True)
        self.stats["battles"] += 1
        for p in s.ctl + s.watch:
            if r.pl[p].alive:
                if p in s.ctl:
                    s.agents[p] = (s.battle << 4) | p
                s.prev[p] = (r.pl[p].hp, r.pl[p].ammo)
                s.ep[p] = {"steps": 0, "ret": 0.0, "terms": {k: 0.0 for k in TERMS}, "attack_steps": 0, "super_steps": 0, "move_steps": 0}

    def _flags(self):
        return P.F_OBS | (P.F_PIXELS if self.pixels else 0) | (P.F_STATE if self.want_state else 0)

    def _pix(self):
        return {"pix": self.obs_size, "pix_obs": self.phone_obs} if self.pixels else {}

    def _frame(self, i, r):
        """update instance i's frame stack from this response (pixel mode)"""
        s = self.I[i]
        if not self.pixels:
            return
        f = s.inst.pixels_obs() if self.phone_obs else s.inst.pixels_rgba()
        if f is None or (r.pix_w, r.pix_h) != self.obs_size:
            self.stats["no_pixels"] = self.stats.get("no_pixels", 0) + 1; return
        f = f.copy() if self.phone_obs else np.ascontiguousarray(f[::-1, :, :3])   # GL readback: bottom-left origin
        for sl in self._mask_sl: f[sl] = 0
        if getattr(s, "first_frame", False):
            s.stack[:] = np.tile(f, (1, 1, self.k)); s.first_frame = False
        else:
            s.stack[:, :, :-3] = s.stack[:, :, 3:]; s.stack[:, :, -3:] = f

    # ------------------------------------------------------------------ the loop
    def _check(self, i):
        """instance i finished its command: process it; returns True if its agents now wait for actions"""
        s = self.I[i]; r = s.inst.resp
        if s.state == "resetting":
            if r.status != 0:
                self._restart(i, f"reset status {r.status} {r.err!r}"); return False
            self._start_battle(i, r); self._frame(i, r)
        elif s.state == "stepping":
            if r.status != 0:
                self._restart(i, f"step status {r.status} {r.err!r}"); return False
            self._frame(i, r); self._finish_step(i, r)
        if s.state == "running":
            if not s.agents:   # no agent left (or none): keep stepping natively until the battle's end condition
                if self._battle_over(i, r):
                    self._reset(i); return False
                self._send_step(i, {}); return False
            return True
        return False

    def _battle_over(self, i, r):
        s = self.I[i]; end = s.cfg.get("end", "agents")
        if r.done or r.tick - s.start_tick >= self.max_ticks:
            return True
        if end == "player0":
            return bool(r.pl[0].eliminated)
        if end == "agents":
            return not s.agents
        return False

    def _finish_step(self, i, r):
        s = self.I[i]; s.state = "running"; w = self.w
        self.stats["env_steps"] += 1
        done = bool(r.done); trunc = (not done) and (r.tick - s.start_tick >= self.max_ticks)
        alive_now = int(r.alive or 0)
        for p in list(s.ep):
            pl = r.pl[p]; hp0, am0 = s.prev[p]
            elim = bool(pl.eliminated) or not pl.alive
            t = {"dealt": (pl.damage_dealt + pl.bomb_damage) / 1000.0, "taken": pl.damage_taken / 1000.0,
                 "hp": (pl.hp - hp0) / 1000.0, "box": pl.box_damage / 1000.0, "cubes": float(pl.cubes), "kills": float(pl.kills),
                 "hits": float(pl.hits), "shots": float(1 if (pl.ammo < am0 - 500) else 0), "gas": float(pl.gas_ticks), "place": 0.0, "win": 0.0}
            rank = None
            if elim:
                rank = int(pl.elim_rank) if pl.elim_rank > 0 else 10
            elif done:
                rank = 1 if (r.winner == p or alive_now <= 1) else max(1, alive_now)
            elif trunc:
                rank = (1 + alive_now) / 2.0
            if rank is not None:
                t["place"] = (10 - rank) / 9.0; t["win"] = 1.0 if rank == 1 else 0.0
            rew = float(sum(w[k] * v for k, v in t.items()))
            ep = s.ep[p]; ep["steps"] += 1; ep["ret"] += rew
            for k, v in t.items(): ep["terms"][k] += v
            a = s.acts.get(p)
            if a is not None:
                ep["move_steps"] += int(a[0] > 0); ep["attack_steps"] += int(a[1] == 1); ep["super_steps"] += int(a[1] == 2)
            fin = rank is not None
            if p in s.agents:
                self._trans.append((s.agents[p], rew, fin, t, rank)); self.stats["agent_steps"] += 1
            s.prev[p] = (pl.hp, pl.ammo)
            if fin:
                self._end_agent(i, p, r, truncated=trunc and not elim and not done, rank=rank)
        if self._battle_over(i, r):
            for p in list(s.ep):   # players still alive at a battle end decided by end="player0"/... : truncated, no rank
                self._end_agent(i, p, r, truncated=True)
            self._reset(i)

    def _send_step(self, i, acts):
        s = self.I[i]; a = {p: action_to_native(v) for p, v in acts.items()}
        s.acts = acts
        s.inst.send(P.CMD_STEP, ticks=self.ticks, flags=self._flags(), ctl_mask=s.send_mask, native_mask=s.native, obs_mask=s.obs_mask,
                    actions=a if s.send_mask else None, **self._pix())
        s.state = "stepping"; s.t_sent = time.time()

    def check_resources(self):
        """every ~5 s: abort (MemoryGuard) when RAM or VRAM is about to run out, instead of freezing the PC"""
        now = time.time()
        if now - getattr(self, "_t_res", 0) < 5:
            return
        self._t_res = now
        av, vr = mem_available_mb(), vram_used_mb()
        self.stats["min_avail_mb"] = min(self.stats.get("min_avail_mb", 1 << 30), av); self.stats["max_vram_mb"] = max(self.stats.get("max_vram_mb", 0), vr)
        if av < MIN_AVAIL_MB_RUN or vr > MAX_VRAM_MB_RUN:
            raise MemoryGuard(f"resources low: {av} MB RAM available, {vr} MB VRAM used")

    def poll(self, min_ready=1, max_wait=0.01):
        """-> obs dict, agent ids (B,), transitions [(aid, reward, done, terms, rank)] finished since the last poll.
        Waits until at least min_ready instances have agents waiting (or max_wait seconds after the first one)."""
        self.check_resources()
        t0 = time.time(); first = None
        while True:
            for i, s in enumerate(self.I):
                if s.state in ("resetting", "stepping"):
                    if s.inst.done_seq():
                        if self._check(i) and i not in self.ready:
                            self.ready.append(i)
                    elif time.time() - s.t_sent > ((30 if self.pixels else 60) if s.state == "resetting" else 30):
                        if not s.inst.alive() or time.time() - s.t_sent > 60:
                            self._restart(i, "died" if not s.inst.alive() else "timeout")
                elif s.state == "boot":
                    if s.inst.ready():
                        self.stats["boot_s"].append(round(time.time() - s.boot_t0, 1)); self._reset(i)
                    elif not s.inst.alive() or time.time() - s.boot_t0 > 400:
                        self._restart(i, "reboot failed")
            if self.ready and first is None:
                first = time.time()
            if len(self.ready) >= min_ready or (first is not None and time.time() - first >= max_wait):
                break
            if all(s.state == "idle" for s in self.I):
                break
            if time.time() - t0 > 120:
                raise RuntimeError("maenv: no instance progressed for 120 s")
            fds = [s.inst.resp_r for s in self.I if s.state in ("resetting", "stepping") and s.inst.resp_r >= 0]
            if fds:
                rd = select.select(fds, [], [], 0.001)[0]
                for s in self.I:
                    if s.inst.resp_r in rd: s.inst.drain_bell()
        return self._gather()

    def _gather(self):
        ids, vecs, grids, owners, meta, tags, pixs, states = [], [], [], [], [], [], [], []
        for i in self.ready:
            s = self.I[i]; r = s.inst.resp
            for p, aid in s.agents.items():
                if not (r.obs_written >> p) & 1:
                    continue
                o = s.inst.obs_f32[p * (NV + NG):(p + 1) * (NV + NG)]
                ids.append(aid); vecs.append(o[:NV]); grids.append(o[NV:]); owners.append((i, s.W, s.H))
                meta.append((r.pl[p].x, r.pl[p].y, r.tick, s.W, s.H)); tags.append(s.cfg.get("tag"))
                if self.pixels:
                    pixs.append(s.stack.copy() if p == r.local_index else np.zeros_like(s.stack))
                if self.want_state:
                    states.append((s.inst.state_bytes(), (float(r.cam_x), float(r.cam_y)) if r.cam_valid else None, r.local_index, p))
        B = len(ids)
        vec = np.array(vecs, np.float32).reshape(B, NV)
        grid = np.array(grids, np.float32).reshape((B,) + GS)
        self.last_meta = np.array(meta, np.int32).reshape(B, 5); self.last_tags = tags
        if self.want_full:
            self.last_full = {"vec": vec.copy(), "grid": np.minimum(grid, 255).astype(np.uint8)}
            self.last_full["ecell"] = FAIR.enemy_cells(vec, 60, 60) if B else np.zeros((0, 9), np.int16)
        if self.fair and B:
            for (W, H) in set((w, h) for _, w, h in owners):
                sel = np.array([(w == W and h == H) for _, w, h in owners])
                if sel.all():
                    FAIR.fair_inplace(vec, grid, W, H)
                else:
                    v2, g2 = vec[sel], grid[sel]; FAIR.fair_inplace(v2, g2, W, H); vec[sel] = v2; grid[sel] = g2
        if B:
            self.last_ecell = FAIR.enemy_cells(vec, 60, 60)
        else:
            self.last_ecell = np.zeros((0, 9), np.int16)
        tr = self._trans; self._trans = []
        self._waiting = {i: [p for p, aid in self.I[i].agents.items()] for i in self.ready}
        self.ready = []
        out = {"vec": vec, "grid": grid.astype(np.uint8)}
        if self.pixels:
            out["pixels"] = np.stack(pixs) if pixs else np.zeros((0, self.obs_size[1], self.obs_size[0], 3 * self.k), np.uint8)
        self.last_state = states
        return out, np.array(ids, np.int64), tr

    def send(self, ids, actions):
        """actions for the agents returned by the last poll (any order; agents without an action get none = stand still)"""
        by_inst = {}
        for aid, a in zip(ids, actions):
            aid = int(aid); p = aid & 15; b = aid >> 4
            by_inst.setdefault(b, {})[p] = a
        for i, players in self._waiting.items():
            s = self.I[i]
            if s.state != "running":
                continue
            acts = by_inst.get(s.battle, {})
            self._send_step(i, {p: acts[p] for p in players if p in acts})
        self._waiting = {}

    def summary(self, rows=None):
        rows = self.episodes if rows is None else rows
        ok = [r for r in rows if r["rank"] is not None and not r["truncated"]]
        if not ok:
            return {}
        rk = np.array([r["rank"] for r in ok], float)
        return {"n": len(ok), "rank": float(rk.mean()), "rank_ci": float(1.96 * rk.std(ddof=1) / np.sqrt(len(rk))) if len(rk) > 1 else None,
                "win": float(np.mean([r["win"] for r in ok]))}
