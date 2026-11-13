"""Stage 1: BC from the native AI, then PPO, Piper on one Showdown map, ObsV5 observation (WSL, after `source env.sh`).

    ~/yolo_env/bin/python -m train_ppo --run s1 --hours 3            # outputs ~/nb68/rl/runs/s1/
    ~/yolo_env/bin/python -m train_ppo --run t --smoke               # 2 instances, tiny rollouts, 3 updates: plumbing check

* Updates 1..--bc-updates: all 10 players are driven by the native AI and observed; loss = cross-entropy to its action
  (env_v5.Env labels) + value regression. Then PPO: k agents per battle (--kmix) vs the native AI, clip + entropy +
  beta * KL(BC snapshot || policy) with beta decaying to --kl-floor over --kl-updates updates (nirkt's recipe).
* GRU: every row stores its input hidden state; updates run chunks of --seq consecutive steps from the stored state.
* Eval inside the run every --eval-min minutes: seeds 1..--eval-episodes, player 0 = the policy vs 9 native (sampled
  actions, not trained on) -> evals.jsonl + ckpt_u<update>.pt. Stage 1 gate: mean rank <= 3.0 at 500 episodes.
* Dashboard: progress.jsonl (one row per update), evals.jsonl, replays/ (one battle of instance 0 per update: all
  players + what the focus agent's policy saw and did) and a copy of viz.html. See README "Dashboard".
* Self-play (PPO phase): --sp of the battles have all 10 players on policies; 0-7 of them (tag "opp") are played by a
  league snapshot (league/u*.pt: saved every --league-every updates), the rest by the learner. PFSP: --opp-slots
  snapshots are loaded at once (a battle is pinned to one slot; one slot is redrawn every --opp-hold updates) and a
  snapshot is drawn with weight ((10 - r) / 9) ** --pfsp-p, r = running mean rank of its players against the learner
  (league/pfsp.json; 5.5 = equal, lower = it beats the learner). When the pool is full the weakest one is evicted.
  The other battles are k agents vs the native AI as before. Only learner rows are trained on.
* Critic input `priv` (true state of all 10 players): see model.py.
* Eval = a frozen copy of the weights: "eval" vs 9 native (sanity gate) and "evalb" vs 9 copies of --base (default: the
  --resume checkpoint; from scratch: the BC snapshot) -> b_rank in evals.jsonl: 5.5 = equal to base, lower = stronger.
* Evals run on the last --eval-n instances only; the others keep training.
shortcut: no exploiters (a second learner trained only to beat this one); add them if b_rank flattens again with PFSP.
"""
import argparse, glob, json, os, shutil, signal, sys, time
import numpy as np
import torch
from nbpc import diversity as D
from nbpc.maenv import REWARD_W, TERMS
import env_v5 as E, obs_v5 as O
from model import Policy, HID, PRIV, dist_logp_ent, masked_kl

NS = O.N_SELF + O.N_COND
HERE = os.path.dirname(os.path.abspath(__file__))


class Buffer:
    """transition rows: observation + hidden state at action time, action, logp, value; reward / done on completion"""
    F = {"s": ((NS,), np.float16), "ent": ((O.N_ENT, O.F_ENT), np.float16), "grid": ((O.G, O.G), np.uint8), "h": ((HID,), np.float16), "priv": ((PRIV,), np.float16),
         "act": ((4,), np.int16), "logp": ((), np.float32), "val": ((), np.float32), "rew": ((), np.float32),
         "done": ((), bool), "complete": ((), bool), "bc": ((), bool), "dead": ((), bool)}

    def __init__(self, cap):
        self.cap, self.n = cap, 0
        for k, (shape, dt) in self.F.items(): setattr(self, k, np.zeros((cap,) + shape, dt))

    def add(self, **rows):
        B = len(rows["act"]); sl = slice(self.n, self.n + B)
        if self.n + B > self.cap: raise RuntimeError("buffer full")
        for k in self.F: getattr(self, k)[sl] = rows.get(k, 0)
        self.n += B
        return np.arange(sl.start, sl.stop)

    def compact(self, keep):
        k = np.asarray(keep, np.int64)
        for name in self.F: a = getattr(self, name); a[:len(k)] = a[k]
        self.n = len(k)


class Recorder:
    """one battle of instance 0 per update -> replays/u<update>.json (+ index.json), the newest `keep` kept"""
    def __init__(self, out, keep=30):
        self.dir = os.path.join(out, "replays"); os.makedirs(self.dir, exist_ok=True)
        self.keep, self.want, self.cur, self.index = keep, True, None, []

    def frame(self, env, ids, o, act, upd):
        s = env.I[0]; rows = np.nonzero((ids >> 4) == s.battle)[0]
        if self.cur is not None and (self.cur["battle"] != s.battle or len(self.cur["frames"]) >= 600):   # 600 steps = 1 min of game
            self.flush()
        if not len(rows) or (self.cur is None and not self.want): return
        r = s.inst.resp; b = int(rows[0]); tag = s.cfg["tag"]
        if self.cur is None:
            tiles = O._tile_lut()[np.clip(np.asarray(s.map_tiles).reshape(s.H, s.W) % 1000000 + 1, 0, 255)] if s.map_tiles is not None else np.zeros((s.H, s.W), np.uint8)
            self.cur = {"battle": s.battle, "update": upd, "kind": tag["kind"], "loc": tag["loc"], "chr": tag["chr"], "ctl": s.ctl, "W": s.W, "H": s.H,
                        "seg": list(O.SEG) + ["unknown"], "ent": list(O.ENT), "map": ["".join(chr(97 + c) for c in row) for row in tiles], "frames": []}
            self.want = False
        e = o["ent"][b]; e = e[e[:, 0] > 0.5]
        self.cur["frames"].append({
            "t": int(r.tick), "f": int(ids[b]) & 15, "a": [int(x) for x in act[b]],
            "p": [[int(r.pl[p].x), int(r.pl[p].y), int(r.pl[p].hp), int(r.pl[p].max_hp), int(bool(r.pl[p].alive) and not r.pl[p].eliminated)] for p in range(10)],
            "e": [[int(x[1:1 + len(O.ENT)].argmax()), round(float(x[O.XY]), 3), round(float(x[O.XY + 1]), 3), round(float(x[O.HP]), 2)] for x in e],
            "s": [round(float(x), 2) for x in o["self"][b, :5]], "g": "".join(chr(97 + c) for c in o["grid"][b].ravel())})

    def flush(self):
        c, self.cur = self.cur, None
        if c is None or len(c["frames"]) < 20:
            self.want = True; return
        name = "u%04d.json" % c["update"]
        json.dump(c, open(os.path.join(self.dir, name), "w"), separators=(",", ":"))
        self.index = [x for x in self.index if x["file"] != name] + [{"file": name, "update": c["update"], "kind": c["kind"], "frames": len(c["frames"])}]
        for x in self.index[:-self.keep]: os.remove(os.path.join(self.dir, x["file"]))
        self.index = self.index[-self.keep:]
        json.dump(self.index, open(os.path.join(self.dir, "index.json"), "w"))


def priv(env, ids):
    """(B, PRIV) critic-only input: every player's true offset (/ 20 tiles), hp fraction, alive; zeros for the dead"""
    by = {s.battle: s for s in env.I}; cache = {}; out = np.zeros((len(ids), 10, 4), np.float32)
    for b, a in enumerate(ids):
        k = int(a) >> 4
        if k not in cache:
            pl = by[k].inst.resp.pl
            cache[k] = np.array([[pl[q].x, pl[q].y, pl[q].hp / max(1, pl[q].max_hp), bool(pl[q].alive) and not pl[q].eliminated] for q in range(10)], np.float32)
        c = cache[k].copy(); c[:, :2] = (c[:, :2] - c[int(a) & 15, :2]) / 6000.0; c[c[:, 3] < 0.5] = 0; out[b] = c
    return out.reshape(len(ids), PRIV)


def chunks(traj, L):
    """trajectories (lists of buffer rows in time order) -> (N, L) row indices, -1 = padding"""
    out = [r[i:i + L] + [-1] * (L - len(r[i:i + L])) for r in traj for i in range(0, len(r), L)]
    return np.array(out, np.int64).reshape(-1, L)


def eval_summary(rows):
    ok = [r for r in rows if r["rank"] is not None and not r["truncated"]]
    rk = np.array([r["rank"] for r in ok], float); m = lambda k: round(float(np.mean([r["terms"][k] for r in ok])), 3) if ok else None
    return {"episodes": len(ok), "rank": round(float(rk.mean()), 3) if ok else None,
            "ci95": round(float(1.96 * rk.std(ddof=1) / np.sqrt(len(rk))), 3) if len(ok) > 1 else None,
            "win": round(float(np.mean([r["win"] for r in ok])), 3) if ok else None, "top3": round(float((rk <= 3).mean()), 3) if ok else None,
            "dealt": m("dealt"), "taken": m("taken"), "kills": m("kills"), "cubes": m("cubes"),
            "ticks": round(float(np.mean([r["ticks"] for r in ok])), 1) if ok else None,
            "hit_rate": round(sum(r["terms"]["hits"] for r in ok) / max(1.0, sum(r["terms"]["shots"] for r in ok)), 3) if ok else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--hours", type=float, default=3.0); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--rollout", type=int, default=32768); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq", type=int, default=8); ap.add_argument("--chunks", type=int, default=512, help="chunks per minibatch")
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--bc-lr", type=float, default=5e-4)
    ap.add_argument("--bc-updates", type=int, default=40); ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.001); ap.add_argument("--kl-beta", type=float, default=0.5)
    ap.add_argument("--kl-updates", type=int, default=60); ap.add_argument("--kl-floor", type=float, default=0.02)
    ap.add_argument("--kmix", default="1,2,3,4,5"); ap.add_argument("--brawler", default="Sniper", help="Sniper = Piper")
    ap.add_argument("--loc", default=None, help="Showdown location (default: the first of env_v5.locations())")
    ap.add_argument("--slider-scale", type=float, default=0.0, help="weight of the slider rewards (stage 1: off)")
    ap.add_argument("--noise", type=int, default=1, help="ObsV5 vision noise + latency")
    ap.add_argument("--eval-min", type=float, default=30.0); ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--max-ticks", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sp", type=float, default=0.8, help="share of PPO battles that are self-play (rest: vs native AI)")
    ap.add_argument("--league-every", type=int, default=20); ap.add_argument("--league-size", type=int, default=16)
    ap.add_argument("--base", default=None, help="checkpoint of the evalb opponent (default: --resume)")
    ap.add_argument("--evalb-episodes", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=99, help="instances that play the evals (the rest keep training; fewer than --n was slower overall, see README)")
    ap.add_argument("--opp-slots", type=int, default=3); ap.add_argument("--opp-hold", type=int, default=2)
    ap.add_argument("--pfsp-p", type=float, default=2.0)
    a = ap.parse_args()
    if a.smoke: a.n, a.rollout, a.chunks, a.bc_updates, a.eval_episodes, a.evalb_episodes, a.hours = 2, 2048, 64, 1, 4, 4, 0.2
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.expanduser("~/nb68/rl/runs"), a.run); os.makedirs(out, exist_ok=True)
    loc = a.loc or E.locations()[0]
    json.dump({**vars(a), "loc": loc, "reward_w": REWARD_W}, open(os.path.join(out, "config.json"), "w"), indent=1)
    shutil.copy(os.path.join(HERE, "viz.html"), os.path.join(out, "index.html"))
    dev = torch.device("cuda")
    def frozen(sd):
        m = Policy().to(dev).eval(); m.load_state_dict(sd, strict=False); [p.requires_grad_(False) for p in m.parameters()]
        return m
    net = Policy().to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.bc_lr, eps=1e-5); ref = bnet = enet = None
    upd = 0; agent_steps = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"], strict=False); upd, agent_steps = ck["update"], ck["agent_steps"]
        try: opt.load_state_dict(ck["opt"])
        except ValueError: print("optimizer state not loaded (new parameters): fresh Adam", flush=True)
        if ck.get("ref"): ref = frozen(ck["ref"])
        bnet = frozen(torch.load(a.base, map_location="cpu", weights_only=False)["model"] if a.base else ck["model"])
    kmix = [int(x) for x in a.kmix.split(",")]; rng_b = np.random.default_rng(a.seed + 7)
    eval_queue = []; ev = {"active": False, "t_last": 0.0 if a.resume else time.time()}      # resumed: eval first
    os.makedirs(os.path.join(out, "league"), exist_ok=True); M = a.opp_slots
    lnets = [frozen(net.state_dict()) for _ in range(M)]; lname = [None] * M     # loaded league opponents / their file names
    pool = sorted(glob.glob(os.path.join(out, "league", "u*.pt"))); pf_path = os.path.join(out, "league", "pfsp.json")
    pf = json.load(open(pf_path)) if os.path.exists(pf_path) else {}             # file name -> [mean rank of its players, rows]
    pfr = lambda p: pf.get(os.path.basename(p), [5.5, 0])[0]
    def load_slot(k):
        w = np.array([((10 - pfr(p)) / 9) ** a.pfsp_p for p in pool]); p = pool[int(rng_b.choice(len(pool), p=w / w.sum()))]
        lnets[k].load_state_dict(torch.load(p, map_location=dev), strict=False); lname[k] = os.path.basename(p)
    if pool: [load_slot(k) for k in range(M)]
    bc = lambda: upd < a.bc_updates

    def battle_fn(rng, i):
        tag = {"loc": loc, "chr": [a.brawler] * 10, "team_size": 1, "groups": [0] * 10}
        cfg = {"div": D.make_div(location=loc, chr=tag["chr"]), "tag": tag, "end": "agents"}
        if eval_queue and not bc() and i >= a.n - a.eval_n:
            sd, kd = eval_queue.pop(0); b = kd == "evalb"
            return {**cfg, "seed": sd, "ctl": list(range(10)) if b else [0], "end": "player0",
                    "tag": {**tag, "kind": kd, "opp": list(range(1, 10)) if b else []}}
        sd = int(rng_b.integers(10_000_000, 2 ** 31 - 1))
        if bc():
            return {**cfg, "seed": sd, "ctl": list(range(10)), "native_ctl": True, "tag": {**tag, "kind": "bc"}}
        if rng_b.random() < a.sp:
            opp = sorted(rng_b.choice(10, int(rng_b.integers(0, 8)), replace=False).tolist())
            k = int(rng_b.integers(M))
            return {**cfg, "seed": sd, "ctl": list(range(10)), "tag": {**tag, "kind": "sp", "opp": opp, "slot": k, "snap": lname[k]}}
        k = int(rng_b.choice(kmix))
        return {**cfg, "seed": sd, "ctl": sorted(rng_b.choice(10, k, replace=False).tolist()), "tag": {**tag, "kind": "k%d" % k}}

    E.Env.slider_scale = a.slider_scale; E.Env.record = a.bc_updates > 0
    env = E.Env(a.n, battle_fn, ticks=2, fair=True, max_ticks=a.max_ticks, verbose=True, seed=a.seed, tag="ppo%d" % os.getpid())
    ob = O.ObsV5(noise=O.NOISE if a.noise else None, seed=a.seed); rec = Recorder(out)
    buf = Buffer(a.rollout + 8192)
    pending, traj, final, hid = {}, {}, set(), {}      # aid -> row waiting for its transition / rows this rollout / ended / GRU state
    epf = open(os.path.join(out, "episodes.jsonl"), "a"); evf = open(os.path.join(out, "evals.jsonl"), "a")
    prf = open(os.path.join(out, "progress.jsonl"), "a")
    t0 = time.time(); n_ep = 0; ep_rows = []; stop = {"now": False}; z = np.zeros(HID, np.float32)
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    T = lambda x, dt=torch.float32: torch.as_tensor(x, device=dev).to(dt)

    def save(path):
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "ref": ref.state_dict() if ref else None, "update": upd,
                    "agent_steps": agent_steps, "args": vars(a), "loc": loc}, path)

    def start_eval():
        nonlocal enet
        enet = frozen(net.state_dict())
        eval_queue.extend([(sd, "eval") for sd in range(1, a.eval_episodes + 1)] + [(sd, "evalb") for sd in range(1, a.evalb_episodes + 1)])
        ev.update(active=True, update=upd, steps=agent_steps, t0=time.time(), n0=len(env.episodes))
        save(os.path.join(out, "ckpt_u%04d.pt" % upd))

    def finish_eval():
        if eval_queue or any(s.state != "idle" and (s.cfg or {}).get("tag", {}).get("kind", "").startswith("eval") for s in env.I): return
        by = lambda kd: list({r["seed"]: r for r in env.episodes[ev["n0"]:] if r["tag"]["kind"] == kd and r["player"] == 0}.values())
        sm = {"update": ev["update"], "agent_steps": ev["steps"], "minutes": round((time.time() - t0) / 60, 1),
              "eval_s": round(time.time() - ev["t0"], 1), **eval_summary(by("eval")),
              **{"b_" + k: v for k, v in eval_summary(by("evalb")).items() if k in ("episodes", "rank", "ci95", "win", "top3")}}
        evf.write(json.dumps(sm) + "\n"); evf.flush(); print("[eval]", json.dumps(sm), flush=True)
        ev.update(active=False, t_last=time.time())

    t_end = t0 + a.hours * 3600; smoke_end = upd + 3
    try:
        while time.time() < t_end and not stop["now"] and not (a.smoke and upd >= smoke_end):
            # ---------------- collect
            dec_t, dec_n = time.time(), 0
            while buf.n < a.rollout and not stop["now"]:
                obs, ids, tr = env.poll(min_ready=max(1, a.n // 3), max_wait=0.002)
                for (aid, rew, done, terms, rank) in tr:
                    j = pending.pop(aid, None)
                    if done: hid.pop(aid, None)
                    if j is None: continue
                    buf.rew[j] = rew; buf.done[j] = done; buf.complete[j] = True
                    if buf.bc[j] and terms.get("_act") is not None: buf.act[j] = terms["_act"]
                    elif buf.bc[j]: continue                 # a BC row without a label (recording went off): not trained on
                    traj.setdefault(aid, []).append(j)
                    if done: final.add(aid)
                for aid in env.dropped: pending.pop(aid, None); hid.pop(aid, None)
                env.dropped = []
                if not len(ids): continue
                o = ob.build(obs["vec"], obs["grid"], env.last_meta, ids)
                s = np.concatenate([o["self"], env.cond(ids)], 1); h = np.stack([hid.get(int(x), z) for x in ids])
                pv = priv(env, ids); B = len(ids); kind = np.array([t["kind"] for t in env.last_tags])
                isev = np.char.startswith(kind, "eval"); opp = np.array([int(x) & 15 in t.get("opp", ()) for x, t in zip(ids, env.last_tags)], bool)
                A = np.zeros((B, 4), np.int64); h2 = np.zeros((B, HID), np.float32); lp = np.zeros(B, np.float32); v = np.zeros(B, np.float32)
                sl = np.array([t.get("slot", 0) for t in env.last_tags])
                with torch.no_grad():                        # learner / frozen eval copy / evalb base / league slots
                    for net_, mk in [(net, ~isev & ~opp), (enet, isev & ~opp), (bnet, isev & opp)] + [(lnets[k], ~isev & opp & (sl == k)) for k in range(M)]:
                        if not mk.any(): continue
                        lg, v_, h_ = net_(T(s[mk]), T(o["ent"][mk]), T(o["grid"][mk], torch.uint8), T(h[mk]), T(pv[mk]))
                        lp_, _, A_ = dist_logp_ent(lg)
                        A[mk], h2[mk], lp[mk], v[mk] = A_.cpu().numpy(), h_.cpu().numpy(), lp_.cpu().numpy(), v_.cpu().numpy()
                for x, hh in zip(ids, h2): hid[int(x)] = hh
                m = ~isev & ~opp & ((kind == "bc") == bc())   # BC battles still running after the switch are not collected
                if m.any():
                    rows = buf.add(s=s[m], ent=o["ent"][m], grid=o["grid"][m], h=h[m], priv=pv[m], act=A[m], logp=lp[m],
                                   val=v[m], bc=kind[m] == "bc", dead=env.dead(ids[m]))
                    for aid, j in zip(ids[m], rows): pending[int(aid)] = int(j)
                    agent_steps += int(m.sum()); dec_n += int(m.sum())
                rec.frame(env, ids, o, A, upd)
                env.send(ids, A)
                for r in env.episodes[n_ep:]:
                    tg = r["tag"]                            # PFSP score: only battles whose snapshot stayed loaded to the end
                    if (tg["kind"] == "sp" and r["player"] in tg["opp"] and tg.get("snap") and tg["snap"] == lname[tg["slot"]]
                            and r["rank"] is not None and not r["truncated"]):
                        e = pf.setdefault(tg["snap"], [5.5, 0]); e[0] += 0.02 * (r["rank"] - e[0]); e[1] += 1
                    if not r["tag"]["kind"].startswith("eval") and r["player"] not in r["tag"].get("opp", ()):
                        ep_rows.append(r); epf.write(json.dumps(r) + "\n")
                n_ep = len(env.episodes)
                if ev["active"]: finish_eval()
                elif not bc() and time.time() - ev["t_last"] > a.eval_min * 60: start_eval()
            if stop["now"]: break
            dps = dec_n / max(1e-9, time.time() - dec_t)
            # ---------------- update
            tu = time.time(); epf.flush(); is_bc = bc(); n = buf.n
            adv = np.zeros(n, np.float32); ret = np.zeros(n, np.float32)
            for aid, rows in traj.items():                   # GAE; bootstrap = the value of the pending (next) row
                nxt = 0.0 if aid in final else float(buf.val[pending[aid]]) if aid in pending else float(buf.val[rows[-1]]); last = 0.0
                for j in rows[::-1]:
                    nt = 0.0 if buf.done[j] else 1.0
                    last = buf.rew[j] + a.gamma * nxt * nt - buf.val[j] + a.gamma * a.lam * nt * last
                    adv[j] = last; ret[j] = last + buf.val[j]; nxt = buf.val[j]
            use = [r for r in traj.values() if bool(buf.bc[r[0]]) == is_bc]
            C = torch.as_tensor(chunks(use, a.seq), device=dev); nrow = int((C >= 0).sum())
            g = {k: T(getattr(buf, k)[:n], dt) for k, dt in (("s", torch.float32), ("ent", torch.float32), ("grid", torch.uint8),
                 ("h", torch.float32), ("priv", torch.float32), ("act", torch.long), ("logp", torch.float32), ("val", torch.float32))}
            ADV, RET = T(adv), T(ret); LIVE = T(~buf.dead[:n], torch.float32)   # respawning rows: value only (actions ignored)
            pu = upd - a.bc_updates
            beta = 0.0 if ref is None else max(a.kl_floor, a.kl_beta * (1 - pu / max(1, a.kl_updates)))
            for pg_ in opt.param_groups: pg_["lr"] = a.bc_lr if is_bc else a.lr
            stats = []; mb = min(a.chunks, len(C))
            for _ in range(a.epochs if mb else 0):
                perm = torch.randperm(len(C), device=dev)
                for s0 in range(0, len(C) - mb + 1, mb):
                    c = C[perm[s0:s0 + mb]]; mk = (c >= 0).flatten(); r = c.clamp(min=0); rf = r.flatten()[mk]
                    lg, v = net.seq(g["s"][r], g["ent"][r], g["grid"][r], g["h"][r[:, 0]], g["priv"][r])
                    lg = [x[mk] for x in lg]; v = v[mk]
                    lp, ent, _ = dist_logp_ent(lg, g["act"][rf]); vl = 0.5 * ((v - RET[rf]) ** 2).mean()
                    lv = LIVE[rf]; lm = lambda x: (x * lv).sum() / lv.sum().clamp(min=1)
                    if is_bc:
                        loss = -lm(lp) + 0.5 * vl; st = [-lm(lp).item(), vl.item(), ent.mean().item(), 0.0, 0.0]
                    else:
                        ratio = (lp - g["logp"][rf]).exp(); ad = ADV[rf]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                        pg = lm(torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)))
                        kl = torch.zeros((), device=dev)
                        if beta > 0:
                            with torch.no_grad(): lr_, _ = ref.seq(g["s"][r], g["ent"][r], g["grid"][r], g["h"][r[:, 0]])
                            kl = masked_kl([x[mk] for x in lr_], lg).mean()
                        loss = pg - a.ent * lm(ent) + beta * kl + 0.5 * vl
                        st = [pg.item(), vl.item(), ent.mean().item(), kl.item(), ((ratio - 1).abs() > a.clip).float().mean().item()]
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                    stats.append(st)
            rows = C[C >= 0].cpu().numpy()
            expl = 1 - float(((ret[rows] - buf.val[rows]) ** 2).mean() / (ret[rows].var() + 1e-8)) if len(rows) else 0.0
            acts = buf.act[rows]; fired = acts[acts[:, 1] > 0]
            upd += 1
            if is_bc and not bc():                           # BC done: freeze the anchor, PPO starts with an eval
                ref = frozen(net.state_dict()); bnet = ref if bnet is None else bnet
                env.record = False; save(os.path.join(out, "bc.pt")); start_eval()
            if not bc():                                     # league: a snapshot now and then, one opponent per update
                if not pool or upd % a.league_every == 0:
                    if len(pool) >= max(2, a.league_size):   # evict the snapshot the learner beats most easily
                        old = pool.pop(int(np.argmax([pfr(p) for p in pool]))); os.remove(old); pf.pop(os.path.basename(old), None)
                    pool.append(os.path.join(out, "league", "u%05d.pt" % upd)); torch.save(net.state_dict(), pool[-1])
                for k in [k for k in range(M) if lname[k] is None] or ([upd // a.opp_hold % M] if upd % a.opp_hold == 0 else []):
                    load_slot(k)
                json.dump(pf, open(pf_path, "w"))
            keep = sorted(set(pending.values())); buf.compact(keep); remap = {j: i for i, j in enumerate(keep)}
            pending = {aid: remap[j] for aid, j in pending.items()}; traj = {}; final = set(); rec.want = True
            # ---------------- log
            s = np.mean(stats, 0) if stats else np.zeros(5); recent = ep_rows[-400:]; ep_rows = recent
            ok = [r for r in recent if r["rank"] is not None and not r["truncated"]]
            mean = lambda xs: round(float(np.mean(xs)), 4) if len(xs) else None
            row = {"update": upd, "phase": "bc" if is_bc else "ppo", "agent_steps": agent_steps, "minutes": round((time.time() - t0) / 60, 2),
                   "decisions_per_s": round(dps, 1), "update_s": round(time.time() - tu, 2), "rows": nrow,
                   "loss_pi": round(float(s[0]), 5), "vloss": round(float(s[1]), 4), "entropy": round(float(s[2]), 4),
                   "kl_bc": round(float(s[3]), 5), "clipfrac": round(float(s[4]), 4), "beta": round(beta, 4), "expl_var": round(expl, 4),
                   "episodes": len(env.episodes), "restarts": env.stats["restarts"],
                   "rank": mean([r["rank"] for r in ok]), "win": mean([r["win"] for r in ok]), "ret": mean([r["ret"] for r in recent]),
                   "ticks": mean([r["ticks"] for r in ok]), "storm_death": mean([r["storm_death"] for r in ok]),
                   "hit_rate": round(sum(r["terms"]["hits"] for r in recent) / max(1.0, sum(r["terms"]["shots"] for r in recent)), 4),
                   "attack_frac": mean([r["attack_steps"] / max(1, r["steps"]) for r in recent]),
                   "move_hist": np.bincount(acts[:, 0], minlength=17)[:17].tolist(), "fire_hist": np.bincount(acts[:, 1], minlength=3)[:3].tolist(),
                   "aim_hist": np.bincount(fired[:, 2] // 15, minlength=24)[:24].tolist()}
            for k in sorted({r["tag"]["kind"] for r in ok}): row["rank_" + k] = mean([r["rank"] for r in ok if r["tag"]["kind"] == k])
            row["pfsp"] = {os.path.basename(p)[:-3]: [round(pfr(p), 2), pf.get(os.path.basename(p), [0, 0])[1]] for p in pool}
            for k in TERMS: row["t_" + k] = mean([r["terms"][k] for r in recent])
            prf.write(json.dumps(row) + "\n"); prf.flush()
            print(json.dumps({k: v for k, v in row.items() if not k.endswith("_hist") and k != "pfsp"}), flush=True)
            if upd % 5 == 0: save(os.path.join(out, "ckpt_last.pt"))
    finally:
        save(os.path.join(out, "ckpt_last.pt")); env.close()
        if a.smoke and sys.exc_info()[0] is None:
            assert upd >= smoke_end and os.path.getsize(os.path.join(out, "progress.jsonl")) > 0
            print("smoke ok: %d updates, %d steps, replays %s" % (upd, agent_steps, [x["file"] for x in rec.index]))


if __name__ == "__main__":
    sys.exit(main())
