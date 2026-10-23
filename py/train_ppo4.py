#!/usr/bin/env python3
"""PPO for the state teacher (pctrain round P1) on the asynchronous multi-agent env (nbpc.maenv).

* Battles: k policy-controlled Pipers (k drawn per battle from --kmix) + 10-k native AI Pipers (a share --lead-opp of them
  with "native + lead" aim); random training seeds >= 10,000,000 (the eval suite uses 1..500, mixed screening 200,001..,
  demos 1,000,000..). A battle is reset as soon as all its policy players are eliminated.
* Actor: models4.StatePolicy (pointer aim with intercept candidates) on the FAIR observation (nbpc.fair); critic: the same
  architecture on the UNFAIR (full) observation (asymmetric critic, value head from scratch, --value-warmup critic-only
  updates). Inference: one CUDA graph per batch bucket for actor + critic (policy_rt.sample_heads).
* Loss: PPO clip + entropy (masked like the log-probability) + beta * KL(pi_BC || pi), the KL MASKED like the policy:
  KL_move + KL_fire + P_BC(fire > 0) KL_aim + P_BC(super) KL_dist (research F1.4); beta decays linearly to --kl-floor over
  --kl-updates updates. Critic: MSE to GAE(lambda) returns.
* Rewards (logged per term, maenv.TERMS): see --reward (JSON overrides maenv.REWARD_W).
* Held-out evaluation inside the run: every --eval-min minutes the 500 eval seeds are played 1 vs 9 (player 0 = a frozen
  snapshot of the actor, sampled actions; their transitions are NOT trained on) -> evals.jsonl + ckpt_u<update>.pt.
Outputs ../runs/<run>/: config.json, progress.csv (per update), episodes.jsonl (training episodes), evals.jsonl, ckpt_*.pt.

  python3 train_ppo4.py --init ../runs/bc4_x/bc_best.pt --run ppo4_a --hours 8
"""
import argparse, copy, csv, json, math, os, signal, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_rocm_fix  # noqa: F401
from models4 import make_state_policy, dist_logp_ent
from policy_rt import sample_heads, GraphPolicy
from grid4 import GridBuilder
from nbpc.maenv import MAEnv, REWARD_W, TERMS
from eval_suite import summarize, EVAL_SEEDS


# the Showdown map (fixed: the same hash 3654183511 in all 1,000 demo battles; data/map_showdown.json)
import numpy as _np, os as _os
MAP_TILES = _np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "map_showdown.npy"))

def value_of(net, v, g):
    """critic value only (no policy heads / pointer aim)"""
    return net.heads.v(net.features(v, g)).squeeze(-1)


class ACGraph:
    """actor(fair obs) + critic(full obs) + sampling in one CUDA graph per bucket"""
    def __init__(self, actor, critic, dev, buckets=(16, 48, 96, 192)):
        self.actor, self.critic, self.dev, self.buckets = actor, critic, dev, sorted(buckets); self.g = {}
        for b in self.buckets: self._cap(b)

    def _fwd(self, vf, gf, vu, gu):
        lg, _ = self.actor(vf, gf); A, L = sample_heads(lg)
        V = value_of(self.critic, vu, gu)
        return torch.cat([A.float(), L[:, None], V[:, None].float()], -1)

    def _cap(self, b):
        t = [torch.zeros(b, 436, device=self.dev), torch.zeros(b, 6, 25, 25, device=self.dev, dtype=torch.uint8),
             torch.zeros(b, 436, device=self.dev), torch.zeros(b, 6, 25, 25, device=self.dev, dtype=torch.uint8)]
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3): self._fwd(*t)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            out = self._fwd(*t)
        self.g[b] = (g, t, out, torch.empty(b, 6).pin_memory())

    @torch.no_grad()
    def act(self, vf, gf, vu, gu):
        B = len(vf)
        if B > self.buckets[-1]:
            o = self._fwd(*(torch.as_tensor(x, device=self.dev) for x in (vf, gf, vu, gu))).cpu().numpy()
        else:
            b = next(x for x in self.buckets if x >= B); g, t, out, pin = self.g[b]
            for dst, src in zip(t, (vf, gf, vu, gu)):
                dst[:B].copy_(torch.from_numpy(src))
                if B < b: dst[B:].zero_()
            g.replay(); pin[:B].copy_(out[:B]); o = pin[:B].numpy().copy()
        return o[:, :4].astype(np.int64), o[:, 4].astype(np.float32), o[:, 5].astype(np.float32)


def masked_kl(lg_ref, lg_new):
    """KL(pi_ref || pi_new) of the factored v3 policy: move + fire + P_ref(fire>0) aim + P_ref(super) distance"""
    kls = []
    for a, b in zip(lg_ref, lg_new):
        pa = torch.log_softmax(a.float(), -1); pb = torch.log_softmax(b.float(), -1)
        kls.append((pa.exp() * (pa - pb)).sum(-1))
    pf = torch.softmax(lg_ref[1].float(), -1)
    return kls[0] + kls[1] + (1 - pf[:, 0]) * kls[2] + pf[:, 2] * kls[3]


class Buffer:
    """transition rows (obs at action time, action, logp, value; reward/done filled on completion)"""
    def __init__(self, cap):
        self.cap = cap
        self.vf = np.zeros((cap, 436), np.float16); self.vu = np.zeros((cap, 436), np.float16)
        self.meta = np.zeros((cap, 3), np.int32); self.ef = np.zeros((cap, 9), np.int16); self.eu = np.zeros((cap, 9), np.int16)
        self.act = np.zeros((cap, 4), np.int16); self.logp = np.zeros(cap, np.float32); self.val = np.zeros(cap, np.float32)
        self.rew = np.zeros(cap, np.float32); self.done = np.zeros(cap, bool); self.complete = np.zeros(cap, bool)
        self.n = 0

    def add(self, vf, vu, meta, ef, eu, act, logp, val):
        B = len(act); i0 = self.n
        if i0 + B > self.cap: raise RuntimeError("buffer full")
        sl = slice(i0, i0 + B)
        self.vf[sl] = vf; self.vu[sl] = vu; self.meta[sl] = meta; self.ef[sl] = ef; self.eu[sl] = eu; self.act[sl] = act
        self.logp[sl] = logp; self.val[sl] = val; self.complete[sl] = False; self.done[sl] = False; self.rew[sl] = 0
        self.n += B
        return np.arange(i0, i0 + B)

    def compact(self, keep):
        """move rows `keep` (pending, incomplete) to the front; returns their new indices"""
        k = np.asarray(keep, np.int64); m = len(k)
        for arr in (self.vf, self.vu, self.meta, self.ef, self.eu, self.act, self.logp, self.val, self.rew, self.done, self.complete):
            arr[:m] = arr[k]
        self.n = m
        return np.arange(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=8.0); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--rollout", type=int, default=65536); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=4096); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--clr", type=float, default=2e-4); ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.001); ap.add_argument("--kl-beta", type=float, default=0.5)
    ap.add_argument("--kl-updates", type=int, default=60); ap.add_argument("--kl-floor", type=float, default=0.02)
    ap.add_argument("--value-warmup", type=int, default=4); ap.add_argument("--kmix", default="1,2,3,4,5")
    ap.add_argument("--lead-opp", type=float, default=0.0, help="probability that a native opponent uses lead aim")
    ap.add_argument("--reward", default="{}"); ap.add_argument("--eval-min", type=float, default=40.0)
    ap.add_argument("--eval-episodes", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=4000); ap.add_argument("--resume", default=None)
    ap.add_argument("--delay-lead", type=float, default=None, help="pointer candidates at delay-aware intercept angles (lead fraction)")
    ap.add_argument("--grid-bf16", type=int, default=1, help="grid CNN in bf16 autocast (inference graphs and training alike)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", a.run); os.makedirs(out, exist_ok=True)
    rw = dict(REWARD_W); rw.update(json.loads(a.reward))
    json.dump({**vars(a), "reward_w": rw}, open(os.path.join(out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda")
    ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
    actor = make_state_policy(ck0, dev, delay_lead=a.delay_lead)
    actor.grid_bf16 = bool(a.grid_bf16)
    ref = copy.deepcopy(actor).eval(); critic = copy.deepcopy(actor)
    nn.init.orthogonal_(critic.heads.v.weight, 1.0); nn.init.zeros_(critic.heads.v.bias)
    for p in ref.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=a.lr, eps=1e-5); copt = torch.optim.Adam(critic.parameters(), lr=a.clr, eps=1e-5)
    upd0 = 0; steps0 = 0
    if a.resume:
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        actor.load_state_dict(rk["model"]); critic.load_state_dict(rk["critic"]); upd0 = rk.get("update", 0); steps0 = rk.get("agent_steps", 0)
        if "opt" in rk: opt.load_state_dict(rk["opt"]); copt.load_state_dict(rk["copt"])
    snap = copy.deepcopy(actor).eval()
    kmix = [int(x) for x in a.kmix.split(",")]
    rng_b = np.random.default_rng(a.seed + 7)
    eval_queue = []; eval_state = {"active": False, "t_last": time.time(), "update": None}

    def battle_fn(rng, i):
        if eval_queue:
            sd = eval_queue.pop(0)
            return {"seed": sd, "ctl": [0], "end": "player0", "tag": "eval"}
        sd = int(rng_b.integers(10_000_000, 2 ** 31 - 1)); k = int(rng_b.choice(kmix))
        S = sorted(rng_b.choice(10, k, replace=False).tolist())
        lm = 0
        if a.lead_opp > 0:
            lm = int(sum(1 << p for p in range(10) if p not in S and rng_b.random() < a.lead_opp))
        return {"seed": sd, "ctl": S, "lead_mask": lm, "end": "agents", "tag": f"k{k}"}

    env = MAEnv(a.n, battle_fn, reward_w=rw, max_ticks=a.max_ticks, fair=True, tag=f"ppo{os.getpid()}", verbose=True, want_full=True,
                seed=a.seed)
    gb = GridBuilder(MAP_TILES, 60, 60, dev)
    acg = ACGraph(actor, critic, dev); snapg = GraphPolicy(snap, None, dev=dev)
    buf = Buffer(a.rollout + 4096)
    pending = {}          # aid -> buffer row of its last action (waiting for the transition)
    traj = {}             # aid -> list of completed rows in this rollout (time order)
    final = set()         # aids whose episode ended (no bootstrap)
    dropped_boot = {}     # aids dropped by the env (restart): bootstrap value
    prog = None; epf = open(os.path.join(out, "episodes.jsonl"), "a"); evf = open(os.path.join(out, "evals.jsonl"), "a")
    t0 = time.time(); upd = upd0; agent_steps = steps0; n_ep_seen = 0; stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    ep_rows = []; dec_t = time.time(); dec_n = 0; t_inf = 0.0; t_upd_tot = 0.0

    def save(path, extra=None):
        torch.save({"model": actor.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(), "copt": copt.state_dict(),
                    "update": upd, "agent_steps": agent_steps, "vec_dim": 436, "grid_shape": [6, 25, 25], "nvec": ck0["nvec"],
                    "pointer_aim": True, "pointer_lead": ck0.get("pointer_lead", False), "fair": True, "init": a.init,
                    "delay_lead": getattr(actor.aim, "delay_lead", None), **(extra or {})}, path)

    def start_eval():
        snap.load_state_dict(actor.state_dict())
        eval_queue.extend(EVAL_SEEDS[:a.eval_episodes]); eval_state.update(active=True, update=upd, steps=agent_steps, t0=time.time(), n0=len(env.episodes))
        save(os.path.join(out, f"ckpt_u{upd:04d}.pt"))

    def finish_eval():
        rows = [r for r in env.episodes[eval_state["n0"]:] if r["tag"] == "eval" and r["player"] == 0]
        if len({r["seed"] for r in rows}) < a.eval_episodes:
            return False
        rows = {r["seed"]: r for r in rows}; rows = list(rows.values())
        for r in rows: r["ctl_player"] = True
        sm = summarize(rows); sm.update(update=eval_state["update"], agent_steps=eval_state["steps"], minutes=round((time.time() - t0) / 60, 1),
                                         eval_s=round(time.time() - eval_state["t0"], 1))
        evf.write(json.dumps(sm) + "\n"); evf.flush()
        print(f"[eval] update {sm['update']}: rank {sm['rank']['mean']:.3f} +- {sm['rank']['ci95']:.3f}, win {sm['win']['mean']:.3f}, "
              f"dealt {sm['dealt']['mean']:.0f}, hit {sm['hit_rate_pooled']}", flush=True)
        eval_state["active"] = False; eval_state["t_last"] = time.time()
        return True

    if upd0 == 0:
        start_eval()   # the BC init on the eval suite (same protocol as eval_suite.py)
    t_end = t0 + a.hours * 3600
    try:
        while time.time() < t_end and not stop["now"]:
            # ---------------- collect
            while buf.n < a.rollout and not stop["now"]:
                obs, ids, tr = env.poll(min_ready=max(1, a.n // 3), max_wait=0.002)
                for (aid, rew, done, terms, rank) in tr:
                    j = pending.pop(aid, None)
                    if j is None: continue
                    buf.rew[j] = rew; buf.done[j] = done; buf.complete[j] = True
                    traj.setdefault(aid, []).append(j)
                    if done: final.add(aid)
                for aid in env.dropped:      # restarted instance: its last action got no transition -> bootstrap from it
                    j = pending.pop(aid, None)
                    if j is not None: dropped_boot[aid] = float(buf.val[j])
                env.dropped = []
                if not len(ids):
                    continue
                tags = env.last_tags; is_ev = np.array([t == "eval" for t in tags])
                act = np.zeros((len(ids), 4), np.int64)
                ti = time.time()
                if (~is_ev).any():
                    m = ~is_ev; full = env.last_full
                    A, L, V = acg.act(obs["vec"][m], obs["grid"][m], full["vec"][m], full["grid"][m])
                    act[m] = A
                    rows = buf.add(obs["vec"][m], full["vec"][m], env.last_meta[m, :3], env.last_ecell[m], full["ecell"][m], A, L, V)
                    for aid, j in zip(ids[m], rows): pending[int(aid)] = int(j)
                    agent_steps += int(m.sum()); dec_n += int(m.sum())
                if is_ev.any():
                    A, _, _ = snapg.act(obs["vec"][is_ev], obs["grid"][is_ev]); act[is_ev] = A
                t_inf += time.time() - ti
                env.send(ids, act)
                # episodes
                new = env.episodes[n_ep_seen:]; n_ep_seen = len(env.episodes)
                for r in new:
                    if r["tag"] != "eval":
                        ep_rows.append(r); epf.write(json.dumps(r) + "\n")
                if eval_state["active"]:
                    finish_eval()
                elif time.time() - eval_state["t_last"] > a.eval_min * 60:
                    start_eval()
            if stop["now"]:
                break
            # ---------------- update
            tu = time.time(); epf.flush()
            comp = np.nonzero(buf.complete[:buf.n])[0]
            adv = np.zeros(buf.n, np.float32); ret = np.zeros(buf.n, np.float32)
            for aid, rows in traj.items():
                rows = np.array(rows)
                if aid in final: boot = 0.0
                elif aid in pending: boot = float(buf.val[pending[aid]])
                elif aid in dropped_boot: boot = dropped_boot[aid]
                else: boot = float(buf.val[rows[-1]])        # vanished (restart): bootstrap with its own value
                last = 0.0; nxt = boot
                for j in rows[::-1]:
                    nonterm = 0.0 if buf.done[j] else 1.0
                    delta = buf.rew[j] + a.gamma * nxt * nonterm - buf.val[j]
                    last = delta + a.gamma * a.lam * nonterm * last
                    adv[j] = last; ret[j] = last + buf.val[j]; nxt = buf.val[j]
            T = len(comp)
            ix = torch.as_tensor(comp, device=dev)
            g = lambda arr, dt=None: torch.as_tensor(arr[comp], device=dev) if dt is None else torch.as_tensor(arr[comp], device=dev, dtype=dt)
            VF, VU = g(buf.vf).float(), g(buf.vu).float(); MT = g(buf.meta).long(); EF, EU = g(buf.ef), g(buf.eu)
            AC = g(buf.act).long(); LP = g(buf.logp); ADV = g(adv); RET = g(ret); VAL = g(buf.val)
            beta = max(a.kl_floor, a.kl_beta * (1 - upd / max(1, a.kl_updates))) if a.kl_updates > 0 else a.kl_beta
            warm = (upd - upd0) < a.value_warmup and not a.resume
            stats = []
            for _ in range(a.epochs):
                perm = torch.randperm(T, device=dev)
                for s0 in range(0, T - a.minibatch + 1, a.minibatch):   # equal sizes only: a new shape costs an MIOpen search
                    j = perm[s0:s0 + a.minibatch]
                    gf = gb.build(MT[j, 0], MT[j, 1], MT[j, 2], EF[j]); gu = gb.build(MT[j, 0], MT[j, 1], MT[j, 2], EU[j])
                    v = value_of(critic, VU[j], gu); vl = 0.5 * ((v - RET[j]) ** 2).mean()
                    copt.zero_grad(set_to_none=True); vl.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); copt.step()
                    if warm:
                        stats.append([0, vl.item(), 0, 0, 0]); continue
                    lg, _ = actor(VF[j], gf); lp, ent, _ = dist_logp_ent(lg, AC[j]); ratio = (lp - LP[j]).exp()
                    ad = ADV[j]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                    with torch.no_grad():
                        lr_, _ = ref(VF[j], gf)
                    kl = masked_kl(lr_, lg).mean()
                    loss = pg - a.ent * ent.mean() + beta * kl
                    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt.step()
                    stats.append([pg.item(), vl.item(), ent.mean().item(), kl.item(), ((ratio - 1).abs() > a.clip).float().mean().item()])
            with torch.no_grad():
                ev = 1 - float(((RET - VAL) ** 2).mean() / (RET.var() + 1e-8))
            upd += 1; t_u = time.time() - tu; t_upd_tot += t_u
            # carry pending rows over
            keep = sorted(set(pending.values()))
            newidx = buf.compact(keep); remap = dict(zip(keep, newidx.tolist()))
            pending = {aid: remap[j] for aid, j in pending.items()}; traj = {}; final = set(); dropped_boot = {}
            s = np.mean(stats, 0) if stats else np.zeros(5)
            rec = ep_rows[-400:]; done_rows = [r for r in rec if r["rank"] is not None and not r["truncated"]]
            by_k = {}
            for r in done_rows: by_k.setdefault(r["tag"], []).append(r["rank"])
            el = time.time() - t0
            row = {"update": upd, "agent_steps": agent_steps, "env_steps": env.stats["env_steps"], "transitions": T, "minutes": round(el / 60, 2),
                   "decisions_per_s": round(dec_n / max(1e-9, time.time() - dec_t), 1), "update_s": round(t_u, 2), "inference_s": round(t_inf, 2),
                   "pg": round(float(s[0]), 5), "vloss": round(float(s[1]), 4), "entropy": round(float(s[2]), 4), "kl_bc": round(float(s[3]), 5),
                   "clipfrac": round(float(s[4]), 4), "beta": round(beta, 4), "warmup": int(warm), "expl_var": round(ev, 4),
                   "episodes": len(ep_rows), "rank_recent": round(float(np.mean([r["rank"] for r in done_rows])), 3) if done_rows else None,
                   "ret_recent": round(float(np.mean([r["ret"] for r in rec])), 3) if rec else None, "restarts": env.stats["restarts"]}
            for kk in sorted(by_k): row[f"rank_{kk}"] = round(float(np.mean(by_k[kk])), 3)
            for tname in TERMS:
                row[f"t_{tname}"] = round(float(np.mean([r["terms"].get(tname, 0.0) for r in rec])), 4) if rec else None
            row["attack_frac"] = round(float(np.mean([r["attack_steps"] / max(1, r["steps"]) for r in rec])), 4) if rec else None
            if prog is None:
                progf = open(os.path.join(out, "progress.csv"), "a"); prog = csv.DictWriter(progf, fieldnames=list(row), extrasaction="ignore"); prog.writeheader()
            prog.writerow(row); progf.flush()
            print(json.dumps(row), flush=True)
            dec_t = time.time(); dec_n = 0; t_inf = 0.0
            if upd % 5 == 0:
                save(os.path.join(out, "ckpt_last.pt"))
    finally:
        save(os.path.join(out, "ckpt_last.pt"))
        env.close()
        json.dump({"updates": upd, "agent_steps": agent_steps, "env_steps": env.stats["env_steps"], "hours": round((time.time() - t0) / 3600, 3),
                   "restarts": env.stats["restarts"], "update_s_total": round(t_upd_tot, 1)}, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
