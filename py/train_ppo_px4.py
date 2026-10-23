#!/usr/bin/env python3
"""P3: PPO on pixels (pctrain round). Actor = the P2 pixel student (student4.PixelStudent4, privileged input OFF: pixels
only), critic = a state network on the FULL (unfair) state (asymmetric actor-critic; initialised from the PPO teacher's
trained critic). Client mode (maenv pixels=True), 1 vs 9 native, player 0 = the actor, training seeds >= 30,000,000.
Loss = PPO clip + entropy + beta * KL(pi_init || pi) (masked, beta decaying to a floor) + the understanding losses
(labels rendered from the state, as in P2) so the decoders keep working.
Resources (2026-10-05): the rollout lives in HOST RAM (frames linked into 4-frame stacks, as dagger4.Replay), batches are
gathered into pinned memory; actor / critic grid CNN run under bf16 autocast (acting and training alike, so the stored
log-probabilities match); fixed-size minibatches (a new shape costs an MIOpen search).
Held-out evaluation inside the run: every --eval-min minutes the eval seeds 1..N are played (frozen snapshot, transitions
not trained on) -> evals.jsonl.
  python3 train_ppo_px4.py --init ../runs/dag4_d/student_last.pt --critic-init ../runs/ppo4_a/ckpt_u0365.pt --run ppx4_a --hours 3
"""
import argparse, copy, csv, json, os, signal, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn"))
import torch_rocm_fix  # noqa: F401
import proj, state2 as S2
from models4 import make_state_policy, dist_logp_ent
from policy_rt import sample_heads
from student4 import PixelStudent4, aux_loss
from dagger4 import render_labels
from train_ppo4 import masked_kl, value_of
from nbpc.maenv import MAEnv, REWARD_W, TERMS
from eval_suite import summarize, EVAL_SEEDS

MAP_TILES = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "map_showdown.npy"))


class RollBuf:
    """host-RAM rollout rows: newest frame + links to the agent's previous 3 rows, labels, full state, action / logp / value"""
    def __init__(self, cap):
        z = lambda *s, dt=np.uint8: np.zeros(s, dt)
        self.cap = cap
        self.frame = z(cap, 115, 256, 3); self.ref = z(cap, 3, dt=np.int64)
        self.vu = z(cap, 436, dt=np.float16); self.gu = z(cap, 6, 25, 25)
        self.maps = z(cap, 5, 29, 64); self.heat = z(cap, 7, 29, 64); self.scal = z(cap, 5, dt=np.float32)
        self.act = z(cap, 4, dt=np.int64); self.logp = z(cap, dt=np.float32); self.val = z(cap, dt=np.float32)
        self.rew = z(cap, dt=np.float32); self.done = z(cap, dt=bool); self.complete = z(cap, dt=bool); self.n = 0
        self._ft = torch.from_numpy(self.frame)

    def gather(self, idx, pin, dev):
        B = len(idx); r = self.ref[idx]
        flat = torch.from_numpy(np.concatenate([r[:, 0], r[:, 1], r[:, 2], idx]).astype(np.int64))
        st = pin[:4 * B]; torch.index_select(self._ft, 0, flat, out=st)
        fr = st.to(dev, non_blocking=True).view(4, B, 115, 256, 3).permute(1, 2, 3, 0, 4).reshape(B, 115, 256, 12)
        g = {k: torch.from_numpy(getattr(self, k)[idx]).to(dev, non_blocking=True) for k in ("vu", "gu", "maps", "heat", "scal", "act", "logp")}
        return fr, g

    def compact(self, keep):
        keep = np.asarray(keep, np.int64); m = len(keep); remap = {int(o): i for i, o in enumerate(keep)}
        refs = self.ref[keep].copy()
        for a_ in (self.frame, self.vu, self.gu, self.maps, self.heat, self.scal, self.act, self.logp, self.val, self.rew, self.done, self.complete):
            a_[:m] = a_[keep]
        # kept rows' links must point into kept rows: a link to a dropped row becomes a link to the row itself
        self.ref[:m] = np.array([[remap.get(int(x), i) for x in row] for i, row in enumerate(refs)], np.int64).reshape(m, 3)
        self.n = m
        return remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--critic-init", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=3.0); ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--rollout", type=int, default=8192); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clr", type=float, default=2e-4); ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.001); ap.add_argument("--kl-beta", type=float, default=0.5)
    ap.add_argument("--kl-updates", type=int, default=40); ap.add_argument("--kl-floor", type=float, default=0.05)
    ap.add_argument("--value-warmup", type=int, default=2); ap.add_argument("--w-aux", type=float, default=0.5)
    ap.add_argument("--reward", default="{}"); ap.add_argument("--eval-min", type=float, default=60.0)
    ap.add_argument("--eval-episodes", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", a.run); os.makedirs(out, exist_ok=True)
    rw = dict(REWARD_W); rw.update(json.loads(a.reward))
    json.dump({**vars(a), "reward_w": rw}, open(os.path.join(out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda"); P = proj.load_params(); bf = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    actor = PixelStudent4().to(dev); actor.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    ref = copy.deepcopy(actor).eval(); snap = copy.deepcopy(actor).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    cck = torch.load(a.critic_init, map_location="cpu", weights_only=False)
    critic = make_state_policy(cck, dev); critic.grid_bf16 = True
    if "critic" in cck: critic.load_state_dict(cck["critic"])    # the PPO teacher's trained full-state critic
    else:
        nn.init.orthogonal_(critic.heads.v.weight, 1.0); nn.init.zeros_(critic.heads.v.bias)
    opt = torch.optim.Adam(actor.parameters(), lr=a.lr, eps=1e-5); copt = torch.optim.Adam(critic.parameters(), lr=a.clr, eps=1e-5)
    rng = np.random.default_rng(a.seed + 3); eval_queue = []; ev = {"active": False, "t_last": time.time()}

    def bfn(r_, i):
        if eval_queue: return {"seed": eval_queue.pop(0), "ctl": [0], "end": "player0", "tag": "eval"}
        return {"seed": int(rng.integers(30_000_000, 2 ** 31 - 1)), "ctl": [0], "end": "player0", "tag": "train"}
    env = MAEnv(a.n, bfn, reward_w=rw, fair=True, tag=f"ppx{os.getpid()}", verbose=True, pixels=True, want_full=True, seed=a.seed)
    M = S2.MapInfo(60, 60, MAP_TILES.tolist())
    buf = RollBuf(a.rollout + 1024); pin = torch.empty(4 * a.minibatch, 115, 256, 3, dtype=torch.uint8).pin_memory()
    pending, traj, final, last_slots, dboot = {}, {}, set(), {}, {}
    stop = {"now": False}; signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    t0 = time.time(); upd = 0; steps = 0; ep_rows = []; nseen = 0; prog = None
    epf = open(os.path.join(out, "episodes.jsonl"), "a"); evf = open(os.path.join(out, "evals.jsonl"), "a")

    def save(path):
        torch.save({"model": actor.state_dict(), "critic": critic.state_dict(), "update": upd, "env_steps": steps, "init": a.init,
                    "critic_init": a.critic_init}, path)

    def start_eval():
        snap.load_state_dict(actor.state_dict()); eval_queue.extend(EVAL_SEEDS[:a.eval_episodes])
        ev.update(active=True, update=upd, steps=steps, t0=time.time(), n0=len(env.episodes)); save(os.path.join(out, f"ckpt_u{upd:04d}.pt"))

    def finish_eval():
        rows = {r["seed"]: r for r in env.episodes[ev["n0"]:] if r["tag"] == "eval" and r["player"] == 0}
        if len(rows) < a.eval_episodes: return
        rows = list(rows.values())
        for r in rows: r["ctl_player"] = True
        sm = summarize(rows); sm.update(update=ev["update"], env_steps=ev["steps"], minutes=round((time.time() - t0) / 60, 1))
        evf.write(json.dumps(sm) + "\n"); evf.flush(); ev.update(active=False, t_last=time.time())
        print(f"[eval] update {sm['update']}: rank {sm['rank']['mean']:.3f} +- {sm['rank']['ci95']:.3f} win {sm['win']['mean']:.3f}", flush=True)

    start_eval()
    try:
        while time.time() - t0 < a.hours * 3600 and not stop["now"]:
            tc = time.time()
            while buf.n < a.rollout and not stop["now"]:
                obs, ids, tr = env.poll(min_ready=max(1, a.n // 2), max_wait=0.006)
                for (aid, rew, done, terms, rank) in tr:
                    j = pending.pop(aid, None)
                    if j is None: continue
                    buf.rew[j] = rew; buf.done[j] = done; buf.complete[j] = True; traj.setdefault(aid, []).append(j)
                    if done: final.add(aid)
                for aid in env.dropped:
                    j = pending.pop(aid, None)
                    if j is not None: dboot[aid] = float(buf.val[j])
                env.dropped = []
                if not len(ids): continue
                isev = np.array([t == "eval" for t in env.last_tags]); act = np.zeros((len(ids), 4), np.int64)
                with torch.no_grad():
                    px = torch.as_tensor(obs["pixels"], device=dev)
                    if isev.any():
                        with bf(): so = snap(px[torch.as_tensor(isev, device=dev)])
                        A, _ = sample_heads([x.float() for x in so["logits"]]); act[isev] = A.cpu().numpy()
                    m = ~isev
                    if m.any():
                        mt = torch.as_tensor(m, device=dev)
                        with bf(): o = actor(px[mt])
                        A, L = sample_heads([x.float() for x in o["logits"]])
                        vu = torch.as_tensor(env.last_full["vec"][m], device=dev); gu = torch.as_tensor(env.last_full["grid"][m], device=dev)
                        V = value_of(critic, vu, gu)
                        A, L, V = A.cpu().numpy(), L.cpu().numpy(), V.float().cpu().numpy(); act[m] = A
                        for q, b in enumerate(np.nonzero(m)[0]):
                            sb, cam, li, p = env.last_state[b]
                            if cam is None: continue
                            maps, heat, scal, _ = render_labels(sb, cam, p, M, P)
                            j = buf.n; buf.n += 1; aid = int(ids[b]); prev = last_slots.get(aid)
                            buf.frame[j] = obs["pixels"][b, :, :, -3:]; buf.ref[j] = prev if prev is not None else [j, j, j]
                            last_slots[aid] = [j, j, j] if prev is None else prev[1:] + [j]
                            buf.vu[j] = env.last_full["vec"][b]; buf.gu[j] = env.last_full["grid"][b]
                            buf.maps[j] = maps; buf.heat[j] = heat; buf.scal[j] = scal
                            buf.act[j] = A[q]; buf.logp[j] = L[q]; buf.val[j] = V[q]
                            buf.complete[j] = False; buf.done[j] = False; buf.rew[j] = 0
                            pending[aid] = j; steps += 1
                env.send(ids, act)
                for r in env.episodes[nseen:]:
                    if r["tag"] != "eval": ep_rows.append(r); epf.write(json.dumps(r) + "\n")
                nseen = len(env.episodes)
                if ev["active"]: finish_eval()
                elif time.time() - ev["t_last"] > a.eval_min * 60: start_eval()
            if stop["now"]: break
            t_col = time.time() - tc
            # ---------------- update
            tu = time.time(); n = buf.n
            adv = np.zeros(n, np.float32); ret = np.zeros(n, np.float32)
            for aid, rows in traj.items():
                boot = 0.0 if aid in final else (float(buf.val[pending[aid]]) if aid in pending else dboot.get(aid, float(buf.val[rows[-1]])))
                last = 0.0; nxt = boot
                for j in rows[::-1]:
                    nt = 0.0 if buf.done[j] else 1.0
                    delta = buf.rew[j] + a.gamma * nxt * nt - buf.val[j]; last = delta + a.gamma * a.lam * nt * last
                    adv[j] = last; ret[j] = last + buf.val[j]; nxt = buf.val[j]
            comp = np.nonzero(buf.complete[:n])[0]; T = len(comp)
            beta = max(a.kl_floor, a.kl_beta * (1 - upd / max(1, a.kl_updates))); warm = upd < a.value_warmup; stats = []
            for _ in range(a.epochs):
                perm = np.random.permutation(comp)
                for s0 in range(0, T - a.minibatch + 1, a.minibatch):
                    idx = perm[s0:s0 + a.minibatch]
                    fr, g = buf.gather(idx, pin, dev)
                    RET = torch.as_tensor(ret[idx], device=dev); ADV = torch.as_tensor(adv[idx], device=dev)
                    v = value_of(critic, g["vu"].float(), g["gu"]); vl = 0.5 * ((v.float() - RET) ** 2).mean()
                    copt.zero_grad(set_to_none=True); vl.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); copt.step()
                    if warm: stats.append([0, vl.item(), 0, 0, 0, 0]); continue
                    with bf(): o = actor(fr)
                    lg = [x.float() for x in o["logits"]]
                    lp, ent, _ = dist_logp_ent(lg, g["act"]); ratio = (lp - g["logp"]).exp()
                    ad = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                    with torch.no_grad(), bf(): rl = [x.float() for x in ref(fr)["logits"]]
                    kl = masked_kl(rl, lg).mean()
                    lm, lh, ls = aux_loss(o, g["maps"], g["heat"].float() / 255.0, g["scal"])
                    loss = pg - a.ent * ent.mean() + beta * kl + a.w_aux * (lm + lh + ls)
                    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt.step()
                    stats.append([pg.item(), vl.item(), ent.mean().item(), kl.item(), ((ratio - 1).abs() > a.clip).float().mean().item(), (lm + lh + ls).item()])
            upd += 1
            keep = sorted(set(pending.values()))
            remap = buf.compact(keep); pending = {aid: remap[j] for aid, j in pending.items()}
            last_slots = {aid: [remap.get(x, pending[aid]) for x in v] for aid, v in last_slots.items() if aid in pending}
            traj, final, dboot = {}, set(), {}
            s = np.mean(stats, 0) if stats else np.zeros(6)
            rec = ep_rows[-200:]; okr = [r for r in rec if r["rank"] is not None and not r["truncated"]]
            row = {"update": upd, "env_steps": steps, "minutes": round((time.time() - t0) / 60, 2), "transitions": T,
                   "steps_per_s_collect": round(T / max(1e-9, t_col), 1), "update_s": round(time.time() - tu, 1),
                   "pg": round(float(s[0]), 5), "vloss": round(float(s[1]), 4), "entropy": round(float(s[2]), 4), "kl_init": round(float(s[3]), 5),
                   "clipfrac": round(float(s[4]), 4), "aux": round(float(s[5]), 4), "beta": round(beta, 4), "episodes": len(ep_rows),
                   "rank_recent": round(float(np.mean([r["rank"] for r in okr])), 3) if okr else None, "restarts": env.stats["restarts"],
                   "min_avail_mb": env.stats.get("min_avail_mb"), "max_vram_mb": env.stats.get("max_vram_mb")}
            for tn in TERMS: row[f"t_{tn}"] = round(float(np.mean([r["terms"].get(tn, 0.0) for r in rec])), 4) if rec else None
            if prog is None:
                progf = open(os.path.join(out, "progress.csv"), "a"); prog = csv.DictWriter(progf, fieldnames=list(row), extrasaction="ignore"); prog.writeheader()
            prog.writerow(row); progf.flush(); print(json.dumps(row), flush=True); epf.flush()
            if upd % 5 == 0: save(os.path.join(out, "ckpt_last.pt"))
    finally:
        save(os.path.join(out, "ckpt_last.pt")); env.close()


if __name__ == "__main__":
    main()
