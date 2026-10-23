#!/usr/bin/env python3
"""PPO on the PC vector env (state-only, --n instances), the learn round's train_ppo_state.py recipe on a batch:
policy initialised from a BC checkpoint (v3 pointer heads), separate critic (copy of the BC net, value head from scratch,
critic-only warm-up updates), loss = PPO clip + value + entropy + beta * KL(pi_BC || pi) with beta decaying linearly.
Rewards: train_ppo_state.REWARD weights; Solo Showdown, 9 native bots (stage C; --n-active-bots for a curriculum stage).

Transitions per slot: the vector env masks slots that are resetting (info["valid"] / ["new_episode"]), so each slot
keeps its own trajectory; GAE per slot, bootstrap from the slot's current observation at the rollout end.
Logs progress.csv per update (env steps/s inside the loop, rollout/update seconds, GPU busy %, losses, episode stats).

  python3 train_ppo_pc.py --init ../../inputs/checkpoints/bc3p_v3_pointer.pt --steps 1000000 --run ppo_smoke
"""
import argparse, copy, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_rocm_fix  # noqa: F401
from models import dist_logp_ent, nvec_of, is_v3, make_state_policy
from nbpc.vecenv import NBVecEnvPC

REWARD = {"damage_dealt": 0.5, "damage_taken": -0.3, "box_damage": 0.1, "cubes": 0.5, "kills": 1.0, "gas_ticks": -0.02,
          "placement": 5.0, "win": 5.0, "step": 0.0, "gas_shaping": 0.3}   # = train_ppo_state.REWARD


def kl_heads(lg_new, lg_ref):
    tot = 0.0
    for a, b in zip(lg_new, lg_ref):
        pa = torch.log_softmax(a, -1); pb = torch.log_softmax(b, -1)
        tot = tot + (pb.exp() * (pb - pa)).sum(-1)
    return tot


def gpu_busy():
    try: return int(open("/sys/class/drm/card1/device/gpu_busy_percent").read())
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo_pc"); ap.add_argument("--init", required=True)
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--rollout", type=int, default=8192, help="transitions per update (all slots)")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95); ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.003); ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--kl-beta", type=float, default=0.5); ap.add_argument("--kl-updates", type=int, default=150)
    ap.add_argument("--value-warmup", type=int, default=3); ap.add_argument("--n-active-bots", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "t7", a.run); os.makedirs(out, exist_ok=True)
    json.dump({**vars(a), "reward": REWARD}, open(os.path.join(out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck0 = torch.load(a.init, map_location="cpu", weights_only=False); nvec = nvec_of(ck0)
    net = make_state_policy(ck0, dev); ref = copy.deepcopy(net).eval(); critic = copy.deepcopy(net)
    for p in ref.parameters(): p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5); copt = torch.optim.Adam(critic.parameters(), lr=a.lr * 3, eps=1e-5)
    t_boot = time.time()
    env = NBVecEnvPC(a.n, reward_weights=REWARD, n_active_bots=a.n_active_bots, max_episode_ticks=4000,
                     action_version=3 if is_v3(nvec) else 2, tag=f"ppo{os.getpid()}", verbose=True, seed0=a.seed)
    boot_s = time.time() - t_boot
    obs, info = env.reset()
    N = a.n
    cur = {"vec": obs["vec"].copy(), "grid": obs["grid"].copy()}
    traj = [[] for _ in range(N)]   # per slot: [vec, grid, act, logp, value, reward, done]
    prog = open(os.path.join(out, "progress.csv"), "w"); pw = None; epf = open(os.path.join(out, "episodes.jsonl"), "w")
    step = 0; upd = 0; t0 = time.time(); ep_rows = []
    try:
        while step < a.steps:
            tr0 = time.time(); n_tr = 0; t_pol = 0.0; t_env = 0.0; gb = []
            while n_tr < a.rollout:
                tp = time.time()
                with torch.no_grad():
                    v = torch.as_tensor(cur["vec"], device=dev); g = torch.as_tensor(cur["grid"], device=dev)
                    logits, _ = net(v, g); _, val = critic(v, g); lp, _, act = dist_logp_ent(logits)
                act_np = act.cpu().numpy(); lp_np = lp.cpu().numpy(); val_np = val.cpu().numpy()
                t_pol += time.time() - tp
                te = time.time(); o2, r, term, trunc, inf = env.step(act_np); t_env += time.time() - te
                for i in range(N):
                    if not inf["valid"][i]:
                        continue
                    if not inf["new_episode"][i]:   # the action given for slot i was applied: a transition
                        traj[i].append((cur["vec"][i].copy(), cur["grid"][i].copy(), act_np[i], lp_np[i], val_np[i], float(r[i]), bool(term[i] or trunc[i])))
                        n_tr += 1; step += 1
                        e = inf["episode"][i]
                        if e is not None:
                            row = {"step": step, "rank": e.get("rank"), "ret": e.get("r"), "len": e.get("l"), "ticks": e.get("ticks"),
                                   "restart": bool(inf["session_restart"][i])}
                            ep_rows.append(row); epf.write(json.dumps(row) + "\n")
                    cur["vec"][i] = o2["vec"][i]; cur["grid"][i] = o2["grid"][i]
                if len(gb) < 2000: gb.append(gpu_busy())
            t_roll = time.time() - tr0
            # GAE per slot; keep nothing across updates (on-policy): the last transition of a non-terminated slot is bootstrapped
            tu = time.time()
            with torch.no_grad():
                _, vnow = critic(torch.as_tensor(cur["vec"], device=dev), torch.as_tensor(cur["grid"], device=dev))
            vnow = vnow.cpu().numpy()
            V, G, A, LP, ADV, RET = [], [], [], [], [], []
            for i in range(N):
                tj = traj[i]
                if not tj: continue
                last = 0.0
                adv = np.zeros(len(tj), np.float32)
                for t in reversed(range(len(tj))):
                    nxt = (vnow[i] if t == len(tj) - 1 else tj[t + 1][4]); nonterm = 0.0 if tj[t][6] else 1.0
                    delta = tj[t][5] + a.gamma * nxt * nonterm - tj[t][4]
                    last = delta + a.gamma * a.lam * nonterm * last; adv[t] = last
                for t, x in enumerate(tj):
                    V.append(x[0]); G.append(x[1]); A.append(x[2]); LP.append(x[3]); ADV.append(adv[t]); RET.append(adv[t] + x[4])
                traj[i] = []
            vb = torch.as_tensor(np.stack(V), device=dev); gbt = torch.as_tensor(np.stack(G), device=dev)
            ab = torch.as_tensor(np.stack(A), device=dev, dtype=torch.long); lpb = torch.as_tensor(np.array(LP), device=dev)
            advb = torch.as_tensor(np.array(ADV), device=dev); retb = torch.as_tensor(np.array(RET), device=dev)
            T = len(vb); beta = a.kl_beta * max(0.0, 1.0 - upd / max(1, a.kl_updates)); warm = upd < a.value_warmup
            stats = []; idx = np.arange(T)
            for _ in range(a.epochs):
                np.random.shuffle(idx)
                for s0 in range(0, T, a.minibatch):
                    if T - s0 < a.minibatch // 2: continue   # no tiny trailing minibatch (a 1-sample batch has no advantage std)
                    j = torch.as_tensor(idx[s0:s0 + a.minibatch], device=dev)
                    logits, _ = net(vb[j], gbt[j]); _, vv = critic(vb[j], gbt[j])
                    lp, ent, _ = dist_logp_ent(logits, ab[j]); ratio = (lp - lpb[j]).exp()
                    ad = advb[j]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                    vl = 0.5 * ((vv - retb[j]) ** 2).mean()
                    with torch.no_grad():
                        lr_ref, _ = ref(vb[j], gbt[j])
                    kl = kl_heads(logits, lr_ref).mean()
                    copt.zero_grad(); (a.vf * vl).backward(); nn.utils.clip_grad_norm_(critic.parameters(), 0.5); copt.step()
                    if not warm:
                        loss = pg - a.ent * ent.mean() + beta * kl
                        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                    stats.append([pg.item(), vl.item(), ent.mean().item(), kl.item(), ((ratio - 1).abs() > a.clip).float().mean().item()])
                    if len(gb) < 4000: gb.append(gpu_busy())
            t_upd = time.time() - tu; upd += 1
            s = np.mean(stats, 0); recent = ep_rows[-50:]; ranks = [x["rank"] for x in recent if x["rank"] is not None]
            g_ok = [x for x in gb if x is not None]
            row = {"update": upd, "env_steps": step, "transitions": T, "steps_per_s_rollout": round(T / t_roll, 1),
                   "steps_per_s_overall": round(step / (time.time() - t0), 1), "rollout_s": round(t_roll, 2), "update_s": round(t_upd, 2),
                   "policy_s": round(t_pol, 2), "env_s": round(t_env, 2), "gpu_busy_mean": round(float(np.mean(g_ok)), 1) if g_ok else None,
                   "pg": round(s[0], 4), "vloss": round(s[1], 4), "entropy": round(s[2], 4), "kl_bc": round(s[3], 4), "clipfrac": round(s[4], 4),
                   "beta": round(beta, 4), "warmup": int(warm), "episodes": len(ep_rows), "recent_mean_rank": round(float(np.mean(ranks)), 3) if ranks else None,
                   "recent_mean_ret": round(float(np.mean([x["ret"] for x in recent])), 3) if recent else None, "restarts": env.stats["restarts"],
                   "minutes": round((time.time() - t0) / 60, 2)}
            if pw is None:
                pw = csv.DictWriter(prog, fieldnames=list(row)); pw.writeheader()
            pw.writerow(row); prog.flush(); epf.flush()
            print(json.dumps(row), flush=True)
            torch.save({"model": net.state_dict(), "critic": critic.state_dict(), "step": step, "update": upd, "vec_dim": ck0["vec_dim"],
                        "grid_shape": ck0["grid_shape"], "nvec": nvec, "pointer_aim": net.pointer_aim,
                        "pointer_lead": ck0.get("pointer_lead", False)}, os.path.join(out, "ckpt_last.pt"))
    finally:
        env.close()
        json.dump({"env_steps": step, "updates": upd, "minutes": round((time.time() - t0) / 60, 2), "boot_s": round(boot_s, 1),
                   "steps_per_s_overall": round(step / max(1e-9, time.time() - t0), 1), "episodes": len(ep_rows), "restarts": env.stats["restarts"]},
                  open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
