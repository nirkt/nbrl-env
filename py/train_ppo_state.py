#!/usr/bin/env python3
"""S2: PPO fine-tuning of the BC policy on the privileged state (fast state-only mode), with a bot curriculum and a
KL regulariser toward the frozen BC policy.

  python3 work/train_ppo_state.py --run ppo1 --init work/runs/bc1/bc_best.pt --minutes 120

Curriculum (bot-control API): stage A = the 9 bots hold still, B = 3 native bots (6 hold), C = 9 native bots.
Promotion when the mean placement (rank) of the last --promote-window episodes is <= the stage threshold
(A: 2.0, B: 3.5); the stage and its promotion log are saved in the checkpoint and progress.csv.
Loss = PPO clip + value + entropy + beta * KL(pi_BC || pi) (per-head categorical KL, summed); beta decays linearly
from --kl-beta to 0 over --kl-updates updates. The critic is a SEPARATE network (same architecture, initialised from
the BC weights, its value head from scratch): a fresh value head on the shared trunk moved the policy during the
warm-up (first try: entropy 1.98 -> 2.91, KL to BC 0.08 -> 0.49 in 3 critic-only updates). The first --value-warmup
updates train the critic only. Rewards: nb_env terms re-weighted (REWARD below; every term logged).
"""
import argparse, copy, csv, json, os, signal, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_rocm_fix  # noqa: F401
from models import StatePolicy, dist_logp_ent, NVEC, nvec_of, is_v3, make_state_policy
from nb_env import NBEnv

REWARD = {"damage_dealt": 0.5, "damage_taken": -0.3, "box_damage": 0.1, "cubes": 0.5, "kills": 1.0, "gas_ticks": -0.02,
          "placement": 5.0, "win": 5.0, "step": 0.0, "gas_shaping": 0.3}
STAGES = [("A", 0, 2.0), ("B", 3, 5.0), ("C", 9, None)]   # B threshold 3.5 -> 5.0 at update 90 (time budget; see NOTES)


def kl_heads(lg_new, lg_ref):
    tot = 0.0
    for a, b in zip(lg_new, lg_ref):
        pa = torch.log_softmax(a, -1); pb = torch.log_softmax(b, -1)
        tot = tot + (pb.exp() * (pb - pa)).sum(-1)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ppo1")
    ap.add_argument("--init", required=True)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--num-steps", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--kl-beta", type=float, default=0.5)
    ap.add_argument("--kl-updates", type=int, default=150)
    ap.add_argument("--value-warmup", type=int, default=3)
    ap.add_argument("--promote-window", type=int, default=20)
    ap.add_argument("--start-stage", type=int, default=0)
    ap.add_argument("--fixed-stage", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--serial", default="RF8R90KCNQH")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", a.run); os.makedirs(out, exist_ok=True)
    json.dump({**vars(a), "reward": REWARD, "stages": STAGES}, open(os.path.join(out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
    nvec = nvec_of(ck0)
    net = make_state_policy(ck0, dev)
    ref = copy.deepcopy(net).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    critic = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    copt = torch.optim.Adam(critic.parameters(), lr=a.lr * 3, eps=1e-5)
    stage = a.start_stage if a.fixed_stage is None else a.fixed_stage
    step0 = upd0 = 0; promotions = []; recent = []
    if a.resume and os.path.exists(os.path.join(out, "ckpt_last.pt")):
        ck = torch.load(os.path.join(out, "ckpt_last.pt"), map_location=dev)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step0, upd0 = ck["step"], ck["update"]
        if "critic" in ck: critic.load_state_dict(ck["critic"]); copt.load_state_dict(ck["copt"])
        stage = ck.get("stage", stage); promotions = ck.get("promotions", []); recent = ck.get("recent", [])
        print("resumed", step0, upd0, "stage", STAGES[stage][0], flush=True)

    def make_env(st):
        return NBEnv(serial=a.serial, roster=["Sniper"] * 9, gas=True, obs_mode="state", n_active_bots=STAGES[st][1],
                     reward_weights=REWARD, bot_control=True, max_episode_ticks=4000, action_version=3 if is_v3(nvec) else 2)
    env = make_env(stage)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *x: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *x: stop.__setitem__("flag", True))
    vn = ck0["vec_dim"]; gs = tuple(ck0["grid_shape"])
    T = a.num_steps
    vb = torch.zeros((T, vn), device=dev); gb = torch.zeros((T,) + gs, device=dev)
    ab = torch.zeros((T, 4), dtype=torch.long, device=dev); lpb = torch.zeros(T, device=dev)
    rb = torch.zeros(T, device=dev); db = torch.zeros(T, device=dev); valb = torch.zeros(T, device=dev)
    prog = open(os.path.join(out, "progress.csv"), "a"); pw = csv.writer(prog); wrote_hdr = os.path.getsize(os.path.join(out, "progress.csv")) > 0
    epf = open(os.path.join(out, "episodes.jsonl"), "a")
    obs, info = env.reset(seed=int(np.random.randint(200000, 2 ** 30)))
    t_start = time.time(); step = step0; upd = upd0

    def save(tag):
        ck = {"model": net.state_dict(), "opt": opt.state_dict(), "critic": critic.state_dict(), "copt": copt.state_dict(), "step": step, "update": upd, "stage": stage,
              "promotions": promotions, "recent": recent, "vec_dim": vn, "grid_shape": list(gs), "args": vars(a), "nvec": nvec, "pointer_aim": net.pointer_aim}
        torch.save(ck, os.path.join(out, f"{tag}.pt"))
    try:
        while time.time() - t_start < a.minutes * 60 and not stop["flag"]:
            t0 = time.time(); ep_rows = []
            for t in range(T):
                v = torch.as_tensor(obs["vec"], device=dev)[None]; g = torch.as_tensor(obs["grid"], device=dev)[None]
                with torch.no_grad():
                    logits, _ = net(v, g)
                    _, val = critic(v, g)
                    lp, _, act = dist_logp_ent(logits)
                vb[t] = v[0]; gb[t] = g[0]; ab[t] = act[0]; lpb[t] = lp[0]; valb[t] = val[0]
                obs, r, term, trunc, info = env.step(act[0].cpu().numpy())
                rb[t] = r; db[t] = float(term or trunc); step += 1
                if term or trunc:
                    e = info.get("episode") or {}
                    row = {"step": step, "stage": STAGES[stage][0], "rank": e.get("rank"), "ret": e.get("r"), "len": e.get("l"),
                           "ticks": e.get("ticks"), "terms": e.get("terms"), "restart": bool(info.get("session_restart"))}
                    epf.write(json.dumps(row) + "\n"); epf.flush(); ep_rows.append(row)
                    if e.get("rank") is not None and not info.get("session_restart"):
                        recent.append(e["rank"]); recent = recent[-a.promote_window:]
                    obs, info = env.reset(seed=int(np.random.randint(200000, 2 ** 30)))
                    # curriculum promotion
                    thr = STAGES[stage][2]
                    if a.fixed_stage is None and thr is not None and len(recent) >= a.promote_window and np.mean(recent) <= thr:
                        promotions.append({"from": STAGES[stage][0], "to": STAGES[stage + 1][0], "step": step, "update": upd,
                                           "mean_rank": float(np.mean(recent))})
                        print("PROMOTE", promotions[-1], flush=True)
                        stage += 1; recent = []
                        env.n_active = STAGES[stage][1]
            t_roll = time.time() - t0
            with torch.no_grad():
                _, nv = critic(torch.as_tensor(obs["vec"], device=dev)[None], torch.as_tensor(obs["grid"], device=dev)[None])
                adv = torch.zeros(T, device=dev); last = 0.0
                for t in reversed(range(T)):
                    nxt = nv[0] if t == T - 1 else valb[t + 1]
                    nonterm = 1.0 - db[t]
                    delta = rb[t] + a.gamma * nxt * nonterm - valb[t]
                    last = delta + a.gamma * a.lam * nonterm * last; adv[t] = last
                ret = adv + valb
            beta = a.kl_beta * max(0.0, 1.0 - (upd - upd0 if False else upd) / max(1, a.kl_updates))
            warm = upd < a.value_warmup
            idx = np.arange(T); stats = []
            for _ in range(a.epochs):
                np.random.shuffle(idx)
                for s0 in range(0, T, a.minibatch):
                    j = torch.as_tensor(idx[s0:s0 + a.minibatch], device=dev)
                    logits, _ = net(vb[j], gb[j])
                    _, v = critic(vb[j], gb[j])
                    lp, ent, _ = dist_logp_ent(logits, ab[j])
                    ratio = (lp - lpb[j]).exp()
                    ad = adv[j]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - a.clip, 1 + a.clip)).mean()
                    vl = 0.5 * ((v - ret[j]) ** 2).mean()
                    with torch.no_grad():
                        lr_ref, _ = ref(vb[j], gb[j])
                    kl = kl_heads(logits, lr_ref).mean()
                    copt.zero_grad(); (a.vf * vl).backward(); nn.utils.clip_grad_norm_(critic.parameters(), 0.5); copt.step()
                    if not warm:
                        loss = pg - a.ent * ent.mean() + beta * kl
                        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                    with torch.no_grad():
                        stats.append([pg.item(), vl.item(), ent.mean().item(), kl.item(), ((ratio - 1).abs() > a.clip).float().mean().item(),
                                      ((ratio - 1) - ratio.log()).mean().item()])
            upd += 1
            s = np.mean(stats, 0)
            ranks = [r_["rank"] for r_ in ep_rows if r_["rank"] is not None]
            row = {"update": upd, "step": step, "stage": STAGES[stage][0], "sps": T / t_roll, "pg": s[0], "vloss": s[1], "entropy": s[2],
                   "kl_bc": s[3], "clipfrac": s[4], "approx_kl": s[5], "beta": beta, "warmup": int(warm), "episodes": len(ep_rows),
                   "mean_rank": float(np.mean(ranks)) if ranks else None, "recent_mean_rank": float(np.mean(recent)) if recent else None,
                   "mean_ret": float(np.mean([r_["ret"] for r_ in ep_rows])) if ep_rows else None, "restarts": env.stats["restarts"],
                   "minutes": (time.time() - t_start) / 60}
            if not wrote_hdr:
                pw.writerow(list(row)); wrote_hdr = True
            pw.writerow([row[k] for k in row]); prog.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            save("ckpt_last")
            if upd % 10 == 0:
                save(f"ckpt_{upd:05d}")
    finally:
        save("ckpt_last"); env.close()
        print("stopped at step", step, "update", upd, flush=True)


if __name__ == "__main__":
    main()
