#!/usr/bin/env python3
"""Held-out evaluation suite (pctrain round, P0). Solo Showdown, 10 Pipers, storm on, max 4000 ticks, sampled actions.

Protocols
  1v9   : player 0 = the policy, players 1-9 native; the episode = player 0's life (rank at its elimination). Gate eval.
          Seeds 1..500 (EVAL_SEEDS, never used in training: training/demo seeds are >= 1,000,000).
  mixed : screening. k players of each battle (drawn from the battle seed) are the policy (or, for native_lead, the lead
          players), the other 10-k native; the battle runs to its end; per battle d = mean rank(policy players) -
          mean rank(native players); reported mean d with 95 % CI. Seeds 200,001.. (MIXED_SEED0).
Policies: native | native_lead (the native AI with lead aim, work/loader lead_aim) | state:CKPT (models4.StatePolicy)
Metrics per controlled episode: rank, win, survival ticks, damage dealt / taken, kills, cubes, storm deaths, hits / shots
(hit rate; shots = ammo drops), attack fraction (policy: attack actions per step; native: shots per step).

  python3 eval_suite.py --policy native --protocol 1v9 --out ../runs/eval/native_1v9
  python3 eval_suite.py --policy state:../runs/bc4/bc_best.pt --protocol 1v9 --out ../runs/eval/bc4_1v9
  python3 eval_suite.py --policy native_lead --protocol mixed --k 5 --battles 200 --out ../runs/eval/lead_mixed5
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.maenv import MAEnv

EVAL_SEEDS = list(range(1, 501))
MIXED_SEED0 = 200001


def ci95(x):
    x = np.asarray([v for v in x if v is not None], np.float64)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def load_policy(spec, fire_shift=0.0, deterministic=False):
    """spec: state:CKPT[@lead=F] (F: pointer candidates at delay-aware intercept angles with lead fraction F)"""
    import torch, torch_rocm_fix  # noqa: F401
    from models4 import make_state_policy
    from policy_rt import GraphPolicy
    path = spec.split(":", 1)[1]; dl = None
    if "@lead=" in path:
        path, dl = path.split("@lead="); dl = float(dl)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = make_state_policy(ck, dev, delay_lead=dl).eval()
    return GraphPolicy(net, None, dev=dev, fire_shift=fire_shift, deterministic=deterministic), ck


class PixelPolicy:
    """student4.PixelStudent4, pixels only (privileged side input off)"""
    def __init__(self, path, deterministic=False):
        import torch, torch_rocm_fix  # noqa: F401
        from student4 import PixelStudent4
        from policy_rt import sample_heads
        self.t, self.sh, self.det = torch, sample_heads, deterministic
        self.dev = torch.device("cuda")
        self.net = PixelStudent4().to(self.dev).eval()
        self.net.load_state_dict(torch.load(path, map_location=self.dev, weights_only=False)["model"])

    def act(self, px):
        with self.t.no_grad():
            o = self.net(self.t.as_tensor(px, device=self.dev))
            A, _ = self.sh(o["logits"], deterministic=self.det)
        return A.cpu().numpy()


def episode_metrics(r):
    t = r["terms"]; steps = max(1, r["steps"])
    shots = t.get("shots", 0.0)
    return {"rank": r["rank"], "win": r["win"], "ticks": r["ticks"], "dealt": 1000 * t.get("dealt", 0.0), "taken": 1000 * t.get("taken", 0.0),
            "kills": t.get("kills", 0.0), "cubes": t.get("cubes", 0.0), "storm_death": r["storm_death"], "gas_ticks": t.get("gas", 0.0),
            "hits": t.get("hits", 0.0), "shots": shots, "hit_rate": (t.get("hits", 0.0) / shots) if shots > 0 else None,
            "attack_frac": (r["attack_steps"] / steps) if r.get("ctl_player") else shots / steps, "steps": r["steps"]}


def summarize(rows):
    ok = [r for r in rows if r["rank"] is not None and not r["truncated"]]
    out = {"episodes": len(rows), "valid": len(ok), "truncated": sum(1 for r in rows if r["truncated"])}
    M = [episode_metrics(r) for r in ok]
    for k in ("rank", "win", "ticks", "dealt", "taken", "kills", "cubes", "storm_death", "gas_ticks", "hit_rate", "attack_frac", "steps"):
        m, h = ci95([x[k] for x in M]); out[k] = {"mean": round(m, 4), "ci95": round(h, 4)}
    hs, ss = sum(x["hits"] for x in M), sum(x["shots"] for x in M)
    out["hit_rate_pooled"] = round(hs / ss, 4) if ss else None
    out["rank_hist"] = {str(k): sum(1 for x in M if x["rank"] == k) for k in range(1, 11)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--protocol", choices=["1v9", "mixed"], default="1v9")
    ap.add_argument("--episodes", type=int, default=500, help="1v9: seeds 1..N"); ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--battles", type=int, default=200)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", required=True)
    ap.add_argument("--fire-shift", type=float, default=0.0); ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--lead-dt", type=float, default=0.0); ap.add_argument("--lead-speed", type=float, default=197.0)
    ap.add_argument("--opp-lead", action="store_true", help="the native opponents use lead aim too")
    ap.add_argument("--lead-dt-list", default=None, help="mixed native_lead: battles cycle through these lead dt values (tuning)")
    ap.add_argument("--lead-frac-list", default=None, help="mixed native_lead: battles cycle through these lead fractions (tuning)")
    ap.add_argument("--fair", type=int, default=1); ap.add_argument("--max-ticks", type=int, default=4000)
    ap.add_argument("--client", action="store_true", help="client mode (renderer) also for a state policy")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    kind = a.policy.split(":")[0]
    pol = None
    if kind == "state":
        pol, ck = load_policy(a.policy, a.fire_shift, a.deterministic)
    elif kind == "pixel":
        pol = PixelPolicy(a.policy.split(":", 1)[1], a.deterministic)
    pixels = kind == "pixel" or a.client
    opp_lead = 0x3ff if a.opp_lead else 0
    if a.protocol == "1v9":
        seeds = list(range(a.seed0, a.seed0 + a.episodes)); queue = list(seeds)
        def battle_fn(rng, i):
            if not queue:
                return None
            sd = queue.pop(0)
            if kind in ("state", "pixel"):
                return {"seed": sd, "ctl": [0], "lead_mask": opp_lead & ~1, "end": "player0", "tag": "eval"}
            return {"seed": sd, "ctl": [], "watch": [0], "lead_mask": (1 if kind == "native_lead" else 0) | (opp_lead & ~1), "end": "player0", "tag": "eval"}
        want = len(seeds)
    else:
        seeds = list(range(MIXED_SEED0, MIXED_SEED0 + a.battles)); queue = list(seeds)
        dts = [float(x) for x in a.lead_dt_list.split(",")] if a.lead_dt_list else None
        fracs = [float(x) for x in a.lead_frac_list.split(",")] if a.lead_frac_list else None
        def battle_fn(rng, i):
            if not queue:
                return None
            sd = queue.pop(0); S = sorted(np.random.default_rng(sd).choice(10, a.k, replace=False).tolist())
            others = [p for p in range(10) if p not in S]
            if kind == "state":
                return {"seed": sd, "ctl": S, "watch": others, "lead_mask": opp_lead, "end": "done", "tag": "eval"}
            cfg = {"seed": sd, "ctl": [], "watch": list(range(10)), "lead_mask": (sum(1 << p for p in S) if kind == "native_lead" else 0) | opp_lead,
                   "end": "done", "tag": "eval", "S": S}
            if dts:
                cfg["lead_dt"] = dts[(sd - MIXED_SEED0) % len(dts)]; cfg["tag"] = "eval"
            if fracs:
                cfg["lead_frac"] = fracs[(sd - MIXED_SEED0) % len(fracs)]
            return cfg
        want = a.battles * 10
    t0 = time.time()
    env = MAEnv(a.n, battle_fn, tag=f"ev{os.getpid()}", fair=bool(a.fair), verbose=True, lead_dt=a.lead_dt, lead_speed=a.lead_speed,
                max_ticks=a.max_ticks, pixels=pixels)
    boot = time.time() - t0; t1 = time.time(); f = open(os.path.join(a.out, "episodes.jsonl"), "w"); nseen = 0; dec = 0; t_pol = 0.0
    try:
        while True:
            obs, ids, tr = env.poll(min_ready=max(1, a.n // 3), max_wait=0.003)
            if len(ids):
                tp = time.time()
                if kind == "pixel":
                    act = pol.act(obs["pixels"])
                elif pol is not None:
                    act, _, _ = pol.act(obs["vec"], obs["grid"])
                else:
                    act = np.zeros((len(ids), 4), np.int64)
                t_pol += time.time() - tp; dec += len(ids)
                env.send(ids, act)
            new = [r for r in env.episodes[nseen:] if r["tag"] == "eval"]; nseen = len(env.episodes)
            for r in new:
                r["ctl_player"] = r["player"] in r["ctl"]
                f.write(json.dumps(r) + "\n")
            f.flush()
            done_rows = [r for r in env.episodes if r["tag"] == "eval"]
            if a.protocol == "1v9":
                n_done = len({r["seed"] for r in done_rows if r["player"] == 0})
            else:
                n_done = len(done_rows)
            if n_done >= want or all(x.state == "idle" for x in env.I):
                break
            if nseen and nseen % 100 == 0 and new:
                print(f"[eval] {n_done}/{want} after {time.time() - t1:.0f} s", flush=True)
    finally:
        env.close()
    dt = time.time() - t1
    rows = [r for r in env.episodes if r["tag"] == "eval"]
    for r in rows: r["ctl_player"] = r["player"] in r["ctl"]
    summ = {"policy": a.policy, "protocol": a.protocol, "minutes": round(dt / 60, 2), "boot_s": round(boot, 1), "instances": a.n,
            "env_steps": env.stats["env_steps"], "env_steps_per_s": round(env.stats["env_steps"] / dt, 1), "decisions": dec,
            "policy_s": round(t_pol, 1), "restarts": env.stats["restarts"], "fire_shift": a.fire_shift, "fair": a.fair}
    if a.protocol == "1v9":
        summ.update(summarize([r for r in rows if r["player"] == 0]))
    else:
        if kind == "state":
            pol_rows = [r for r in rows if r["ctl_player"]]; nat_rows = [r for r in rows if not r["ctl_player"]]
        else:
            pol_rows = [r for r in rows if (r["lead_mask"] >> r["player"]) & 1]; nat_rows = [r for r in rows if not (r["lead_mask"] >> r["player"]) & 1]
        summ["policy_players"] = summarize(pol_rows); summ["native_players"] = summarize(nat_rows)
        by = {}
        for r in rows:
            if r["rank"] is None or r["truncated"]: continue
            isp = r["ctl_player"] if kind == "state" else bool((r["lead_mask"] >> r["player"]) & 1)
            by.setdefault(r["seed"], ([], []))[0 if isp else 1].append(r["rank"])
        d = [np.mean(p) - np.mean(q) for p, q in by.values() if p and q]
        m, h = ci95(d); summ["paired_rank_diff"] = {"mean": round(m, 4), "ci95": round(h, 4), "battles": len(d)}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
