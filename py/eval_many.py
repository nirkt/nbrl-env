#!/usr/bin/env python3
"""The 1v9 held-out protocol of eval_suite.py (seeds 1..N, player 0 = the policy, 9 native, sampled actions, max 4000
ticks) for SEVERAL policies in one booted env (the 12-instance boot costs ~4.6 min, a 500-episode state eval ~0.5 min).
Same rows / summary.json as eval_suite.py, one subdirectory per policy.
  python3 eval_many.py --out ../runs/eval --pol bc4_s0=state:../runs/bc4_native_s0/bc_best.pt --pol native=native [--pixels]
--ticks sets the env's ticks per step (the teacher was trained with 4)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.maenv import MAEnv
from eval_suite import summarize, load_policy, PixelPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", action="append", required=True, help="name=spec (native | native_lead | state:CKPT | pixel:CKPT)")
    ap.add_argument("--out", required=True); ap.add_argument("--episodes", type=int, default=500); ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--pixels", action="store_true", help="client mode (needed for pixel policies)")
    ap.add_argument("--fair", type=int, default=1); ap.add_argument("--lead-frac", type=float, default=1.0); ap.add_argument("--lead-dt", type=float, default=-2.0)
    ap.add_argument("--ticks", type=int, default=4)
    a = ap.parse_args()
    cur = {"queue": [], "kind": None}

    def bf(rng, i):
        if not cur["queue"]: return None
        sd = cur["queue"].pop(0); k = cur["kind"]
        if k in ("state", "pixel"): return {"seed": sd, "ctl": [0], "end": "player0", "tag": cur["name"]}
        return {"seed": sd, "ctl": [], "watch": [0], "lead_mask": 1 if k == "native_lead" else 0, "lead_frac": a.lead_frac, "end": "player0", "tag": cur["name"]}
    pixels = a.pixels or any(p.split("=", 1)[1].startswith("pixel:") for p in a.pol)
    env = MAEnv(a.n, bf, fair=bool(a.fair), verbose=True, pixels=pixels, tag=f"em{os.getpid()}", lead_dt=a.lead_dt, ticks=a.ticks)
    for item in a.pol:
        name, spec = item.split("=", 1); kind = spec.split(":")[0]
        pol = load_policy(spec)[0] if kind == "state" else (PixelPolicy(spec.split(":", 1)[1]) if kind == "pixel" else None)
        out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
        json.dump({"policy": spec, "episodes": a.episodes, "seed0": a.seed0, "pixels": pixels, "fair": a.fair}, open(os.path.join(out, "config.json"), "w"), indent=1)
        cur.update(queue=list(range(a.seed0, a.seed0 + a.episodes)), kind=kind, name=name)
        for i, s in enumerate(env.I):          # wake idle instances
            if s.state == "idle": env._reset(i)
        t1 = time.time(); n0 = len(env.episodes); steps0 = env.stats["env_steps"]
        while True:
            obs, ids, tr = env.poll(min_ready=max(1, a.n // 3), max_wait=0.003)
            if len(ids):
                if kind == "pixel": act = pol.act(obs["pixels"])
                elif kind == "state": act = pol.act(obs["vec"], obs["grid"])[0]
                else: act = np.zeros((len(ids), 4), np.int64)
                env.send(ids, act)
            rows = [r for r in env.episodes[n0:] if r["tag"] == name and r["player"] == 0]
            if len({r["seed"] for r in rows}) >= a.episodes or all(s.state == "idle" for s in env.I):
                break
        while not all(s.state == "idle" for s in env.I):   # drain (spare battles never start: the queue is empty)
            obs, ids, tr = env.poll(min_ready=1, max_wait=0.003)
            if len(ids): env.send(ids, np.zeros((len(ids), 4), np.int64))
        rows = {r["seed"]: r for r in env.episodes[n0:] if r["tag"] == name and r["player"] == 0}; rows = list(rows.values())
        for r in rows: r["ctl_player"] = kind in ("state", "pixel")
        with open(os.path.join(out, "episodes.jsonl"), "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        dt = time.time() - t1
        sm = {"policy": spec, "protocol": "1v9", "minutes": round(dt / 60, 2), "env_steps": env.stats["env_steps"] - steps0,
              "env_steps_per_s": round((env.stats["env_steps"] - steps0) / dt, 1), "instances": a.n, "restarts": env.stats["restarts"],
              "client_mode": pixels, "fair": a.fair, **summarize(rows)}
        json.dump(sm, open(os.path.join(out, "summary.json"), "w"), indent=1)
        print(f"[eval_many] {name}: rank {sm['rank']['mean']:.3f} +- {sm['rank']['ci95']:.3f}, win {sm['win']['mean']:.3f}, "
              f"dealt {sm['dealt']['mean']:.0f}, hit {sm['hit_rate_pooled']}, {sm['minutes']} min", flush=True)
    env.close()


if __name__ == "__main__":
    main()
