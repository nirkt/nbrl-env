#!/usr/bin/env python3
"""End-to-end smoke test: boots 2 instances in client mode with obs_mode="both" (pixels + state; each instance has its own
headless nbgl renderer), resets, steps 300 times with random v3 actions (several episodes end and auto-reset), forces a
reset, and checks: observation shapes/dtypes, pixel frames present and not blank, HUD mask applied, state vectors finite,
rewards finite, episodes completed, no restarts, both processes alive. Exit code 0 = PASS.

  python3 work/tests/test_e2e_smoke.py        (~2 min)"""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "py"))
from nbpc.vecenv import NBVecEnvPC

fails = []
def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond: fails.append(what)

t0 = time.time()
v = NBVecEnvPC(2, obs_mode="both", tag="e2e", verbose=True, max_episode_ticks=600)
try:
    sp = v.single_observation_space
    obs, info = v.reset(seeds=[11, 12])
    check(obs["pixels"].shape == (2, 115, 256, 12) and obs["pixels"].dtype == np.uint8, f"reset pixels shape {obs['pixels'].shape}")
    check(obs["vec"].shape == (2, 436) and obs["grid"].shape == (2, 6, 25, 25), "reset state shapes")
    check(all(sp[k].contains(obs[k][0]) for k in ("pixels",)), "pixels in observation space")
    check(float(obs["pixels"][:, :, :, -3:].mean()) > 10, f"reset frames not blank (mean {obs['pixels'][:, :, :, -3:].mean():.1f})")
    # HUD mask box (1015, 25, 1160, 95) on 2400x1080 -> rows 2..11, cols 108..124 of the 256x115 frame are zero
    check(int(obs["pixels"][:, 2:11, 108:124].max()) == 0, "HUD mask applied")
    rng = np.random.default_rng(0); eps = 0; steps = 0; rets = []; diffs = []; per_slot = np.zeros(2, int)
    last = obs["pixels"][:, :, :, -3:].astype(np.int16)
    for k in range(300):
        acts = np.stack([rng.integers(0, 17, 2), rng.choice(3, 2, p=[0.85, 0.12, 0.03]), rng.integers(0, 360, 2), rng.integers(0, 8, 2)], 1)
        obs, r, te, tr, inf = v.step(acts)
        steps += int(inf["valid"].sum() - inf["new_episode"].sum()); eps += int((te | tr).sum())
        per_slot += (inf["valid"] & ~inf["new_episode"]).astype(int)
        check_ok = np.isfinite(r).all() and np.isfinite(obs["vec"]).all()
        if not check_ok: check(False, f"finite rewards/obs at step {k}"); break
        cur = obs["pixels"][:, :, :, -3:].astype(np.int16); diffs.append(float(np.abs(cur - last).mean())); last = cur
        for e in inf["episode"]:
            if e is not None: rets.append(e["r"])
    # while one slot resets (~2 s in pixel mode) the other keeps stepping alone, so fewer than 600 transitions is expected
    check(steps >= 250 and per_slot.min() >= 100, f"env steps {steps}, per slot {per_slot.tolist()} (300 calls; resets are asynchronous)")
    check(eps >= 2, f"episodes completed {eps} (max_episode_ticks 600)")
    check(np.median(diffs) > 0.3, f"frames change between steps (median |diff| {np.median(diffs):.2f})")
    check(v.stats["no_pixels"] == 0, f"no step without pixels ({v.stats['no_pixels']})")
    check(all(np.isfinite(x) for x in rets) and len(rets) == eps, f"episode returns {['%.2f' % x for x in rets[:6]]}")
    obs, info = v.reset(seeds=[13, 14])
    check(bool(info["valid"].all()), "forced reset of both slots")
    check(v.stats["restarts"] == 0, f"restarts {v.stats['restarts']}")
    check(all(s.inst.alive() for s in v.slots), "both processes alive")
finally:
    v.close()
print(json.dumps({"result": "PASS" if not fails else "FAIL", "fails": fails, "seconds": round(time.time() - t0, 1),
                  "resets_s": v.stats["resets_s"][:8], "boot_s": v.stats["boot_s"]}), flush=True)
sys.exit(1 if fails else 0)
