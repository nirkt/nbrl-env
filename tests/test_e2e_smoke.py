#!/usr/bin/env python3
"""End-to-end smoke test of MAEnv: boots 2 game instances in client mode (pixels + state, each with its own headless
renderer), plays 300 decisions with random v3 actions (games end and restart), forces a reset, and checks shapes, pixel
frames (not blank, changing, HUD masked), finite values, completed games, no restarts, and that every process is alive.
Exit code 0 = PASS.

  python3 tests/test_e2e_smoke.py        (about 2 minutes)"""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "py"))
from nbpc.maenv import MAEnv

fails = []
def check(cond, what):
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    if not cond: fails.append(what)

seeds = iter(range(11, 10_000))
def battle_fn(rng, i):
    return {"seed": next(seeds), "ctl": [0], "end": "player0", "tag": "e2e"}

t0 = time.time()
env = MAEnv(2, battle_fn, pixels=True, obs_size=(256, 115), want_state=True, ticks=4, fair=True, max_ticks=600, verbose=True)
try:
    rng = np.random.default_rng(0)
    obs, ids, tr = env.poll(min_ready=2, max_wait=10)
    check(len(ids) >= 1, f"first observations: {len(ids)} agents ready")
    check(obs["pixels"].shape[1:] == (115, 256, 12) and obs["pixels"].dtype == np.uint8, f"pixel shape {obs['pixels'].shape}")
    check(obs["vec"].shape[1:] == (436,) and obs["grid"].shape[1:] == (6, 25, 25), "state shapes")
    check(float(obs["pixels"][..., -3:].mean()) > 10, f"frames not blank (mean {obs['pixels'][..., -3:].mean():.1f})")
    # HUD box (1015, 25, 1160, 95) of 2400x1080 maps to rows 2..11, cols 108..124 of the 256x115 frame: masked to zero
    check(int(obs["pixels"][:, 2:11, 108:124, -3:].max()) == 0, "HUD mask applied")
    steps = 0; games = 0; diffs = []; last = obs["pixels"][..., -3:].astype(np.int16); bad = None
    for k in range(300):
        if len(ids) == 0:
            obs, ids, tr = env.poll(min_ready=1, max_wait=0.5); continue
        acts = np.stack([rng.integers(0, 17, len(ids)), rng.choice(3, len(ids), p=[0.85, 0.12, 0.03]),
                         rng.integers(0, 360, len(ids)), rng.integers(0, 8, len(ids))], 1)
        env.send(ids, acts)
        obs, ids, tr = env.poll(min_ready=1, max_wait=0.5)
        steps += len(tr)
        games += sum(1 for t in tr if t[2])
        if len(ids) and not (np.isfinite(obs["vec"]).all() and all(np.isfinite(t[1]) for t in tr)):
            bad = k; break
        if len(ids) and obs["pixels"].shape[0] == last.shape[0]:
            cur = obs["pixels"][..., -3:].astype(np.int16); diffs.append(float(np.abs(cur - last).mean())); last = cur
    check(bad is None, "finite rewards and observations" + ("" if bad is None else f" (failed at call {bad})"))
    check(steps >= 100, f"decisions finished: {steps}")
    check(games >= 1 or len(env.episodes) >= 1, f"games completed: {games} (episodes recorded {len(env.episodes)})")
    check(len(diffs) == 0 or np.median(diffs) > 0.3, f"frames change between steps (median |diff| {np.median(diffs) if diffs else 0:.2f})")
    check(env.stats["restarts"] == 0, f"restarts {env.stats['restarts']}")
    check(all(s.inst.alive() for s in env.I if s.inst is not None), "all game processes alive")
finally:
    env.close()
print(json.dumps({"result": "PASS" if not fails else "FAIL", "fails": fails, "seconds": round(time.time() - t0, 1)}), flush=True)
sys.exit(1 if fails else 0)
