"""Next-round settings check: client mode, 20 decisions/s (ticks=1), 512x230 view (GPU box-downscale of the full frame).
2 instances, native AI drives player 0 (observed via native_ctl), 300 steps each: frame shape, a saved sample, throughput."""
import os, sys, time, json
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbpc.maenv import MAEnv
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "runs", "p0_checks", "obs512_20hz"); os.makedirs(out, exist_ok=True)
q = [61_000_000, 61_000_001]
bf = lambda rng, i: ({"seed": q.pop(0), "ctl": [0], "end": "player0", "tag": "t", "native_ctl": True} if q else None)
env = MAEnv(2, bf, pixels=True, ticks=1, obs_size=(512, 230), tag=f"o512{os.getpid()}", verbose=True)
n = 0; t0 = None; saved = 0; shapes = set()
while n < 600:
    obs, ids, tr = env.poll(min_ready=1, max_wait=0.005)
    if len(ids):
        if t0 is None: t0 = time.time()
        shapes.add(tuple(obs["pixels"].shape[1:]))
        n += len(ids)
        if n in (200, 201) or (n > 400 and saved < 2):
            f = obs["pixels"][0]
            Image.fromarray(np.concatenate([f[:, :, 9:12], f[:, :, 0:3]], 0)).save(os.path.join(out, f"frame_{n}.png")); saved += 1
        env.send(ids, np.zeros((len(ids), 4), np.int64))
    if all(s.state == "idle" for s in env.I): break
dt = time.time() - t0; env.close()
res = {"steps": n, "seconds": round(dt, 1), "steps_per_s_2_instances": round(n / dt, 1), "game_seconds_per_s": round(n / dt / 20, 2), "frame_shapes": sorted(shapes),
       "no_pixels": env.stats.get("no_pixels", 0)}
print(json.dumps(res)); json.dump(res, open(os.path.join(out, "result.json"), "w"))
