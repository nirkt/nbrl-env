"""C observation encoder (nbhost encode.c) vs the Python state2.encode on the same snap2 dumps: all 10 players,
random actions for player 0, native bots, several episodes. Prints the max abs difference per block."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.instance import Instance
from nbpc import proto as P
import state2 as S2

def extras(r):
    return {"localIndex": r.local_index, "cubes": {str(r.cubes[i][0]): int(r.cubes[i][1]) for i in range(r.ncubes)},
            "jumps": [list(r.jumps[i]) for i in range(r.njumps)]}

inst = Instance(0, tag="testenc"); inst.start(); inst.wait_ready()
worst = {"vec": 0.0, "grid": 0.0}; n = 0; nfail = 0; nnone = 0
try:
    for ep in range(int(os.environ.get("EPS", "4"))):
        r = inst.reset(seed=4000 + ep, flags=P.F_STATE | P.F_OBS)
        M = S2.MapInfo(r.map_w, r.map_h, inst.map_tiles().tolist())
        prev = None; rng = np.random.default_rng(ep); steps = 0
        while True:
            v = S2.StateView(inst.state_bytes(), extras(r), M, prev); prev = v
            for p in range(10):
                enc = S2.encode(v, p)
                off = p * (P.OBS_VEC + P.OBS_GRID)
                cv, cg = inst.obs_f32[off:off + P.OBS_VEC], inst.obs_f32[off + P.OBS_VEC: off + P.OBS_VEC + P.OBS_GRID].reshape(6, 25, 25)
                if enc is None:
                    nnone += 1
                    if (r.obs_written >> p) & 1: nfail += 1; print("C wrote obs where python has none", ep, steps, p)
                    continue
                pv, pg = S2.flatten(enc)
                if not (r.obs_written >> p) & 1: nfail += 1; print("C missing obs", ep, steps, p); continue
                dv, dg = float(np.abs(pv - cv).max()), float(np.abs(pg - cg).max())
                if dv > 1e-5 or dg > 0:
                    nfail += 1
                    if nfail < 40:
                        i = int(np.abs(pv - cv).argmax()); print(f"ep {ep} step {steps} p {p}: vec diff {dv} at {i} (py {pv[i]} c {cv[i]}), grid diff {dg}")
                worst["vec"] = max(worst["vec"], dv); worst["grid"] = max(worst["grid"], dg); n += 1
            if r.done or steps >= 400: break
            a = (int(rng.integers(-1, 16)), int(rng.choice([0, 0, 1, 2])), float(rng.uniform(0, 360)), float(rng.uniform(0.1, 1)))
            r = inst.step(actions={0: a}, ticks=4, flags=P.F_STATE | P.F_OBS, obs_mask=0x3ff)
            if r.status: print("status", r.status, r.err); break
            steps += 1
        print(f"episode {ep}: {steps} steps, compared so far {n}, mismatches {nfail}", flush=True)
finally:
    inst.close()
print(json.dumps({"compared": n, "none_both": nnone, "mismatches": nfail, "max_abs_diff": worst}))
