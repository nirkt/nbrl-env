"""T2 determinism: same seed + same scripted actions -> bit-identical state (snap2 bytes) for 2000 ticks.
Runs each (seed, action sequence) twice in one process and once in a second process; compares the state2 bytes
after every 4-tick step (500 steps = 2000 ticks). Reports the first divergence (step, tick, field) if any.
Usage: t2_determinism.py [--seeds 3] [--extra-a ARGS] [--extra-b ARGS]   (extra nbhost args per process, e.g.
'--no-snapshot' to check that an optimisation does not change the simulation)"""
import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.instance import Instance
from nbpc import proto as P
import state2 as S2

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--extra-a", default=""); ap.add_argument("--extra-b", default=""); ap.add_argument("--out", default=None)
ap.add_argument("--native-local", action="store_true", help="the native AI plays the local player too (no scripted actions)")
ap.add_argument("--gas", type=int, default=1); ap.add_argument("--hold-bots", action="store_true"); ap.add_argument("--seed0", type=int, default=77000)
a = ap.parse_args()

def actions(seed, n):
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(-1, 16)), int(rng.choice(3, p=[0.8, 0.15, 0.05])), float(rng.integers(0, 360)), float(rng.integers(1, 9)) / 8)
            for _ in range(n)]

def run(inst, seed, acts):
    native = 0xffffffff if a.native_local else 0xfffffffe
    hold = 0
    if a.hold_bots: hold, native = 0x3fe, native & ~0x3fe
    r = inst.reset(seed=seed, flags=P.F_STATE, native_mask=native, hold_mask=hold, gas=a.gas); out = [inst.state_bytes()]
    for k, act in enumerate(acts):
        r = inst.step(actions=None if a.native_local else {0: act}, ticks=4, flags=P.F_STATE, native_mask=native, hold_mask=hold,
                      ctl_mask=0 if a.native_local else 1)
        if r.status: out.append(None); break
        out.append(inst.state_bytes())
        if r.done: break
    return out, r.seed_applied, r.seed_plus1

def first_diff(x, y):
    if x == y: return None
    if x is None or y is None: return {"missing": True}
    ax, ay = np.frombuffer(x, "<i4"), np.frombuffer(y, "<i4")
    if len(ax) != len(ay):
        px, py = S2.parse(x), S2.parse(y)
        return {"len": [len(ax), len(ay)], "counts": [[len(px[k]) for k in ("players", "chars", "proj", "items")], [len(py[k]) for k in ("players", "chars", "proj", "items")]]}
    i = int(np.nonzero(ax != ay)[0][0]); p = S2.parse(x)
    hdr = 8; npl, nch, npj = len(p["players"]), len(p["chars"]), len(p["proj"])
    if i < hdr: where = ("header", i)
    elif i < hdr + npl * 8: where = ("player", (i - hdr) // 8, S2.P_F[(i - hdr) % 8])
    elif i < hdr + npl * 8 + nch * S2.C2: j = i - hdr - npl * 8; where = ("char", j // S2.C2, S2.C_F[j % S2.C2])
    elif i < hdr + npl * 8 + nch * S2.C2 + npj * S2.J2: j = i - hdr - npl * 8 - nch * S2.C2; where = ("proj", j // S2.J2, S2.J_F[j % S2.J2])
    else: j = i - hdr - npl * 8 - nch * S2.C2 - npj * S2.J2; where = ("item", j // S2.I2, S2.I_F[j % S2.I2])
    return {"int_index": i, "where": where, "values": [int(ax[i]), int(ay[i])], "tick": int(ax[2])}

A = Instance(0, tag="detA", extra_args=a.extra_a.split()); B = Instance(1, tag="detB", extra_args=a.extra_b.split())
A.start(); B.start(); A.wait_ready(); B.wait_ready()
rows = []
try:
    for s in range(a.seeds):
        seed = a.seed0 + s; acts = actions(seed, a.steps)
        a1, sa, p1 = run(A, seed, acts); a2, _, _ = run(A, seed, acts); b1, sb, p2 = run(B, seed, acts)
        res = {"seed": seed, "seed_applied": [sa, sb], "seed_plus1": [p1, p2], "steps": [len(a1) - 1, len(a2) - 1, len(b1) - 1],
               "ticks_compared": None, "same_process": None, "cross_process": None}
        for name, other in (("same_process", a2), ("cross_process", b1)):
            n = min(len(a1), len(other)); d = None
            for k in range(n):
                d = first_diff(a1[k], other[k])
                if d: d["step"] = k; break
            res[name] = d if d else "identical"
        n = min(len(a1), len(a2), len(b1)) - 1
        res["ticks_compared"] = int(np.frombuffer(a1[n], "<i4")[2]) - int(np.frombuffer(a1[0], "<i4")[2])
        rows.append(res); print(json.dumps(res), flush=True)
finally:
    A.close(); B.close()
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
