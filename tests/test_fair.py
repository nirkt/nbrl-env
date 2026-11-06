"""fair.py checks on real observations (one instance, all-native battles, obs of all 10 players every step):
 (1) observer = local player 0: fair.enemy_visible == the encoder's own feature 11 (nbhost's emulation centred on the local
     brawler) -> must agree on every present enemy slot;
 (2) grid channel 5 rebuilt from ALL present enemies == the encoder's channel 5 (cell arithmetic exact);
 (3) for observers 1..9: how often the encoder's feature 11 (the local player's view) disagrees with their own view (F1);
 (4) fair_inplace writes into the given arrays (views), idempotent."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
from nbpc.instance import Instance
from nbpc import proto as P
from nbpc import fair
inst = Instance(0, tag=f"tfair{os.getpid()}"); inst.start(); inst.wait_ready()
NV, NG = P.OBS_VEC, P.OBS_GRID
res = {"local_slots": 0, "local_agree": 0, "grid_cells_equal": 0, "grid_obs": 0, "other_slots": 0, "other_disagree": 0, "fails": []}
for ep in range(3):
    r = inst.reset(seed=4242 + ep, native_mask=0xffffffff, obs_mask=0x3ff, flags=P.F_OBS)
    W, H = r.map_w, r.map_h
    for k in range(400):
        r = inst.step(actions=None, ctl_mask=0, native_mask=0xffffffff, obs_mask=0x3ff, flags=P.F_OBS)
        for p in range(10):
            if not (r.obs_written >> p) & 1: continue
            o = inst.obs_f32[p * (NV + NG):(p + 1) * (NV + NG)]
            v = o[:NV].copy()[None]; g = o[NV:].reshape(1, 6, 25, 25).copy()
            en = v[0, 22:166].reshape(9, 16); pres = en[:, 0] > 0.5
            vis = fair.enemy_visible(v, W, H)[0]
            if p == r.local_index:
                res["local_slots"] += int(pres.sum()); res["local_agree"] += int(((en[:, 11] > 0.5) == vis)[pres].sum())
            else:
                res["other_slots"] += int(pres.sum()); res["other_disagree"] += int(((en[:, 11] > 0.5) != vis)[pres].sum())
            # (2) rebuild channel 5 from all present enemies with fair's arithmetic
            mx, my = fair.own_xy(v, W, H)
            ex = mx[0] + np.rint(en[:, 1].astype(np.float64) * 3000); ey = my[0] + np.rint(en[:, 2].astype(np.float64) * 3000)
            gx = (ex // 300 - mx[0] // 300 + 12).astype(int); gy = (ey // 300 - my[0] // 300 + 12).astype(int)
            g5 = np.zeros((25, 25))
            for kk in np.nonzero(pres)[0]:
                if 0 <= gx[kk] < 25 and 0 <= gy[kk] < 25: g5[gy[kk], gx[kk]] += 1
            res["grid_obs"] += 1; res["grid_cells_equal"] += int(np.array_equal(g5, g[0, 5]))
            # (4) in place + idempotent
            v2, g2 = v.copy(), g.copy(); fair.fair_inplace(v2, g2, W, H); v3, g3 = v2.copy(), g2.copy(); fair.fair_inplace(v3, g3, W, H)
            if not (np.array_equal(v2, v3) and np.array_equal(g2, g3)): res["fails"].append(("idempotent", ep, k, p))
            if np.array_equal(v2, v) and (vis != pres).any(): res["fails"].append(("not_in_place", ep, k, p))
        if r.done: break
inst.close()
res["local_agree_frac"] = res["local_agree"] / max(1, res["local_slots"])
res["grid_equal_frac"] = res["grid_cells_equal"] / max(1, res["grid_obs"])
res["other_disagree_frac"] = res["other_disagree"] / max(1, res["other_slots"])
res["PASS"] = res["local_agree_frac"] == 1.0 and res["grid_equal_frac"] == 1.0 and not res["fails"]
res["fails"] = res["fails"][:10]
print(json.dumps(res, indent=1))
