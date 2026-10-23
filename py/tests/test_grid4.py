"""grid4.GridBuilder == the C encoder's grid (after the fair transform) for every observation of all 10 players in 2
all-native battles: rebuilt from (sim x, y, tick) + fair.enemy_cells only."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_rocm_fix  # noqa
from nbpc.instance import Instance
from nbpc import proto as P
from nbpc import fair
from grid4 import GridBuilder
NV, NG = P.OBS_VEC, P.OBS_GRID
inst = Instance(0, tag=f"tgrid{os.getpid()}"); inst.start(); inst.wait_ready()
dev = torch.device("cuda"); gb = None; n = 0; bad = 0; badch = np.zeros(6, int); ex = []
for ep in range(2):
    r = inst.reset(seed=5151 + ep, native_mask=0xffffffff, obs_mask=0x3ff, flags=P.F_OBS)
    if gb is None: gb = GridBuilder(inst.map_tiles(), r.map_w, r.map_h, dev)
    for k in range(700):
        alive = [p for p in range(10) if (r.obs_written >> p) & 1]
        if alive:
            O = np.stack([inst.obs_f32[p * (NV + NG):p * (NV + NG) + NV] for p in alive]).astype(np.float32)
            G = np.stack([inst.obs_f32[p * (NV + NG) + NV:(p + 1) * (NV + NG)].reshape(6, 25, 25) for p in alive]).astype(np.float32)
            fair.fair_inplace(O, G, r.map_w, r.map_h); E = fair.enemy_cells(O, r.map_w, r.map_h)
            x = torch.tensor([r.pl[p].x for p in alive], device=dev); y = torch.tensor([r.pl[p].y for p in alive], device=dev)
            t = torch.full((len(alive),), r.tick, device=dev)
            g2 = gb.build(x, y, t, torch.as_tensor(E, device=dev)).cpu().numpy()
            for j in range(len(alive)):
                n += 1; d = (g2[j].astype(np.float32) != G[j]).reshape(6, -1).any(1)
                if d.any():
                    bad += 1; badch += d
                    if len(ex) < 5: ex.append({"ep": ep, "step": k, "player": alive[j], "tick": r.tick, "channels": np.nonzero(d)[0].tolist()})
        r = inst.step(actions=None, ctl_mask=0, native_mask=0xffffffff, obs_mask=0x3ff, flags=P.F_OBS)
        if r.done: break
inst.close()
res = {"observations": n, "mismatching": bad, "per_channel": badch.tolist(), "examples": ex, "PASS": bad == 0}
print(json.dumps(res, indent=1))
