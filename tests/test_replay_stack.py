"""dagger4.Replay frame links reproduce the env's frame stack (first frame tiled x4, then shifted), incl. ring wrap and
interleaved agents; overwritten references are rejected by sample()."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dagger4 as D
dev = torch.device("cpu"); rep = D.Replay(50, dev); last = {}; truth = {}
rng = np.random.default_rng(0); stacks = {}; envstack = {}; ok = 0; n = 0
for t in range(400):
    aid = int(rng.integers(0, 4)) + 100 * int(t // 60)      # 4 interleaved agents, new episodes every 60 steps
    val = t % 251
    f = np.full((115, 256, 3), val, dtype=np.uint8)
    es = envstack.get(aid)
    es = [val] * 4 if es is None else es[1:] + [val]
    envstack[aid] = es
    prev = last.get(aid, [None, None, None])
    z = np.zeros
    j = rep.add(f, prev, z((5, 29, 64), np.uint8), z((7, 29, 64), np.uint8), z(5, np.float32), z(388, np.float16),
                z(436, np.float16), z(3, np.int64), z(9, np.int16))
    last[aid] = [j, j, j] if prev[0] is None else prev[1:] + [j]
    truth[j] = list(es)
    # every valid sample's stack must equal the env stack recorded when it was written
    idx = rep.sample(32)
    st = rep.stack(idx)[:, 0, 0, ::3]
    for b, i in enumerate(list(idx)):
        n += 1; ok += int(st[b].tolist() == truth[i])
print({"checked": n, "equal": ok, "PASS": n == ok})
