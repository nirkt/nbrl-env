"""multi-player semantics smoke test: one instance, players CTL env-controlled (random v3 actions), the rest native"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
from nbpc.instance import Instance
from nbpc import proto as P
CTL = [0, 3, 5]
inst = Instance(0, tag=f"masmoke{os.getpid()}"); inst.start(); print("boot", round(inst.wait_ready(), 1), "s", flush=True)
ctl_mask = sum(1 << i for i in CTL); native = 0xffffffff & ~ctl_mask
rng = np.random.default_rng(0)
for ep in range(2):
    t0 = time.time(); r = inst.reset(seed=1000 + ep, native_mask=native, obs_mask=0x3ff, flags=P.F_STATE | P.F_OBS)
    print("reset", round(time.time() - t0, 3), "tick", r.tick, "local", r.local_index, "obs_written", bin(r.obs_written), "alive", r.alive)
    n = 0; t0 = time.time()
    while True:
        acts = {i: (int(rng.integers(-1, 16)), int(rng.choice(3, p=[0.85, 0.12, 0.03])), float(rng.uniform(0, 360)), 0.5) for i in CTL}
        r = inst.step(actions=acts, ctl_mask=ctl_mask, native_mask=native, obs_mask=0x3ff, flags=P.F_OBS)
        n += 1
        if n % 100 == 0 or r.done:
            pl = [(k, r.pl[k].alive, r.pl[k].hp, r.pl[k].eliminated, r.pl[k].elim_rank, r.pl[k].damage_dealt, r.pl[k].ammo) for k in range(10)]
            print(n, "tick", r.tick, "done", r.done, "alive", r.alive, "local_rank", r.local_rank, "winner", r.winner, "obsw", bin(r.obs_written))
            print("   ", pl)
        if r.done or n > 1200:
            break
    print("episode steps", n, "s/step", round((time.time() - t0) / n * 1000, 2), "ms")
    print("final elim ranks", [(k, r.pl[k].elim_rank, r.pl[k].elim_tick) for k in range(10)])
inst.close()
