"""Live check of the host / env round (one boot, random agents, always teams):
  1. team reward (Env._finish_step hold-back): teammates' final transitions share team_place == (10 - best member rank) / 9;
  2. area effects (nbhost_rl encode.c item feature 4: 1 = bomb / enemy area, -1 = my team's area) reach the observation;
  3. generated maps (--gen k instances on ~/nb68/rl/genmaps) play without restarts;
  4. teammate respawn (host, 15 s): knock-outs with a living mate come back at a mate's position, play on (can be
     knocked out again), dead steps carry terms["dead"] = 1 and a zero observation.
python -m live_check --n 2 --minutes 8 [--gen 1]"""
import argparse, collections, time, numpy as np
import env_v5 as E
from nbpc import fair as F

RS = collections.Counter(); RD = []   # respawn stats; distance respawned -> nearest living mate
class T(E.Env):
    bg = {}; bl = {}
    def _finish_step(self, i, r):
        s = self.I[i]; g = s.cfg["tag"]["groups"]; prev = set(getattr(s, "wprev", set()))
        for p in prev:
            if r.pl[p].reserved[3] == 0 and not r.pl[p].eliminated:
                m = [np.hypot(r.pl[q].x - r.pl[p].x, r.pl[q].y - r.pl[p].y) for q in range(10) if q != p and g[q] == g[p] and r.pl[q].alive and r.pl[q].reserved[3] == 0]
                RD.append(min(m) if m else -1); s.back = getattr(s, "back", set()) | {p}
        for p in getattr(s, "back", set()) - prev:
            if r.pl[p].reserved[3] > 0 or r.pl[p].eliminated: RS["again"] += 1; s.back.discard(p)   # out again after a respawn
        for p in range(10): RS["wait_steps"] += int(r.pl[p].reserved[3] > 0)
        super()._finish_step(i, r)
    def _start_battle(self, i, r):
        super()._start_battle(i, r); tag = self.I[i].cfg["tag"]; self.bg[self.I[i].battle] = tag["groups"]; self.bl[self.I[i].battle] = tag["loc"]

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2); ap.add_argument("--minutes", type=float, default=8)
ap.add_argument("--gen", type=int, default=0)
a = ap.parse_args()
gen = E.use_genmaps(a.gen) if a.gen else None
env = T(a.n, E.make_battle_fn(10, p_team=1.0, gen=gen, p_gen=1.0), ticks=2, fair=True, max_ticks=4000)
rng = np.random.default_rng(1); t0 = None; fins = []; steps = 0; area = collections.Counter()
try:
    while t0 is None or time.time() - t0 < a.minutes * 60:
        obs, ids, tr = env.poll(min_ready=1, max_wait=0.01)
        fins += [t for t in tr if t[2]]; RS["dead_tr"] += sum(1 for t in tr if t[3].get("dead"))
        if not len(ids): continue
        it = obs["vec"][:, F.IT0:F.IT0 + F.IT_K * F.IT_F].reshape(len(ids), F.IT_K, F.IT_F)
        dd = env.dead(ids); RS["dead_obs"] += int(dd.sum()); RS["dead_obs_nonzero"] += int(np.abs(obs["vec"][dd]).sum(1).astype(bool).sum())
        on = it[..., 0] > 0.5; area["enemy_or_bomb"] += int((on & (it[..., 4] > 0.5)).sum()); area["mine"] += int((on & (it[..., 4] < -0.5)).sum())
        area["cube"] += int((on & (it[..., 3] > 0.5)).sum()); area["other"] += int((on & (abs(it[..., 4]) < 0.5) & (it[..., 3] < 0.5)).sum())
        t0 = t0 or time.time(); B = len(ids); steps += B
        env.send(ids, np.stack([rng.integers(0, 17, B), rng.integers(0, 3, B), rng.integers(0, 360, B), rng.integers(0, 8, B)], 1))
finally:
    env.close()
grp = collections.defaultdict(list); nogroup_final = 0
for aid, rew, fin, t, rank in fins:
    b, p = aid >> 4, aid & 15; g = env.bg.get(b)
    if g is None or not g[p]: continue
    if "team_place" not in t: nogroup_final += 1; continue      # crashed battle: own placement only
    grp[(b, g[p])].append((t["team_place"], rank, t["place"]))
n_ok = n_full = bad = 0
for k, v in grp.items():
    tps = {round(x[0], 6) for x in v}; assert len(tps) == 1, ("teammates differ", k, v)
    b, gid = k; size = sum(1 for x in env.bg[b] if x == gid)
    if len(v) == size:
        n_full += 1; best = min(x[1] for x in v)
        if abs(v[0][0] - (10 - best) / 9) < 1e-6: n_ok += 1
        else: bad += 1; print("MISMATCH", k, v)
mine = [x[0] for v in grp.values() for x in v]
locs = collections.Counter(env.bl.values()); genb = sum(c for l, c in locs.items() if not l.startswith("Survival"))
print("steps %d battles %d episodes %d restarts %d | grouped finals %d in %d team-battles, full teams %d, rank match %d, mismatch %d, no-team_place finals %d"
      % (steps, env.stats["battles"], len(env.episodes), env.stats["restarts"], len(mine), len(grp), n_full, n_ok, bad, nogroup_final))
print("team_place mean %.3f min %.3f max %.3f; own place mean %.3f" % (np.mean(mine), min(mine), max(mine), np.mean([x[2] for v in grp.values() for x in v])))
print("item rows seen (agent-steps x slots):", dict(area), "| battles on generated maps %d of %d" % (genb, len(env.bl)), "| crashes", [c.get("loc") for c in env.crashes])
assert n_full and not bad
assert area["mine"] > 0, "no own-team area effect seen (host encode.c not rebuilt?)"
print("respawn:", env.resp, dict(RS), "| dist to mate at respawn: n %d median %.0f max %.0f" % (len(RD), np.median(RD) if RD else -1, max(RD) if RD else -1),
      "| final dead transitions", sum(1 for f in fins if f[3].get("dead")))
if a.gen: assert genb > 0
assert env.resp["out"] > 0 and env.resp["back"] > 0 and RD and max(RD) < 600 and RS["dead_obs_nonzero"] == 0
