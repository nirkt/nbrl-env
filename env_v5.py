"""Randomised Showdown battles on top of nirkt's MAEnv (run in WSL after `source rl/env.sh`).

    python3 -m env_v5 --n 12 --minutes 5 --agents 10      # benchmark: random actions, one summary line + bench.jsonl

battle_fn(): every battle a random Showdown map, a random brawler per player, solo / duo (5 x 2) / trio (3 x 3 + 1
solo: the sim has 10 players, the real mode 12) teams, and `agents` policy-controlled slots (the rest = native AI).
The battle is still Solo Showdown underneath (teams only change who can hurt whom), so MAEnv's rank / place reward is
per player; `Env` holds a grouped agent's final transition until its team's rank is known and swaps in the team's
place / win (`team_ranks`). Solo players: unchanged.
"""
import argparse, json, os, re, sys, time
import numpy as np
from nbpc import diversity as D, proto as P
from nbpc.maenv import MAEnv
import obs_v5 as O

SKIP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skip.json")   # {"brawlers": [...], "locations": [...]}


def _skip():
    return json.load(open(SKIP_FILE)) if os.path.exists(SKIP_FILE) else {}


def heroes():
    """playable brawlers = Hero rows that own cards (drops transforms, bosses and the hashed unreleased rows)"""
    owners = {r["Target"] for r in D._rows("cards.csv")}
    bad = set(_skip().get("brawlers", []))
    return [n for n, (_, r) in D.table("characters.csv").items()
            if r.get("Type") == "Hero" and r.get("Disabled") != "true" and n in owners
            and not re.fullmatch(r"[0-9a-f]{40}", n) and n not in bad]


def locations():
    bad = set(_skip().get("locations", []))
    return [n for n, (_, r) in D.table("locations.csv").items()
            if r.get("GameModeVariation") == "Showdown" and r.get("Disabled") != "true" and n not in bad]


def make_battle_fn(agents=10, seed0=10_000_000, roster=None, locs=None, p_team=0.5):
    roster = roster or heroes(); locs = locs or locations()
    seeds = iter(range(seed0, seed0 + 10 ** 9))

    def battle_fn(rng, i):
        br = [roster[int(rng.integers(len(roster)))] for _ in range(10)]
        loc = locs[int(rng.integers(len(locs)))]
        div = D.make_div(location=loc, chr=br)
        size = int(rng.integers(2, 4)) if rng.random() < p_team else 1
        groups = [0] * 10                                    # group id per player, 0 = alone (host: nbhost_rl)
        if size > 1:
            for k, p in enumerate(rng.permutation(10)[:10 // size * size]): groups[int(p)] = k // size + 1
            div["team_groups"] = groups
        ctl = sorted(int(p) for p in rng.choice(10, size=agents, replace=False))
        return {"seed": next(seeds), "ctl": ctl, "end": "agents", "div": div,
                "tag": {"loc": loc, "chr": br, "team_size": size, "groups": groups}}
    return battle_fn


SL = O.SLIDERS
_COVER = [O.SEG.index(c) for c in ("indestructible", "destructible_wall", "fragile_wall", "bouncer", "gravity", "destructible")]
_PL = ("x", "y", "hp", "max_hp", "damage_dealt", "bomb_damage", "damage_taken", "box_damage", "cubes", "cubes_total")


def slider_state(tiles, W, H, brawlers, rng, n=10):
    """per-battle state of the slider rewards: weights 0-2 per player and slider, map masks, trackers"""
    t = O._tile_lut()[np.clip(np.asarray(tiles).reshape(H, W) % 1000000 + 1, 0, 255)] if tiles is not None else np.zeros((H, W), np.uint8)
    cover = np.pad(np.isin(t, _COVER), 2)
    near = np.zeros((H, W), bool)
    for dy in range(5):
        for dx in range(5): near |= cover[dy:dy + H, dx:dx + W]
    rg = [float(O.brawler_stats(b)[0]) * 3000 if b else 2400.0 for b in (list(brawlers) + [None] * n)[:n]]
    return {"w": rng.uniform(0, 2, (n, len(SL))), "bush": t == O.SEG.index("bush"), "nearcover": near, "range": np.array(rg),
            "hid": np.zeros(n, bool), "low": np.zeros(n, bool), "dist": np.full(n, np.nan), "hp": np.full(n, np.nan),
            "checked": np.zeros((n, H, W), bool), "sum": np.zeros((n, len(SL))), "ret": np.zeros(n)}


def slider_terms(sl, pl, n=10):
    """(n, 10) slider reward terms of one env step, order SLIDERS; pl = the response's players. Units as MAEnv's terms
    (1000 damage = 1). The host reports damage per player, not per target, so `focus`, `protect` and `third_party` use
    "that enemy took damage this step" instead of "I hit it".
    shortcut: bushes / cover are the tiles at the battle start (destroyed ones still count); re-read map_tiles to upgrade."""
    A = np.array([[getattr(pl[p], k) for k in _PL] + [pl[p].reserved[2], pl[p].alive and not pl[p].eliminated] for p in range(n)], np.float64)
    x, y, hp, mhp, dealt, bomb, taken, box, cubes, ctot, team, alive = A.T
    alive = alive > 0; H, W = sl["bush"].shape; dk = (dealt + bomb) / 1000.0; idx = np.arange(n)
    d = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
    both = alive[:, None] & alive[None, :]
    en = both & (team[:, None] != team[None, :]); mate = both & (team[:, None] == team[None, :]) & (idx[:, None] != idx[None, :])
    tx, ty = np.clip(x // 300, 0, W - 1).astype(int), np.clip(y // 300, 0, H - 1).astype(int)
    inb = sl["bush"][ty, tx]
    # sees[q, p]: p is on q's screen (the camera rule of nbpc.fair) and not hidden in a bush
    camx = np.clip(x, 3000.0, W * 300 - 3000.0); ry = y[None, :] - y[:, None]
    sees = (np.abs(x[None, :] - camx[:, None]) < 4500) & (ry > -3500) & (ry < 2900) & ~(inb[None, :] & (d > 900))
    hid = inb & alive & ~(en.T & sees).any(0)
    d_en = np.where(en, d, np.inf); near = d_en.min(1); has = np.isfinite(near); j = d_en.argmin(1)
    inr = en & (d <= sl["range"][:, None])
    low_hp = np.where(inr, (hp / np.maximum(mhp, 1))[None, :], np.inf).argmin(1)
    mate_taken = (mate * (d < 1800) * taken[None, :]).sum(1) / 1000.0     # only mates near enough to help
    gain = np.nan_to_num(np.maximum(hp - sl["hp"], 0)) / 1000.0
    low = sl["low"] | (hp < 0.4 * mhp)
    closing = np.nan_to_num(np.clip(sl["dist"] - np.where(has, near, np.nan), 0, 300)) / 15000.0   # 0.02 per tile
    others = taken[None, :] > (dealt + bomb)[:, None]    # others[p, q]: q took more than p dealt = someone else hit it too
    new = np.zeros(n)
    for p in np.nonzero(alive)[0]:                       # bush tiles within 3 tiles, first time this battle
        ys, xs = slice(max(ty[p] - 3, 0), ty[p] + 4), slice(max(tx[p] - 3, 0), tx[p] + 4)
        fresh = sl["bush"][ys, xs] & ~sl["checked"][p, ys, xs]
        new[p] = fresh.sum(); sl["checked"][p, ys, xs] |= fresh
    T = np.stack([
        0.01 * hid + dk * sl["hid"],                                              # bush_camp: hidden time + ambush damage
        box / 1000.0 + cubes,                                                     # collect
        dk * (mate_taken > 0) - mate_taken + 0.01 * (np.where(mate, d, np.inf).min(1) < 1800),   # protect
        gain * low,                                                               # retreat: HP regained after dropping low
        dk * (taken == 0) * (has & (near >= 0.7 * sl["range"])),                  # kite
        dk * sl["nearcover"][ty, tx],                                             # peek
        dk * inr.any(1) * (taken[low_hp] > 0),                                    # focus
        closing * (has & (hp > hp[j]) & (ctot >= ctot[j])),                       # push
        dk * (taken == 0) * (inr & others).any(1),                                # third_party
        0.005 * new], 1) * alive[:, None]                                         # bush_control
    sl["hid"], sl["dist"], sl["hp"], sl["low"] = hid, np.where(has, near, np.nan), np.where(alive, hp, np.nan), low & (hp < 0.8 * mhp)
    return T


def player_ranks(r):
    """(10,) rank each player has / would get if its episode ended now (MAEnv._finish_step's rule), (10,) eliminated"""
    done, alive = bool(r.done), int(r.alive or 0); pr = np.zeros(10); dead = np.zeros(10, bool)
    for p in range(10):
        pl = r.pl[p]
        if pl.eliminated or not pl.alive: pr[p] = pl.elim_rank if pl.elim_rank > 0 else 10; dead[p] = True
        elif done: pr[p] = 1 if (r.winner == p or alive <= 1) else max(1, alive)
        else: pr[p] = (1 + alive) / 2.0
    return pr, dead


def team_ranks(groups, pr, dead, over):
    """(10,) placement per player: group 0 = own rank; a group = the best rank among its members (alive members outrank
    eliminated ones), nan while a member lives and the battle goes on.
    shortcut: a battle cut short (end "agents" once every agent is out, max_ticks) ranks a living native mate by MAEnv's
    truncation estimate (1 + alive) / 2; exact only with end "done"."""
    g, out = np.asarray(groups), np.asarray(pr, np.float64).copy()
    for k in set(g[g > 0].tolist()):
        m = g == k; out[m] = out[m].min() if over or dead[m].all() else np.nan
    return out


class Env(MAEnv):
    """MAEnv + a log of every instance restart with the battle that caused it + the slider rewards: every player gets
    random weights 0-2 per battle; the transitions' reward and terms ("s_<slider>") and the episode rows include them."""
    crashes = []
    slider_scale = 1.0
    record = False    # True: battles with cfg "native_ctl" put the native AI's action into the transition terms ("_act": BC labels)

    def _flags(self):
        return super()._flags() | (P.F_REC if self.record else 0)

    def _labels(self, s, r):
        """player -> [move, fire, aim, dist] the native AI did this step: the rules of nirkt's collect_demos4 (move = the
        displacement's 16-direction bucket, fire = first unblocked controller record, super wins)"""
        rec = np.frombuffer(s.inst.mm, np.int32, r.rec_n * 8, P.OFF_REC).reshape(-1, 8) if r.rec_n else np.zeros((0, 8), np.int32)
        jump = {int(r.jumps[k][0]) for k in range(r.njumps)}; out = {}
        for p in s.ep:
            pl = r.pl[p]; cid = int(pl.char_id); x0, y0 = s.pos[p]; dx, dy = float(pl.x - x0), float(pl.y - y0); act = [0, 0, 0, 0]
            if cid not in jump and cid not in s.pjump and dx * dx + dy * dy >= (10.0 * self.ticks) ** 2 and pl.alive:
                act[0] = 1 + int(np.round(np.arctan2(dy, dx) / (np.pi / 8))) % 16
            for c in rec[(rec[:, 1] == cid) & (rec[:, 2] == 1)]:
                if c[6] == 1: continue
                f = min(1.0, max(0.0, (float(c[0]) - s.ptick) / max(1.0, float(r.tick - s.ptick))))
                tx, ty = float(c[3] - (x0 + f * dx)), float(c[4] - (y0 + f * dy)); ang = int(round(np.degrees(np.arctan2(ty, tx)))) % 360
                if c[5] == 1 and act[1] != 2: act[1:] = [2, ang, min(7, max(0, int(np.ceil(np.hypot(tx, ty) / 2600.0 * 8)) - 1))]
                elif c[5] == 0 and act[1] == 0: act[1], act[2] = 1, ang
            out[p] = act
        return out

    def _restart(self, i, why):
        self.crashes.append({"why": str(why)[:200], **(self.I[i].cfg or {}).get("tag", {})})
        self._trans += getattr(self.I[i], "hold", {}).values(); self.I[i].hold = {}   # crashed: own placement only
        return super()._restart(i, why)

    def _start_battle(self, i, r):
        super()._start_battle(i, r)
        s = self.I[i]; s.hold = {}                       # player -> its final transition, held until the team's rank is known
        s.sl = slider_state(s.map_tiles, s.W, s.H, (s.cfg.get("tag") or {}).get("chr", []), self.rng)
        s.pos = {p: (r.pl[p].x, r.pl[p].y) for p in range(10)}; s.ptick = r.tick; s.pjump = set()

    def _finish_step(self, i, r):
        s = self.I[i]; sl = s.sl; T = slider_terms(sl, r.pl)
        extra = (T * sl["w"]).sum(1) * self.slider_scale; sl["sum"] += T; sl["ret"] += extra
        n0, e0 = len(self._trans), len(self.episodes)
        lab = self._labels(s, r) if self.record and s.cfg.get("native_ctl") else {}
        s.pos = {p: (r.pl[p].x, r.pl[p].y) for p in range(10)}; s.ptick = r.tick; s.pjump = {int(r.jumps[k][0]) for k in range(r.njumps)}
        g = np.array((s.cfg.get("tag") or {}).get("groups") or [0] * 10)
        if g.any(): pr, dead = player_ranks(r); b0 = s.battle   # read r before a reset reuses it
        super()._finish_step(i, r)                       # may reset the instance: sl stays this battle's
        for k in range(n0, len(self._trans)):
            aid, rew, fin, t, rank = self._trans[k]; p = aid & 15
            self._trans[k] = (aid, rew + float(extra[p]), fin, {**t, **{"s_" + m: float(v) for m, v in zip(SL, T[p])}, "_act": lab.get(p)}, rank)
        if g.any():                                      # duo / trio: placement = the team's (Solo Showdown underneath)
            for k in range(len(self._trans) - 1, n0 - 1, -1):
                p = self._trans[k][0] & 15
                if self._trans[k][2] and g[p]: s.hold[p] = self._trans.pop(k)
            tr = team_ranks(g, pr, dead, s.battle != b0)
            for p in [p for p in s.hold if not np.isnan(tr[p])]:
                aid, rew, fin, t, rank = s.hold.pop(p); place = (10 - tr[p]) / 9.0; win = float(tr[p] == 1)
                rew += self.w.get("place", 0) * (place - t["place"]) + self.w.get("win", 0) * (win - t["win"])
                self._trans.append((aid, rew, fin, {**t, "team_place": place}, rank))
        for row in self.episodes[e0:]:
            p = row["player"]; row["ret"] += float(sl["ret"][p]); row["sliders"] = [round(float(v), 3) for v in sl["w"][p]]
            row["terms"].update({"s_" + m: round(float(v), 4) for m, v in zip(SL, sl["sum"][p])})

    def cond(self, ids):
        """(B, 26) conditioning vectors (obs_v5.cond) of the agents of the last poll()"""
        by = {s.battle: s for s in self.I}
        out = []
        for a in ids:
            s, p = by[int(a) >> 4], int(a) & 15; tag = s.cfg["tag"]
            out.append(O.cond(tag["chr"][p], s.sl["w"][p], tag["team_size"] if tag["groups"][p] else 1))
        return np.stack(out) if out else np.zeros((0, O.N_COND), np.float32)


def _check():
    from types import SimpleNamespace as NS
    def players(**over):
        pl = [NS(x=1650 + 3000 * (p % 5), y=1650 + 6000 * (p // 5), hp=3000, max_hp=4000, damage_dealt=0, bomb_damage=0, damage_taken=0,
                 box_damage=0, cubes=0, cubes_total=0, reserved=[0, 0, p], alive=1, eliminated=0) for p in range(10)]
        for k, v in over.items(): setattr(pl[int(k[1])], k[3:], v)
        return pl
    tiles = np.full((40, 60), 27000000); bush = D.table("tiles.csv")["Forest"][0]; wall = D.table("tiles.csv")["Indestructible"][0]
    tiles[5, 5] = bush; tiles[20:23, 30] = bush; tiles[25, 10] = wall
    sl = slider_state(tiles.ravel(), 60, 40, ["Sniper"] * 10, np.random.default_rng(0))
    assert sl["w"].shape == (10, 10) and 0 <= sl["w"].min() and sl["w"].max() <= 2 and sl["range"][0] == 3000
    assert sl["bush"].sum() == 4 and sl["nearcover"][23, 8] and sl["nearcover"][27, 12] and not sl["nearcover"][28, 10]
    col = lambda T, name: T[:, SL.index(name)]
    T = slider_terms(sl, players())                      # p0 at tile (5, 5) = the bush, nearest enemy 10 tiles away
    assert col(T, "bush_camp")[0] == 0.01 and col(T, "bush_camp")[1:].sum() == 0 and col(T, "bush_control")[0] == 0.005
    T = slider_terms(sl, players(p0_damage_dealt=500, p0_box_damage=1000, p0_cubes=1, p1_damage_taken=800))
    assert np.isclose(col(T, "bush_camp")[0], 0.51) and col(T, "collect")[0] == 2.0 and col(T, "bush_control")[0] == 0
    assert col(T, "kite")[0] == 0.5 and col(T, "focus")[0] == 0.5 and col(T, "third_party")[0] == 0.5 and col(T, "peek")[0] == 0
    assert col(T, "kite")[1] == 0 and col(T, "protect").sum() == 0 and col(T, "push").sum() == 0
    T = slider_terms(sl, players(p0_hp=1000)); assert col(T, "retreat")[0] == 0 and sl["low"][0]
    T = slider_terms(sl, players(p0_hp=1600)); assert np.isclose(col(T, "retreat")[0], 0.6)
    T = slider_terms(sl, players(p0_hp=3500)); assert np.isclose(col(T, "retreat")[0], 1.9) and not sl["low"][0]
    T = slider_terms(sl, players(p0_hp=3600, p0_x=1800)); assert col(T, "retreat")[0] == 0 and np.isclose(col(T, "push")[0], 0.01)
    mates = players(p2_damage_taken=1000, p1_x=3000); mates[1].reserved[2] = 0; mates[2].reserved[2] = 0
    T = slider_terms(sl, mates); assert np.allclose(col(T, "protect")[:3], [0.01, 0.01, 0])    # p2 is 15-20 tiles away
    mates[2].x = 2500                                    # p2 now 850 / 500 from p0 / p1: its damage counts for both
    T = slider_terms(sl, mates); assert np.isclose(col(T, "protect")[0], -1 + 0.01) and np.isclose(col(T, "protect")[1], -1 + 0.01)
    dead = players(p0_alive=0, p0_damage_dealt=900); assert not slider_terms(sl, dead)[0].any()
    pl = players(p0_alive=0, p1_eliminated=1, p2_eliminated=1, p3_eliminated=1)          # team placement
    for p, k in ((0, 9), (1, 8), (2, 10), (3, 7)): pl[p].elim_rank = k
    pr, dd = player_ranks(NS(done=0, alive=6, winner=-1, pl=pl))
    assert list(pr[:5]) == [9, 8, 10, 7, 3.5] and list(dd[:5]) == [1, 1, 1, 1, 0]
    g = [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]                   # team 1 out (best 8), team 2 alive through p4, rest alone
    tr = team_ranks(g, pr, dd, False); assert list(tr[:2]) == [8, 8] and np.isnan(tr[2:5]).all() and tr[5] == 3.5
    assert list(team_ranks(g, pr, dd, True)[:5]) == [8, 8, 3.5, 3.5, 3.5]                     # cut short: p4's estimate
    assert np.array_equal(team_ranks([0] * 10, pr, dd, False), pr)                            # solo: own rank
    def fake(self, i, r):                                # MAEnv._finish_step stand-in: the final transitions of r.fin
        for p in r.fin: rk = r.pl[p].elim_rank; self._trans.append((16 + p, 1.0, True, {"place": (10 - rk) / 9, "win": 0.0}, rk))
        self.I[i].battle += r.over
    env = Env.__new__(Env); env._trans, env.episodes, env.w, env.slider_scale = [], [], {"place": 5.0, "win": 3.0}, 0.0
    for groups, want in (([0] * 10, [1.0, 1.0]), ([1, 1] + [0] * 8, [1.0, 1.0 + 5 * (10 - 3) / 9 - 5 * (10 - 8) / 9])):
        env.I = [NS(sl=sl, cfg={"tag": {"groups": groups}}, hold={}, battle=1, pos={}, ptick=0, pjump=set())]
        mk = lambda fin, over, **o: NS(pl=players(**o), fin=fin, over=over, tick=1, njumps=0, jumps=[], done=0, alive=8, winner=-1)
        r1 = mk([1], 0, p1_eliminated=1); r1.pl[1].elim_rank = 8
        r2 = mk([0], 0, p0_eliminated=1, p1_eliminated=1); r2.pl[0].elim_rank = 3; r2.pl[1].elim_rank = 8
        sup, MAEnv._finish_step = MAEnv._finish_step, fake
        try:
            env._finish_step(0, r1); n1 = len(env._trans); env._finish_step(0, r2)
        finally: MAEnv._finish_step = sup
        got = {a & 15: rw for a, rw, *_ in env._trans}; env._trans = []
        assert n1 == (0 if groups[0] else 1) and np.allclose([got[0], got[1]], want), (groups, n1, got)   # p1 held until p0 out
    print("env_v5 sliders ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--agents", type=int, default=10); ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--out", default="bench.jsonl")
    ap.add_argument("--check", action="store_true", help="slider reward self-check (no game)")
    a = ap.parse_args()
    if a.check: return _check()
    env = Env(a.n, make_battle_fn(a.agents), ticks=a.ticks, fair=True, max_ticks=4000)
    rng = np.random.default_rng(0); t0 = None; steps = 0
    try:
        while t0 is None or time.time() - t0 < a.minutes * 60:
            obs, ids, _ = env.poll(min_ready=1, max_wait=0.003)
            if not len(ids): continue
            if t0 is None: t0 = time.time()     # clock starts at the first observation (after boot)
            B = len(ids); steps += B
            if steps == B: assert env.cond(ids).shape == (B, O.N_COND)
            act = np.stack([rng.integers(0, 17, B), rng.integers(0, 3, B), rng.integers(0, 360, B), rng.integers(0, 8, B)], 1)
            env.send(ids, act)
    finally:
        dt = time.time() - (t0 or time.time()); env.close()
    with open(a.out, "w") as f:
        for c in env.crashes: f.write(json.dumps(c) + "\n")
    print("n %d agents %d ticks %d: %.0f decisions/s, %d battles, %d episodes, %d restarts (%s), %d brawlers, %d maps" % (
        a.n, a.agents, a.ticks, steps / max(dt, 1e-9), env.stats["battles"], len(env.episodes), env.stats["restarts"],
        a.out, len(heroes()), len(locations())))
    rows = [e["terms"] for e in env.episodes if "sliders" in e]
    print("slider terms per episode (mean over %d): %s" % (len(rows), {m: round(float(np.mean([t["s_" + m] for t in rows])), 3) for m in SL} if rows else {}))


if __name__ == "__main__":
    sys.exit(main())
