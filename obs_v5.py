"""Vision-shaped observation for the v5 policy: what the two vision models give the real bot, nothing more.

    python -m obs_v5            # self-check (no game)
    python -m obs_v5 --live     # + 1 instance, 300 decisions: real MAEnv observations through the builder

    ob = ObsV5()                                     # one per MAEnv
    o = ob.build(obs["vec"], obs["grid"], env.last_meta, ids)
    o["self"] (B, 8)   hp fraction, ammo 0-1, super / gadget / hypercharge ready, in bush, own velocity x, y
    o["ent"]  (B, 39, 16)  present | class one-hot (ENT) | dx, dy (10 tiles, x right / y down) | vx, vy (tiles/tick) | hp fraction
    o["grid"] (B, 25, 25) uint8  seg class id (SEG) per tile around the player, UNKNOWN outside the camera view
    cond(brawler, sliders, team_size) (26,)  own brawler stats + behaviour weights + team size

Source = the host's state2 vector + grid after nbpc.fair (already camera- and bush-limited). Needs the rl-round
host `nbhost_rl` (enemy features 14 / 15 = summon / on my team, self 18 = hypercharge fraction, grid channel 0 =
tiles.csv row + 1); with an older host teammates read as enemies and the tiles are wrong.
shortcut: area effects have no radius / kind (one class per side), the host dumps no gadget state (self 3 stays 0), and faded_bush / power_box tiles are
render states the tile map does not have; add them to snap2 / encode.c when the policy needs them.
shortcut: no box size (the sim has none; a per-class constant adds nothing to the one-hot); add with the bot adapter.
"""
import functools, sys
import numpy as np
from nbpc import fair as F, diversity as D

# same order as tools/libg_qemu/classes.py SEG_CLASSES (the seg model's class ids)
SEG = ("floor", "indestructible", "destructible_wall", "fragile_wall", "bush", "faded_bush", "water", "rope_fence",
       "bouncer", "gravity", "slow_tile", "fast_tile", "ice", "damage_tile", "power_box", "poison", "out_of_bounds",
       "destructible")
UNKNOWN = len(SEG)
ENT = ("enemy", "teammate", "enemy_summon", "teammate_summon", "power_box", "power_cube", "enemy_projectile",
       "teammate_projectile", "enemy_area_effect", "teammate_area_effect")
SLIDERS = ("bush_camp", "collect", "protect", "retreat", "kite", "peek", "focus", "push", "third_party", "bush_control")
N_SELF, N_ENT, F_ENT, G = 8, F.EN_K + F.PJ_K + F.BX_K + F.IT_K, 1 + len(ENT) + 5, 25
XY, VEL, HP = 1 + len(ENT), 3 + len(ENT), 5 + len(ENT)
N_COND = 13 + len(SLIDERS) + 3
_S, _E = SEG.index, ENT.index

# Ranges, sampled once per episode. Guesses until rl/measure_noise.py measures the exported models on the v5 val split.
NOISE = {"jitter": (0.0, 0.03),    # box centre sigma, 10-tile units (0.03 = 0.3 tile)
         "drop": (0.0, 0.10),      # chance per entity and decision that it is not detected
         "fp": (0.0, 0.01),        # chance per empty slot and decision of a false detection
         "seg": (0.0, 0.03),       # chance per visible tile and decision of a wrong class
         "latency": (1, 2)}        # decisions between the frame and the action (inclusive)


@functools.lru_cache(None)
def _tile_lut():
    """host grid channel 0 (tiles.csv row + 1; 0 = outside the map) -> seg class id.
    The rule of tools/libg_qemu/classes.py seg_class, on this game version's tiles.csv."""
    lut = np.full(256, _S("floor"), np.uint8); lut[0] = _S("out_of_bounds")
    for name, (i, r) in D.table("tiles.csv").items():
        t = lambda k: r.get(k) == "true"
        num = lambda k: int(r.get(k) or 0)
        if t("IsForest"): c = "bush"
        elif t("IsBouncer"): c = "bouncer"
        elif name.startswith("Gravity"): c = "gravity"
        elif name == "Fragile": c = "destructible"
        elif t("BlocksMovement") and t("BlocksProjectiles"):
            c = ("fragile_wall" if t("IsDestructibleNormalWeapon") or num("Health") > 0 else
                 "destructible_wall" if t("IsDestructible") else "indestructible")
        elif t("BlocksMovement"): c = "rope_fence" if t("IsDestructible") else "water"
        elif num("SpeedChange") < 0: c = "slow_tile"
        elif num("SpeedChange") > 0: c = "fast_tile"
        elif name == "Ice": c = "ice"
        elif num("Damage") > 0 or name == "IntervalDamage": c = "damage_tile"
        else: c = "floor"
        if 0 <= i % 1000000 < 255: lut[i % 1000000 + 1] = _S(c)
    return lut


def from_state2(vec, grid, meta):
    """fair state2 vec (B, 436), grid (B, 6, 25, 25), meta (B, 5) = x, y, tick, W, H (MAEnv.last_meta) -> clean obs"""
    B = len(vec)
    s = np.zeros((B, N_SELF), np.float32)
    s[:, 0] = vec[:, 0]; s[:, 1] = np.minimum(vec[:, 2], 1.0); s[:, 2] = vec[:, 5] >= 1.0; s[:, 5] = vec[:, 8]
    s[:, 4] = vec[:, 18] >= 1.0
    s[:, 6:8] = vec[:, 12:14] * (40 / 300)
    ent = np.zeros((B, N_ENT, F_ENT), np.float32)

    def fill(k0, blk, cls, vel=None, hp=None):
        e = ent[:, k0:k0 + blk.shape[1]]
        e[..., 0] = blk[..., 0] > 0.5
        e[..., XY:XY + 2] = blk[..., 1:3]
        if vel: e[..., VEL:VEL + 2] = blk[..., 3:5] * vel
        if hp: e[..., HP] = blk[..., hp]
        np.put_along_axis(e, (1 + cls)[..., None], 1.0, axis=2)
        e *= e[..., :1]
        return k0 + blk.shape[1]

    blk = lambda off, K, Fn: vec[:, off:off + K * Fn].reshape(B, K, Fn)
    en, pj, bx, it = blk(F.EN0, F.EN_K, F.EN_F), blk(F.PJ0, F.PJ_K, F.PJ_F), blk(F.BX0, F.BX_K, F.BX_F), blk(F.IT0, F.IT_K, F.IT_F)
    # ENT order: enemy, teammate, enemy_summon, teammate_summon = 2 * summon (14) + on my team (15)
    k = fill(0, en, (2 * (en[..., 14] > 0.5) + (en[..., 15] > 0.5)).astype(np.int64), vel=40 / 300, hp=5)
    k = fill(k, pj, np.where(pj[..., 5] > 0.5, _E("teammate_projectile"), _E("enemy_projectile")), vel=200 / 300)
    k = fill(k, bx, np.full(bx.shape[:2], _E("power_box")), hp=3)
    it = it.copy(); it[..., 0] *= (it[..., 3] > 0.5) | (np.abs(it[..., 4]) > 0.5)  # other items are not detected classes
    # item 4: 1 = showdown bomb or enemy area effect, -1 = area effect of my team (nbhost_rl encode.c)
    fill(k, it, np.select([it[..., 3] > 0.5, it[..., 4] < -0.5], [_E("power_cube"), _E("teammate_area_effect")], _E("enemy_area_effect")))

    g = _tile_lut()[grid[:, 0]]
    g[(grid[:, 3] > 0) & (g == _S("floor"))] = _S("poison")
    g[grid[:, 4] > 0] = _S("out_of_bounds")
    rx, ry = _cam(meta)
    seen = (np.abs(rx) < F.VIEW_X)[:, None, :] & ((ry > -F.VIEW_UP) & (ry < F.VIEW_DOWN))[:, :, None]
    g[~seen] = UNKNOWN
    return {"self": s, "ent": ent, "grid": g}


def _cam(meta):
    """tile centres relative to the camera, the rule of nbpc.fair: rx (B, G) from the camera x, ry (B, G) from the player y"""
    mx, my, W = meta[:, 0].astype(np.float64), meta[:, 1].astype(np.float64), meta[:, 3]
    off = (np.arange(G) - G // 2) * F.TILE + F.TILE / 2
    rx = (mx // F.TILE * F.TILE)[:, None] + off - np.clip(mx, 3000.0, W * F.TILE - 3000.0)[:, None]
    return rx, (my // F.TILE * F.TILE - my)[:, None] + off


# HUD controls (repos/nbgc/labeller/controls.py CTRL, 960 x 540 label pixels): centre x, y, radius; the seg model says 'ui' there
HUD = ((130, 455, 54), (822, 404, 54), (722, 445, 24), (670, 497, 24), (785, 497, 28))


def hud_mask(meta):
    """(B, G, G) bool: tiles whose centre lies under a fixed HUD control (joysticks, super / hypercharge / gadget buttons).
    shortcut: screen = 480 + 0.085 * rx, 297.5 + 0.085 * ry px (view height-bound, ~85 px per 1000 units, y -3500..+2900);
    replace by a measured mask once the sim-vs-labels test exists."""
    rx, ry = _cam(meta)
    px, py = 480 + 0.085 * rx[:, None, :], 297.5 + 0.085 * ry[:, :, None]
    return np.any([(px - cx) ** 2 + (py - cy) ** 2 <= r * r for cx, cy, r in HUD], 0)


class ObsV5:
    """builder + per-episode noise; noise=None gives the clean observation. Noisy: entity velocity = differenced noisy
    positions (0 on first sight), HUD-covered tiles UNKNOWN."""

    GATE = 0.1   # 1 tile: a detection within this of where a last-decision detection of its class should be = the same object

    def __init__(self, noise=NOISE, seed=0, ticks=2):
        self.noise, self.rng, self.ep, self.ticks = noise, np.random.default_rng(seed), {}, ticks   # ticks per decision (Env)

    def _episode(self, aid):
        """agent id -> [jitter, drop, fp, seg, latency, frames, last decision's detections]"""
        e = self.ep.get(aid)
        if e is None:
            n, r = self.noise, self.rng
            e = self.ep[aid] = [r.uniform(*n["jitter"]), r.uniform(*n["drop"]), r.uniform(*n["fp"]), r.uniform(*n["seg"]),
                                int(r.integers(n["latency"][0], n["latency"][1] + 1)), [], None]
            if len(self.ep) > 4096:   # shortcut: ended episodes are never reported here, so drop the oldest half
                for k in list(self.ep)[:2048]: del self.ep[k]
        return e

    def build(self, vec, grid, meta, ids):
        o = from_state2(vec, grid, meta)
        if self.noise is None or not len(ids):
            return o
        r, B = self.rng, len(ids)
        eps = [self._episode(int(a)) for a in ids]
        jit, drop, fp, seg = (np.array([e[i] for e in eps], np.float32) for i in range(4))
        ent, g = o["ent"], o["grid"]
        here = ent[..., 0] > 0.5
        false = ~here & (r.random(here.shape) < fp[:, None])
        ent[here & (r.random(here.shape) < drop[:, None])] = 0.0
        nse = r.normal(size=(B, N_ENT, 2)).astype(np.float32) * jit[:, None, None] * ent[..., :1]
        ent[..., XY:XY + 2] += nse
        n = int(false.sum())
        if n:
            row = np.zeros((n, F_ENT), np.float32); row[:, 0] = 1.0
            row[np.arange(n), 1 + r.integers(len(ENT), size=n)] = 1.0
            row[:, XY] = r.uniform(-1.5, 1.5, n); row[:, XY + 1] = r.uniform(-F.VIEW_UP, F.VIEW_DOWN, n) / F.R
            row[:, HP] = r.random(n)
            ent[false] = row
        g[hud_mask(meta)] = UNKNOWN                       # the seg model labels the HUD controls 'ui'
        wrong = (g != UNKNOWN) & (r.random(g.shape) < seg[:, None, None])
        g[wrong] = r.integers(len(SEG), size=int(wrong.sum()))
        # velocity as the bot gets it: the difference of two noisy detections, 0 for one not matched to the last decision
        on, cls = ent[..., 0] > 0.5, ent[..., 1:1 + len(ENT)].argmax(-1)
        move = (ent[..., VEL:VEL + 2] - o["self"][:, None, 6:8]) * (self.ticks / 10)   # tiles/tick -> 10-tile units per decision
        for b, e in enumerate(eps):
            v = np.zeros((N_ENT, 2), np.float32)
            if e[6] is not None:
                pon, pcls, pxy, pn = e[6]
                d = np.linalg.norm(ent[b, :, None, XY:XY + 2] - move[b, :, None] - pxy[None], axis=2)
                d[~(on[b][:, None] & pon[None] & (cls[b][:, None] == pcls[None]))] = np.inf
                m = d.min(1) < self.GATE; j = d.argmin(1)[m]
                v[m] = ent[b, m, VEL:VEL + 2] + (nse[b, m] - pn[j]) * (10 / self.ticks)
            e[6] = (on[b], cls[b], ent[b, :, XY:XY + 2].copy(), nse[b])
            ent[b, :, VEL:VEL + 2] = v
        for b, e in enumerate(eps):                       # latency: the policy acts on the frame of e[4] decisions ago
            q = e[5]; q.append({k: v[b].copy() for k, v in o.items()})
            if len(q) > e[4] + 1: q.pop(0)
            for k in o: o[k][b] = q[0][k]
        return o


@functools.lru_cache(None)
def _tables():
    return ({r["Name"]: r for r in D._rows("skills.csv")}, {r["Name"]: r for r in D._rows("projectiles_logic.csv")})


@functools.lru_cache(None)
def brawler_stats(name):
    """(13,) range, speed, reload, projectile speed, spread, bullets, thrown, melee, hp, super attack / charge / other,
    super range; from characters.csv / skills.csv / projectiles_logic.csv, scaled to about 0-1"""
    sk, pj = _tables()
    c = D.table("characters.csv")[name][1]
    w, u = sk.get(c.get("WeaponSkill"), {}), sk.get(c.get("UltimateSkill"), {})
    p = pj.get(w.get("Projectiles"), {})
    num = lambda r, k: float(r.get(k) or 0)
    ub = u.get("BehaviorType")
    v = np.array([num(w, "CastingRange") / 30, num(c, "Speed") / 1000, num(w, "RechargeTime") / 3000, num(p, "Speed") / 4000,
                  num(w, "Spread") / 300, num(w, "NumBulletsInOneAttack") / 10, p.get("Indirect") == "true", not p,
                  num(c, "Hitpoints") / 10000, ub == "Attack", ub == "Charge", ub not in ("Attack", "Charge"),
                  num(u, "CastingRange") / 30], np.float32)
    return np.minimum(v, 2.0)    # a few rows are far out (a 29000-speed projectile, 120-tile supers)


def cond(brawler, sliders, team_size=1):
    """conditioning vector: brawler stats (13) + slider weights 0-2 (10, order SLIDERS) + team size 1-3 one-hot"""
    t = np.zeros(3, np.float32); t[team_size - 1] = 1.0
    return np.concatenate([brawler_stats(brawler), np.asarray(sliders, np.float32), t])


def _check():
    vec = np.zeros((2, 436), np.float32); grid = np.zeros((2, 6, 25, 25), np.uint8)
    meta = np.array([[9150, 9150, 100, 60, 60], [150, 9150, 100, 60, 60]], np.int32)   # mid-map; left edge
    vec[:, 0] = 0.5; vec[:, 2] = 2 / 3; vec[:, 5] = 1.0; vec[:, 12] = 1.0
    row = lambda off, Fn, k, v: vec.__setitem__((0, slice(off + k * Fn, off + k * Fn + len(v))), v)
    row(F.EN0, F.EN_F, 0, [1, 0.5, -0.2, 1, 0, 0.8])
    row(F.EN0, F.EN_F, 1, [1, 0.6, 0, 0, 0, 1] + [0] * 9 + [1]); row(F.EN0, F.EN_F, 2, [1, 0.7, 0, 0, 0, 1] + [0] * 8 + [1, 0])
    row(F.EN0, F.EN_F, 3, [1, 0.8, 0, 0, 0, 1] + [0] * 8 + [1, 1]); vec[0, 18] = 1.0
    tile = lambda n: D.table("tiles.csv")[n][0] % 1000000 + 1
    grid[:, 0] = tile("Open")
    row(F.PJ0, F.PJ_F, 0, [1, 0.1, 0.1, 1, 0, 1, 0]); row(F.PJ0, F.PJ_F, 1, [1, 0.2, 0.1, 0, 1, 0, 1])
    row(F.BX0, F.BX_F, 0, [1, 0.3, 0.3, 0.6])
    row(F.IT0, F.IT_F, 0, [1, 0.4, 0, 1, 0]); row(F.IT0, F.IT_F, 1, [1, 0.4, 0, 0, 1]); row(F.IT0, F.IT_F, 2, [1, 0.4, 0, 0, 0]); row(F.IT0, F.IT_F, 3, [1, 0.4, 0, 0, -1])
    grid[0, 0, 12, 13] = tile("Indestructible"); grid[0, 0, 12, 14] = tile("Forest"); grid[0, 3, 12, 14] = 1
    grid[0, 0, 12, 15] = tile("Wall1"); grid[0, 0, 12, 16] = tile("Water"); grid[0, 0, 12, 17] = tile("RopeFence")
    grid[0, 3, 5, 5] = 1; grid[0, 4, 3, 3] = 1; grid[0, 0, 3, 3] = 0
    o = from_state2(vec, grid, meta)
    assert o["self"].shape == (2, N_SELF) and o["ent"].shape == (2, N_ENT, F_ENT) and o["grid"].shape == (2, G, G)
    assert np.allclose(o["self"][0], [0.5, 2 / 3, 1, 0, 1, 0, 40 / 300, 0])
    e = o["ent"][0]; cls = lambda k: ENT[int(e[k, 1:1 + len(ENT)].argmax())]
    got = [(k, cls(k)) for k in np.nonzero(e[:, 0])[0]]
    assert got == [(0, "enemy"), (1, "teammate"), (2, "enemy_summon"), (3, "teammate_summon"), (9, "teammate_projectile"),
                   (10, "enemy_projectile"), (25, "power_box"), (31, "power_cube"), (32, "enemy_area_effect"), (34, "teammate_area_effect")], got
    assert np.allclose(e[0, XY:], [0.5, -0.2, 40 / 300, 0, 0.8]) and np.allclose(e[10, VEL:VEL + 2], [0, 200 / 300])
    assert e[:, 1:1 + len(ENT)].sum() == 10 and not o["ent"][1].any()
    g = o["grid"][0]
    assert (g[12, 13], g[12, 14], g[5, 5], g[3, 3], g[12, 12]) == (_S("indestructible"), _S("bush"), _S("poison"),
                                                                 _S("out_of_bounds"), _S("floor"))
    assert (g[12, 15], g[12, 16], g[12, 17]) == (_S("destructible_wall"), _S("water"), _S("rope_fence"))
    assert (g[0] == UNKNOWN).all() and (g[1] != UNKNOWN).all() and (g[21] != UNKNOWN).all() and (g[22:] == UNKNOWN).all()
    assert (g[1:22] != UNKNOWN).all()                          # +-12 tiles is inside the +-15 tile view mid-map
    g1 = o["grid"][1]                                          # x = 150: camera centred 10 tiles right, sees x > -1500
    assert (g1[5, :7] == UNKNOWN).all() and (g1[5, 7:] != UNKNOWN).all()

    ids = np.array([16, 17])
    clean = ObsV5(noise=None).build(vec, grid, meta, ids)
    assert all(np.array_equal(clean[k], o[k]) for k in o)
    lat = ObsV5(noise={"jitter": (0, 0), "drop": (0, 0), "fp": (0, 0), "seg": (0, 0), "latency": (2, 2)})
    outs = []
    for t in range(5):
        v = vec.copy(); v[:, 0] = t / 10
        outs.append(lat.build(v, grid, meta, ids)["self"][0, 0])
    assert np.allclose(outs, [0, 0, 0, 0.1, 0.2]), outs       # 2 decisions late, the first frame repeated until then
    hi = {"jitter": (0.03, 0.03), "drop": (0.5, 0.5), "fp": (0.2, 0.2), "seg": (0.2, 0.2), "latency": (0, 0)}
    hud = hud_mask(meta)
    assert hud[0, 18, 21] and not hud[0, 12, 12] and 0 < hud[0].sum() < 40, hud[0].sum()   # super button, 6 / 9 tiles off
    nz = ObsV5(noise=hi); kept = fps = flips = 0; d = []; on0 = False; back = vback = 0
    for t in range(300):
        n = nz.build(vec, grid, meta, ids)
        assert all(n[k].shape == o[k].shape and n[k].dtype == o[k].dtype for k in o) and n["grid"].max() <= UNKNOWN
        assert np.array_equal(n["grid"] == UNKNOWN, (o["grid"] == UNKNOWN) | hud) and np.array_equal(n["self"], o["self"])
        on = n["ent"][0, :, 0] > 0; was = o["ent"][0, :, 0] > 0
        kept += (on & was).sum(); fps += (on & ~was).sum(); flips += (n["grid"] != o["grid"]).mean()
        if on[0]: d.append(n["ent"][0, 0, XY] - 0.5)
        back += on[0] and not on0; vback += on[0] and not on0 and n["ent"][0, 0, VEL:VEL + 2].any()
        on0 = on[0]
    assert back > 50 and vback <= 3, (back, vback)       # back after a drop: no velocity (rare: matched to a false positive)
    assert 0.4 < kept / (300 * 9) < 0.6 and 0.15 < fps / (300 * (N_ENT - 9)) < 0.25, (kept, fps)
    assert 0.1 < flips / 300 < 0.2 and 0.02 < np.std(d) < 0.04, (flips, np.std(d))
    vj = ObsV5(noise=dict(hi, drop=(0, 0), fp=(0, 0), seg=(0, 0))); dv = []   # enemy 0: moves with the player (rel. 0)
    for t in range(300):
        e0 = vj.build(vec, grid, meta, ids)["ent"][0, 0]
        if t == 0: assert not e0[VEL:VEL + 2].any()       # first sight: no velocity
        elif e0[VEL:VEL + 2].any(): dv.append(e0[VEL:VEL + 2] - [40 / 300, 0])
    sd = np.std(dv, 0)                                    # differenced jitter: sqrt(2) * 0.03 * 10 / 2 ticks = 0.21 tiles/tick
    assert len(dv) > 250 and (0.17 < sd).all() and (sd < 0.26).all(), (len(dv), sd)
    ex = ObsV5(noise=dict(hi, jitter=(0, 0), drop=(0, 0), fp=(0, 0), seg=(0, 0))); ex.build(vec, grid, meta, ids)
    assert np.allclose(ex.build(vec, grid, meta, ids)["ent"][0, 0, VEL:VEL + 2], [40 / 300, 0])   # no jitter: exact
    many = ObsV5(); [many._episode(a) for a in range(5000)]
    assert len(many.ep) <= 4096 and 4999 in many.ep

    import env_v5
    hs = env_v5.heroes()
    st = np.stack([brawler_stats(h) for h in hs])
    assert st.shape == (len(hs), 13) and np.isfinite(st).all() and st.min() >= 0 and st.max() <= 2, (st.min(), st.max())
    assert np.allclose(brawler_stats("ShotgunGirl")[[0, 1, 3, 6, 7, 8]], [23 / 30, 0.77, 3100 / 4000, 0, 0, 0.39])
    assert brawler_stats("Barkeep")[6] == 1 and brawler_stats("Undertaker")[7] == 1
    c = cond("Sniper", np.ones(10), 3)
    assert c.shape == (N_COND,) and c[-1] == 1 and c[-3] == 0
    print("obs_v5 ok: %d brawlers, %d thrown, %d melee, cond %d, ent %dx%d" % (
        len(hs), st[:, 6].sum(), st[:, 7].sum(), N_COND, N_ENT, F_ENT))


def _live(decisions=300):
    import collections, time, env_v5
    env = env_v5.Env(1, env_v5.make_battle_fn(10, p_team=1.0), ticks=2, fair=True, max_ticks=4000)
    ob, rng, n, seen, t = ObsV5(noise=None), np.random.default_rng(0), 0, collections.Counter(), 0.0   # clean: a real class census
    try:
        while n < decisions:
            obs, ids, _ = env.poll(min_ready=1, max_wait=0.003)
            if not len(ids): continue
            if not n:                                     # the host's teams (player +0xc) = the requested groups
                s = env.I[0]; gr = s.cfg["tag"]["groups"]; tm = [s.inst.resp.pl[p].reserved[2] for p in range(10)]
                same = [[tm[a] == tm[b] for b in range(10)] for a in range(10)]
                want = [[a == b or (gr[a] and gr[a] == gr[b]) for b in range(10)] for a in range(10)]
                assert same == want, (gr, tm)
                print("teams ok: groups %s -> teams %s" % (gr, tm), flush=True)
            t0 = time.perf_counter(); o = ob.build(obs["vec"], obs["grid"], env.last_meta, ids); t += time.perf_counter() - t0
            assert all(np.isfinite(v).all() for v in o.values()) and o["self"][:, :2].max() <= 1.0, o["self"].max(0)
            assert np.abs(o["ent"][..., XY:XY + 2]).max() < 2.0 and o["grid"].max() <= UNKNOWN
            for c in o["ent"][..., 1:1 + len(ENT)].sum((0, 1)).nonzero()[0]: seen[ENT[c]] += 1
            for c in np.unique(o["grid"]): seen["seg:" + (SEG + ("unknown",))[c]] += 1
            n += len(ids)
            env.send(ids, np.stack([rng.integers(0, 17, len(ids)), rng.integers(0, 3, len(ids)),
                                    rng.integers(0, 360, len(ids)), rng.integers(0, 8, len(ids))], 1))
    finally:
        env.close()
    print("live ok: %d decisions, %.0f us per decision, polls with class: %s" % (n, t / n * 1e6, dict(seen)))


if __name__ == "__main__":
    _check()
    if "--live" in sys.argv: _live()
