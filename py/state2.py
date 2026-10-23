#!/usr/bin/env python3
"""Privileged state v2 (learn round S0): parse the agent's snap2 dump and encode it egocentrically for any player.

Binary layout (int32, agent CModule snap2):
  header [0] magic 'NB2S' [1] total ints [2] tick [3] nPlayers [4] nChars [5] nProj [6] nItems [7] reserved
  players  (P2=8)  index, team, characterId, controlledId, superCharge(+0x54), superMax(+0x58), hudAlive(+0x48), bot(+0x238)
  chars    (C2=20) id, dataId, owner(player idx), x, y, z, hp, maxHp, state(+0x374), facing(+0x328), lastDamageTick(+0x7d8),
                   spawnProtection(+0x431), f469, ammoCharge(weapon +0x18, 1000/shot), weaponCooldownMs(+0x10),
                   ultiActive(skill1 +0x30), lastAttackerPlayer(+0xaa0), characterType(data +0x23c), vtKind(0 char, 1 other),
                   client (1 = present in the client's object list = in the viewer's view, what gets rendered),
                   cx, cy, cz (the client object's rendered position +0x30/+0x34/+0x38, NIL when absent)
  proj     (J2=18) id, dataId, owner, x, y, z, prevX(+0x9c), prevY(+0xa0), sourceId, damage(+0xd4), travel‰(+0x144),
                   travelStart‰(+0x148), finished(+0x90), impact(+0x114), client, cx, cy, cz
  items    (I2=12) id, dataId, owner, x, y, z, itemType(data +0x54), vtKind, client, cx, cy, cz
Extras (info["s2"], JSON): frame, localIndex, cubes {charId: n}, seen {charId: frames since last client decode},
  jumps [[id, t, T, x1, y1]], mapHash.
Units: world units, 300 per tile; 20 ticks per second. Storm: step = floor(max(0, tick-440)/100), tile (tx,ty) in gas
when min(tx, ty, W-1-tx, H-1-ty) < step (agent NB68Clock.gasStepFromTick / gasTileSimple).
"""
import numpy as np

NIL = -2147483648
MAGIC = 0x4E423253
P2, C2, J2, I2, HDR = 8, 23, 18, 12, 8
P_F = ["index", "team", "charId", "controlledId", "superCharge", "superMax", "hudAlive", "bot"]
C_F = ["id", "dataId", "owner", "x", "y", "z", "hp", "maxHp", "state", "facing", "lastDmgTick", "spawnProt", "f469",
       "ammo", "cooldown", "ultiActive", "lastAttacker", "charType", "vtKind", "client", "cx", "cy", "cz"]
J_F = ["id", "dataId", "owner", "x", "y", "z", "px", "py", "src", "damage", "travel", "travel0", "finished", "impact", "client",
       "cx", "cy", "cz"]
I_F = ["id", "dataId", "owner", "x", "y", "z", "itemType", "vtKind", "client", "cx", "cy", "cz"]
TILE = 300
SNIPER = 16000015
BOX_IDS = {16000120}                 # power-cube box (showdown); a character object
CUBE_ITEM = 18000009
BOMB_BOX_ITEM, BOMB_ITEM = 18000011, 18000035
# tiles.csv rows (data id = 27000000 + row): BlocksMovement+BlocksProjectiles = wall, BlocksMovement only = water/rope,
# forest rows = bush
_WALL_ROWS = [1, 2, 3, 4, 6, 10, 12, 13, 14, 15, 16, 20, 21, 39, 40, 41, 42, 43, 44, 45, 46, 53, 54, 55, 56, 57, 58, 59, 60,
              61, 62, 63, 65, 66, 69, 70, 71, 72, 73]
_WATER_ROWS = [9, 11, 18]
_BUSH_ROWS = [7, 8, 17, 68]
TILE_WALL = {27000000 + r for r in _WALL_ROWS}
TILE_WATER = {27000000 + r for r in _WATER_ROWS}
TILE_BUSH = {27000000 + r for r in _BUSH_ROWS}


def parse(buf):
    a = np.frombuffer(buf, dtype="<i4")
    if a[0] != MAGIC or a[1] != len(a):
        raise ValueError("bad state2 buffer")
    tick, npl, nch, npj, nit = (int(x) for x in a[2:7])
    k = HDR
    pl = a[k:k + npl * P2].reshape(npl, P2); k += npl * P2
    ch = a[k:k + nch * C2].reshape(nch, C2); k += nch * C2
    pj = a[k:k + npj * J2].reshape(npj, J2); k += npj * J2
    it = a[k:k + nit * I2].reshape(nit, I2); k += nit * I2
    return {"tick": tick, "players": pl, "chars": ch, "proj": pj, "items": it}


class MapInfo:
    """Static tile classes from map_state(): wall / water / bush masks (H, W)."""

    def __init__(self, width, height, tiles):
        self.W, self.H = int(width), int(height)
        t = np.array([(-1 if v is None else v) for v in tiles], dtype=np.int64).reshape(self.H, self.W)
        self.tiles = t
        self.wall = np.isin(t, list(TILE_WALL))
        self.water = np.isin(t, list(TILE_WATER))
        self.bush = np.isin(t, list(TILE_BUSH))

    def gas_mask(self, tick):
        step = gas_step(tick)
        ys, xs = np.mgrid[0:self.H, 0:self.W]
        d = np.minimum(np.minimum(xs, ys), np.minimum(self.W - 1 - xs, self.H - 1 - ys))
        return d < step

    def tile_at(self, x, y):
        tx = np.clip(np.floor_divide(np.asarray(x), TILE), 0, self.W - 1)
        ty = np.clip(np.floor_divide(np.asarray(y), TILE), 0, self.H - 1)
        return tx, ty


def gas_step(tick):
    return max(0, int(tick) - 440) // 100


def gas_next_ticks(tick):
    """ticks until the storm advances one ring (100 per ring after tick 440)."""
    t = int(tick) - 440
    return 440 - int(tick) if t < 0 else 100 - (t % 100)


def in_gas(x, y, tick, W, H):
    step = gas_step(tick)
    tx, ty = int(x) // TILE, int(y) // TILE
    return min(tx, ty, W - 1 - tx, H - 1 - ty) < step


class StateView:
    """One parsed dump + extras, with convenience accessors. prev (a StateView) gives velocities by id."""

    def __init__(self, buf, extras, mapinfo, prev=None):
        self.s = parse(buf)
        self.x = extras or {}
        self.m = mapinfo
        self.tick = self.s["tick"]
        ch = self.s["chars"]
        # diversity round: a brawler = the current character of a player (players rows [2] characterId, [3] controlledId);
        # summons (Nita's bear, Jessie's / Penny's turrets) carry their owner's player index too but are not brawlers
        pl = self.s["players"]
        bid = [int(p[3]) if int(p[3]) not in (0, NIL) else int(p[2]) for p in pl]
        self.brawler = (ch[:, 18] == 0) & ~np.isin(ch[:, 1], list(BOX_IDS)) & (ch[:, 2] >= 0) & np.isin(ch[:, 0], bid)
        self.summon = (ch[:, 18] == 0) & ~np.isin(ch[:, 1], list(BOX_IDS)) & (ch[:, 2] >= 0) & ~np.isin(ch[:, 0], bid)
        self.box = (ch[:, 18] == 0) & np.isin(ch[:, 1], list(BOX_IDS))
        self.vel = {}
        if prev is not None and prev.tick < self.tick:
            dt = self.tick - prev.tick
            pmap = {int(r[0]): (int(r[3]), int(r[4])) for r in prev.s["chars"]}
            for r in ch:
                q = pmap.get(int(r[0]))
                if q is not None:
                    self.vel[int(r[0])] = ((int(r[3]) - q[0]) / dt, (int(r[4]) - q[1]) / dt)

    def player_char(self, pidx):
        """row of the brawler controlled by player index pidx (or None)."""
        ch = self.s["chars"]
        rows = np.nonzero((ch[:, 2] == pidx) & self.brawler)[0]
        return None if len(rows) == 0 else ch[rows[0]]

    def alive_players(self):
        ch = self.s["chars"]
        return sorted(int(ch[i, 2]) for i in np.nonzero(self.brawler & (ch[:, 6] > 0))[0])

    def super_frac(self, pidx):
        p = self.s["players"]
        r = p[p[:, 0] == pidx]
        if len(r) == 0 or r[0, 5] <= 0:
            return 0.0
        return float(r[0, 4]) / float(r[0, 5])

    def cubes(self, char_id):
        return int((self.x.get("cubes") or {}).get(str(char_id), 0))

    def jumping(self, char_id):
        for j in self.x.get("jumps") or []:
            if int(j[0]) == int(char_id):
                return j
        return None


# ---------------------------------------------------------------------------------------------------------------
# Egocentric fixed-size encoding (the S1/S2 policy input). All positions relative to the observer, x right / y down,
# scaled by 3000 (Piper's range). Sorted by distance, zero-padded, with a 'present' flag.
K_ENEMY, K_PROJ, K_BOX, K_ITEM = 9, 16, 6, 8
GRID = 25                              # egocentric tile crop 25x25 (+-12 tiles = +-3600 units)
F_SELF, F_ENEMY, F_PROJ, F_BOX, F_ITEM, C_GRID = 22, 16, 12, 5, 6, 6
R = 3000.0


def encode(view, pidx):
    """-> dict(self (F_SELF,), enemy (K_ENEMY, F_ENEMY), proj (K_PROJ, F_PROJ), box (K_BOX, F_BOX), item (K_ITEM, F_ITEM),
    grid (C_GRID, GRID, GRID)) float32, or None if the player has no living brawler."""
    me = view.player_char(pidx)
    if me is None or me[6] <= 0:
        return None
    s = view.s; m = view.m; W, H = m.W, m.H
    mx, my, mid = float(me[3]), float(me[4]), int(me[0])
    tick = view.tick
    vx, vy = view.vel.get(mid, (0.0, 0.0))
    tx, ty = int(mx) // TILE, int(my) // TILE
    jmp = view.jumping(mid)
    alive = view.alive_players()
    f_self = np.array([
        me[6] / max(1, me[7]), me[6] / 10000.0, max(0, me[13]) / 3000.0, (max(0, me[13]) % 1000) / 1000.0,
        max(0, me[14]) / 600.0, view.super_frac(pidx), min(view.cubes(mid), 10) / 10.0, float(me[11] > 0),
        float(m.bush[min(ty, H - 1), min(tx, W - 1)]), float(in_gas(mx, my, tick, W, H)), mx / (W * TILE), my / (H * TILE),
        vx / 40.0, vy / 40.0, max(0, me[5]) / 700.0, float(jmp is not None),
        min(tick, 3000) / 3000.0, len(alive) / 10.0, gas_step(tick) / 30.0, gas_next_ticks(tick) / 440.0,
        (W * TILE / 2 - mx) / (W * TILE / 2), (H * TILE / 2 - my) / (H * TILE / 2)], dtype=np.float32)
    ch = s["chars"]
    en = []
    for i in np.nonzero(view.brawler)[0]:
        r = ch[i]
        if int(r[2]) == pidx or r[6] <= 0:
            continue
        dx, dy = (r[3] - mx) / R, (r[4] - my) / R
        ev = view.vel.get(int(r[0]), (0.0, 0.0))
        etx, ety = int(r[3]) // TILE, int(r[4]) // TILE
        en.append((dx * dx + dy * dy, [1.0, dx, dy, ev[0] / 40.0, ev[1] / 40.0, r[6] / max(1, r[7]), r[6] / 10000.0,
                   max(0, r[13]) / 3000.0, view.super_frac(int(r[2])), min(view.cubes(int(r[0])), 10) / 10.0,
                   float(m.bush[min(ety, H - 1), min(etx, W - 1)]), float(r[19] == 1), float(r[11] > 0),
                   float(view.jumping(int(r[0])) is not None), float(int(r[16]) == pidx), float(int(r[1]) == SNIPER)]))
    f_en = _topk(en, K_ENEMY, F_ENEMY)
    pr = []
    for r in s["proj"]:
        dx, dy = (r[3] - mx) / R, (r[4] - my) / R
        pvx, pvy = float(r[3] - r[6]), float(r[4] - r[7])
        if r[6] == 0 and r[7] == 0:
            pvx = pvy = 0.0
        own = int(r[2]) == pidx
        # closest approach of the projectile's straight path to the observer (units of R) and time to it (ticks)
        rx, ry = mx - r[3], my - r[4]
        vv = pvx * pvx + pvy * pvy
        tca = max(0.0, (rx * pvx + ry * pvy) / vv) if vv > 1e-6 else 0.0
        cx, cy = r[3] + pvx * tca - mx, r[4] + pvy * tca - my
        pr.append((dx * dx + dy * dy, [1.0, dx, dy, pvx / 200.0, pvy / 200.0, float(own), float(not own and r[2] >= 0),
                   r[9] / 4000.0, r[10] / 1000.0, float(r[12] != 0), min(tca, 40) / 40.0, np.hypot(cx, cy) / R]))
    f_pj = _topk(pr, K_PROJ, F_PROJ)
    bx = []
    for i in np.nonzero(view.box)[0]:
        r = ch[i]
        if r[6] <= 0:
            continue
        dx, dy = (r[3] - mx) / R, (r[4] - my) / R
        bx.append((dx * dx + dy * dy, [1.0, dx, dy, r[6] / max(1, r[7]), r[6] / 10000.0]))
    f_bx = _topk(bx, K_BOX, F_BOX)
    itm = []
    for r in s["items"]:
        dx, dy = (r[3] - mx) / R, (r[4] - my) / R
        itm.append((dx * dx + dy * dy, [1.0, dx, dy, float(r[1] == CUBE_ITEM), float(r[1] in (BOMB_ITEM, BOMB_BOX_ITEM)),
                    max(0, r[5]) / 1500.0]))
    f_it = _topk(itm, K_ITEM, F_ITEM)
    # egocentric tile grid: wall, water, bush, gas, out-of-map, enemies (count per tile)
    g = np.zeros((C_GRID, GRID, GRID), np.float32)
    half = GRID // 2
    xs = np.arange(tx - half, tx + half + 1); ys = np.arange(ty - half, ty + half + 1)
    inside = (xs[None, :] >= 0) & (xs[None, :] < W) & (ys[:, None] >= 0) & (ys[:, None] < H)
    cx_ = np.clip(xs, 0, W - 1); cy_ = np.clip(ys, 0, H - 1)
    gas = m.gas_mask(tick)
    g[0] = m.wall[np.ix_(cy_, cx_)] & inside
    g[1] = m.water[np.ix_(cy_, cx_)] & inside
    g[2] = m.bush[np.ix_(cy_, cx_)] & inside
    g[3] = gas[np.ix_(cy_, cx_)] | ~inside
    g[4] = ~inside
    for i in np.nonzero(view.brawler)[0]:
        r = ch[i]
        if int(r[2]) == pidx or r[6] <= 0:
            continue
        gx, gy = int(r[3]) // TILE - tx + half, int(r[4]) // TILE - ty + half
        if 0 <= gx < GRID and 0 <= gy < GRID:
            g[5, gy, gx] += 1.0
    return {"self": f_self, "enemy": f_en, "proj": f_pj, "box": f_bx, "item": f_it, "grid": g}


def _topk(rows, k, f):
    out = np.zeros((k, f), np.float32)
    rows.sort(key=lambda t: t[0])
    for i, (_, v) in enumerate(rows[:k]):
        out[i] = np.asarray(v, np.float32)
    return out


def flat_size():
    return F_SELF + K_ENEMY * F_ENEMY + K_PROJ * F_PROJ + K_BOX * F_BOX + K_ITEM * F_ITEM, (C_GRID, GRID, GRID)


def flatten(enc):
    """-> (vector float32 [flat_size()[0]], grid float32 (C, G, G))"""
    v = np.concatenate([enc["self"], enc["enemy"].ravel(), enc["proj"].ravel(), enc["box"].ravel(), enc["item"].ravel()])
    return v, enc["grid"]
