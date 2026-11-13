"""Bot-side adapter: vision output of one 960 x 540 game frame (seg mask + det boxes) -> the ObsV5 arrays (obs_v5.py
from_state2 layout, B = 1). README "Adapter contract".

  A = Adapter(); o = A(seg, det, img=None, own_vel=(0, 0), ticks=2)   # o["self"] (1, 8), o["ent"] (1, 39, 16), o["grid"] (1, 25, 25)
  seg (540, 960) uint8 = seg model class ids (yolo/seg_names.txt, 19 classes incl. ui)
  det (N, 5) = class id (yolo/det_names.txt), x1, y1, x2, y2 in px;  img (540, 960, 3) RGB for the HP / ammo bars

Camera: the 68 client's battle camera only translates (its clip matrix's linear part is the same in every v5 label,
checked on 1149 frames of all jobs): a world move plus a height (zoom) change. The height is estimated per frame (the one
that puts the seg edges on a 300-unit grid), then one homography maps pixels to ground points ("world0" = world shifted
by an unknown per-frame offset). Positions are taken relative to the player (det 'player' box), and the tile grid's phase
(where the 300-unit tile edges lie in world0) is read from the class edges of the seg mask each frame.
Tiles: walls / bushes are labelled by their visible shape (they cover the ground behind them), so each tile is read at
its BASE (its near-camera edge, larger y = lower on screen). A tile hidden behind a tall tile reads as that tall class
(then its far edge is read, which may show above the wall top; behind_unknown=True makes such tiles UNKNOWN instead); rl/test_adapter_v5.py measures both.
shortcut: own velocity comes from the caller (the bot's move command); entity velocity = matched position difference
+ own velocity; HP bars calibrated on GL-captured frames (HISTORY 54), not yet on live-game frames."""
import numpy as np
from obs_v5 import SEG, ENT, UNKNOWN, G, N_SELF, N_ENT, F_ENT, XY, VEL, HP
from nbpc.fair import TILE, VIEW_X, VIEW_UP, VIEW_DOWN, R

W, H = 960, 540
DET = ("player teammate enemy teammate_summon enemy_summon power_box boss objective teammate_projectile enemy_projectile "
       "teammate_area_effect enemy_area_effect gem power_cube item player_health teammate_health enemy_health player_ammo "
       "destructible movement_joystick attack_joystick super_ready super_empty gadget_ready gadget_empty hypercharge_ready "
       "hypercharge_empty").split()
_D = DET.index
SEGN = SEG + ("ui",)                     # seg model ids; ui = 18 = UNKNOWN
SEGMAP = np.arange(len(SEGN), dtype=np.uint8)
SEGMAP[SEGN.index("faded_bush")] = SEGN.index("bush"); SEGMAP[SEGN.index("power_box")] = SEGN.index("floor")
assert SEGN.index("ui") == UNKNOWN
TALL = np.isin(np.arange(len(SEGN)), [SEGN.index(n) for n in
               ("indestructible", "destructible_wall", "fragile_wall", "bush", "faded_bush", "rope_fence", "destructible")])
OOB = SEGN.index("out_of_bounds")

# clip matrix of one v5 frame (labels j00000_f0000730.json 'camera'). Its linear part is the same in every frame; the
# translation = a world move (x, y) + a camera height change z (zoom: z 350-860 vs this frame, median 849, all match long)
CAM = np.array([[4.272611141204834, 0.0, 0.0, -61963.40625], [0.0, 6.1450934410095215, 4.464672088623047, 48084.70703125],
                [0.0, 0.6283222436904907, -0.8648113012313843, 21620.61328125], [0.0, 0.5877853035926819, -0.80901700258255, 22161.216796875]])
ZS = np.arange(300.0, 901.0, 20.0)   # camera heights tried per frame


def cam(z=840.0):
    """(gh, gi): ground homography logic (x, y, 1) -> clip (x, y, w) (labeller/seg.py) at camera height z, and its inverse"""
    m = CAM.copy(); m[:, 3] += CAM[:, 2] * z
    gh = m[[0, 1, 3]][:, [0, 1, 3]] * np.array([1, -1, 1])
    return gh, np.linalg.inv(gh)


CAMS = {z: cam(z) for z in ZS}
SX, SY = np.meshgrid([-0.3, 0.0, 0.3], [0.3, 0.42])        # base samples per tile, tile units from its centre (+y = near camera)
SX, SY = SX.ravel() * TILE, SY.ravel() * TILE
FLY = [_D(n) for n in ("teammate_projectile", "enemy_projectile")]   # in the air: box centre, not its bottom
DET2ENT = {_D(n): ENT.index(n) for n in ENT}


def ground(u, v, gi=CAMS[840.0][1]):
    """pixel coords (u right, v down; pixel centres at +0.5) -> world0 x, y"""
    nx, ny = np.asarray(u, float) / W * 2 - 1, 1 - np.asarray(v, float) / H * 2
    q = np.tensordot(gi, np.stack(np.broadcast_arrays(nx, ny, np.ones_like(nx))), 1)
    return q[0] / q[2], q[1] / q[2]


def screen(x, y, gh=CAMS[840.0][0]):
    c = np.tensordot(gh, np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.ones(1))), 1)
    return (c[0] / c[2] + 1) / 2 * W, (1 - c[1] / c[2]) / 2 * H


def _phase(vals):
    """circular mode (mod TILE) of edge coordinates; None if too few"""
    if len(vals) < 20: return None
    m = np.mod(vals, TILE); h = np.bincount((m // 5).astype(int) % (TILE // 5), minlength=TILE // 5)
    p = (np.argmax(np.convolve(np.r_[h[-2:], h, h[:2]], np.ones(5), "valid")) + .5) * 5   # peak of a 25-unit window
    d = (m - p + TILE / 2) % TILE - TILE / 2
    return float((p + d[np.abs(d) < 30].mean()) % TILE)


def tile_phase(seg):
    """(z, ox, oy): camera height (ZS) under which the seg mask's class edges fall closest to a 300-unit grid, and the
    world0 coordinate of the tile edges mod TILE (None if too few edges)"""
    s, tall = SEGMAP[seg], TALL[seg]
    bad = (s == UNKNOWN) | (seg == OOB)
    bottom = np.vstack([~tall[1:], np.ones((1, W), bool)])        # the pixel below is not tall: a flat pixel or a wall's base row
    ok = ~bad & (~tall | bottom)                                    # higher up a wall's side leans with perspective
    vx, ux = np.nonzero((s[:, 1:] != s[:, :-1]) & ok[:, 1:] & ok[:, :-1])
    vy, uy = np.nonzero((s[1:] != s[:-1]) & ~bad[1:] & ~bad[:-1] & ~tall[1:])   # lower pixel not tall: a base, not a wall top
    if len(ux) < 20 or len(uy) < 20: return 840.0, None, None
    sc = lambda a: np.abs(np.exp(2j * np.pi * a / TILE).mean())   # how tightly the edges sit on one phase
    score = lambda z: sc(ground(ux + 1.0, vx + .5, cam(z)[1])[0]) + sc(ground(uy + .5, vy + 1.0, cam(z)[1])[1])
    best = max(ZS, key=score)
    best = max(np.arange(best - 15, best + 16, 5.0), key=score)   # refine to 5 units
    gi = cam(best)[1]
    return best, _phase(ground(ux + 1.0, vx + .5, gi)[0]), _phase(ground(uy + .5, vy + 1.0, gi)[1])


def anchor(det, c):
    """world0 of the player: bottom centre of the det 'player' box (largest), else the screen point the camera keeps it at"""
    p = det[det[:, 0] == _D("player")] if len(det) else det
    if not len(p): return ground(W / 2, PLAYER_V, c[1])
    b = p[np.argmax((p[:, 3] - p[:, 1]) * (p[:, 4] - p[:, 2]))]
    return ground((b[1] + b[3]) / 2, b[4], c[1])


PLAYER_V = 309.0   # px: median player-box bottom on the v5 val frames (rl/test_adapter_v5.py prints it)


def grid(seg, anc, phase, c, behind_unknown=False):
    """(G, G) uint8 seg class per tile around the player tile, UNKNOWN off screen / under ui / outside the sim window"""
    ox, oy = phase; ax, ay = anc
    k = np.arange(G) - G // 2
    cx = ox + (np.floor((ax - ox) / TILE) + k + .5) * TILE
    cy = oy + (np.floor((ay - oy) / TILE) + k + .5) * TILE
    def vote(sy):
        u, v = screen(cx[None, :, None] + SX, cy[:, None, None] + sy, c[0])
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        raw = np.where(inside, seg[np.clip(v, 0, H - 1).astype(int), np.clip(u, 0, W - 1).astype(int)], UNKNOWN)
        n = (SEGMAP[raw][..., None] == np.arange(len(SEGN))).sum(-2); n[..., UNKNOWN] = 0
        return np.where(n.max(-1) > 0, n.argmax(-1), UNKNOWN).astype(np.uint8)
    g = vote(SY)
    hid = np.zeros_like(g, bool); hid[:-1] = TALL[g[1:]] & (g[:-1] == g[1:])   # base reads the tall tile in front of it
    if behind_unknown: g[hid] = UNKNOWN
    else:   # its far edge may show above that wall's top
        f = vote(-SY); g[hid & (f != UNKNOWN)] = f[hid & (f != UNKNOWN)]
    camx, _ = ground(W / 2, H / 2, c[1])
    seen = (np.abs(cx - camx) < VIEW_X)[None, :] & ((cy - ay > -VIEW_UP) & (cy - ay < VIEW_DOWN))[:, None]
    g[~seen] = UNKNOWN
    return g


# HP truth fit (calib_bars.py, HISTORY 54: 969 GL-captured frames with the state dump, 8253 bars): hp = a * read + b.
# MAE enemy 0.204 -> 0.125, mate 0.127 -> 0.084, own 0.118 -> 0.043; ammo reads right uncorrected (MAE 0.058)
AMMO_FULL = 0.75   # orange share of a full live ammo box (box padding), 70.757 frames (HISTORY 57)
HP_CAL = {"enemy": (0.980, 0.182), "mate": (1.019, 0.074), "own": (1.080, 0.047)}


def bar_fill(img, b, side=None):
    """filled fraction of a bar = share of columns whose median pixel is bright and saturated (fill colour vs the dark
    empty part); side ('own' / 'mate' / 'enemy') = an HP bar: apply its HP_CAL fit"""
    x1, y1, x2, y2 = (int(round(t)) for t in b[1:5])
    p = img[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].astype(int)
    if not p.size: return 0.0
    c = np.median(p, 0)
    if side == "ammo":   # orange columns only: the empty grey-blue segment passes the generic test (HISTORY 57)
        return float(np.clip(((c[:, 0] > 150) & (c[:, 0] - c[:, 2] > 60)).mean() / AMMO_FULL, 0.0, 1.0))
    f = float(((c.max(-1) > 100) & (c.max(-1) - c.min(-1) > 60)).mean())
    if side is None: return f
    a, k = HP_CAL[side]
    return float(np.clip(a * f + k, 0.0, 1.0))


class Adapter:
    def __init__(self, behind_unknown=False):
        self.behind_unknown, self.prev, self.phase = behind_unknown, None, (0.0, 0.0)

    def __call__(self, seg, det, img=None, own_vel=(0.0, 0.0), ticks=2):
        det = np.asarray(det, float).reshape(-1, 5)
        z, *ph = tile_phase(seg)
        self.phase = tuple(p if p is not None else q for p, q in zip(ph, self.phase))   # no edges: keep the last phase
        c = cam(z)
        anc = anchor(det, c)
        g = grid(seg, anc, self.phase, c, self.behind_unknown)
        cls = det[:, 0].astype(int)
        s = np.zeros(N_SELF, np.float32)
        has = lambda n: bool((cls == _D(n)).any())
        if img is not None:
            hb = det[cls == _D("player_health")]
            if len(hb): s[0] = bar_fill(img, hb[0], "own")
            am = det[cls == _D("player_ammo")]
            if len(am): s[1] = np.mean([bar_fill(img, b, "ammo") for b in am])
        s[2], s[3], s[4] = has("super_ready"), has("gadget_ready"), has("hypercharge_ready")
        s[5] = g[G // 2, G // 2] == SEG.index("bush")
        s[6:8] = own_vel
        ent = np.zeros((N_ENT, F_ENT), np.float32)
        m = np.isin(cls, list(DET2ENT))
        e = det[m]
        if len(e):
            fly = np.isin(e[:, 0], FLY)
            gx, gy = ground((e[:, 1] + e[:, 3]) / 2, np.where(fly, (e[:, 2] + e[:, 4]) / 2, e[:, 4]), c[1])
            camx, _ = ground(W / 2, H / 2, c[1])
            keep = (np.abs(gx - camx) < VIEW_X) & (gy - anc[1] > -VIEW_UP) & (gy - anc[1] < VIEW_DOWN)
            e, dx, dy = e[keep], (gx[keep] - anc[0]) / R, (gy[keep] - anc[1]) / R
            o = np.argsort(dx * dx + dy * dy)[:N_ENT]
            e, dx, dy = e[o], dx[o], dy[o]
            n = len(e); ec = np.array([DET2ENT[int(c)] for c in e[:, 0]], int)
            ent[:n, 0] = 1; ent[np.arange(n), 1 + ec] = 1; ent[:n, XY], ent[:n, XY + 1] = dx, dy
            if self.prev is not None and len(self.prev[0]):   # nearest same-class detection of the last call within 1 tile
                pc, px, py = self.prev
                d = np.hypot(dx[:, None] - px[None], dy[:, None] - py[None]) + 1e9 * (ec[:, None] != pc[None])
                j = d.argmin(1); ok = d[np.arange(n), j] < TILE / R; f = R / TILE / ticks   # R units -> tiles per tick
                ent[:n, VEL] = np.where(ok, (dx - px[j]) * f + own_vel[0], 0)
                ent[:n, VEL + 1] = np.where(ok, (dy - py[j]) * f + own_vel[1], 0)
            if img is not None:   # brawler HP: the bar of the same side whose centre is nearest above the box
                for i, c in enumerate(e[:, 0]):
                    bar, side = {_D("enemy"): ("enemy_health", "enemy"), _D("teammate"): ("teammate_health", "mate")}.get(int(c), (None, None))
                    hb = det[cls == _D(bar)] if bar else det[:0]
                    if len(hb):
                        bx, by = (e[i, 1] + e[i, 3]) / 2, e[i, 2]
                        k_ = np.argmin(np.abs((hb[:, 1] + hb[:, 3]) / 2 - bx) + np.abs(hb[:, 4] - by))
                        ent[i, HP] = bar_fill(img, hb[k_], side)
            self.prev = (ec, dx, dy)
        else:
            self.prev = (np.zeros(0, int), np.zeros(0), np.zeros(0))
        return {"self": s[None], "ent": ent[None], "grid": g[None]}
