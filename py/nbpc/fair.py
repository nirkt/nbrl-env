"""Fair observation (research E1, F1): only what the observing player could see, for ANY player.

The C encoder (state2.encode) writes the enemy 'visible' feature (vector enemy feature 11) from the LOCAL player's client
flags (sim-only mode: the visibility emulation of nbhost, fitted to the phone, centred on the LOCAL brawler; client mode:
the real client list) — wrong for every other observer (research F1: 89 % of the old BC samples). This module recomputes
visibility per observer with the same emulation rule (nbhost env.c emulate_client_flags, 98.5 % agreement with the phone's
flags): camera x = the observer's x clamped to [3000, W*300 - 3000], camera y = the observer's y; in view iff
|x - camx| < 4500 and -3500 < y - camy < 2900; an enemy brawler on a bush tile is hidden beyond 900 units.

fair_inplace(vec, grid, W, H) (numpy, batched) then
  * enemies: keeps the visible ones (compacted to the front, distance order kept), feature 11 = 1; the rest zeroed;
  * projectiles / boxes / items: keeps the ones in view (compacted);
  * grid channel 5 (enemy count per tile) rebuilt from the visible enemies only.
The static channels (walls, water, bushes, storm, off-map) are kept: the map is fixed and on screen around the player.
Vector layout (state2.flatten): self 22 | enemy 9 x 16 | projectile 16 x 12 | box 6 x 5 | item 8 x 6 (= 436)."""
import numpy as np

R = 3000.0
TILE = 300
EN0, EN_K, EN_F = 22, 9, 16
PJ0, PJ_K, PJ_F = EN0 + EN_K * EN_F, 16, 12
BX0, BX_K, BX_F = PJ0 + PJ_K * PJ_F, 6, 5
IT0, IT_K, IT_F = BX0 + BX_K * BX_F, 8, 6
VIEW_X, VIEW_UP, VIEW_DOWN, BUSH_REVEAL = 4500.0, 3500.0, 2900.0, 900.0


def own_xy(vec, W, H):
    """observer position (world units, exact integers) from self features 10/11 (x / (W*300), y / (H*300))"""
    return np.rint(vec[:, 10].astype(np.float64) * (W * TILE)), np.rint(vec[:, 11].astype(np.float64) * (H * TILE))


def _inview(mx, my, dx, dy, W):
    """dx, dy: relative positions in units of R (B, K); mx, my: (B,)"""
    camx = np.clip(mx, 3000.0, W * TILE - 3000.0)
    ex = mx[:, None] + np.rint(dx.astype(np.float64) * R); ey = my[:, None] + np.rint(dy.astype(np.float64) * R)
    rx, ry = ex - camx[:, None], ey - my[:, None]
    return (np.abs(rx) < VIEW_X) & (ry > -VIEW_UP) & (ry < VIEW_DOWN), ex, ey


def _compact(block, keep):
    """block (B, K, F), keep (B, K) bool -> kept rows moved to the front in order, the rest zero (in place)"""
    B, K, F = block.shape
    order = np.argsort(~keep, axis=1, kind="stable")
    out = np.take_along_axis(block, order[:, :, None], axis=1)
    n = keep.sum(1)
    out[np.arange(K)[None, :] >= n[:, None]] = 0.0
    block[...] = out
    return n


def enemy_visible(vec, W, H):
    """(B, 9) bool: per enemy slot, visible to this observer (present & in view & not hidden in a bush)"""
    mx, my = own_xy(vec, W, H)
    en = vec[:, EN0:EN0 + EN_K * EN_F].reshape(-1, EN_K, EN_F)
    inv, _, _ = _inview(mx, my, en[:, :, 1], en[:, :, 2], W)
    dist = np.hypot(en[:, :, 1].astype(np.float64), en[:, :, 2].astype(np.float64)) * R
    hidden = (en[:, :, 10] > 0.5) & (dist > BUSH_REVEAL)
    return (en[:, :, 0] > 0.5) & inv & ~hidden


def fair_inplace(vec, grid, W=60, H=60):
    """vec (B, 436) float32, grid (B, 6, 25, 25) (float32 or uint8): made fair in place. Returns the visible-enemy count."""
    B = vec.shape[0]
    if B == 0:
        return np.zeros(0, int)
    mx, my = own_xy(vec, W, H)
    en = vec[:, EN0:EN0 + EN_K * EN_F].reshape(B, EN_K, EN_F)
    vis = enemy_visible(vec, W, H)
    # grid channel 5 from the visible enemies (state2.encode: cell = enemy tile - own tile + 12)
    _, ex, ey = _inview(mx, my, en[:, :, 1], en[:, :, 2], W)
    tx, ty = np.floor_divide(mx, TILE), np.floor_divide(my, TILE)
    gx = (np.floor_divide(ex, TILE) - tx[:, None] + 12).astype(np.int64)
    gy = (np.floor_divide(ey, TILE) - ty[:, None] + 12).astype(np.int64)
    ok = vis & (gx >= 0) & (gx < 25) & (gy >= 0) & (gy < 25)
    grid[:, 5] = 0
    bi, ki = np.nonzero(ok)
    np.add.at(grid, (bi, 5, gy[bi, ki], gx[bi, ki]), 1)
    en[:, :, 11] = vis.astype(np.float32)
    n_en = _compact(en, vis)
    for off, K, F in ((PJ0, PJ_K, PJ_F), (BX0, BX_K, BX_F), (IT0, IT_K, IT_F)):
        blk = vec[:, off:off + K * F].reshape(B, K, F)
        inv, _, _ = _inview(mx, my, blk[:, :, 1], blk[:, :, 2], W)
        _compact(blk, (blk[:, :, 0] > 0.5) & inv)
    return n_en


def enemy_cells(vec, W=60, H=60):
    """(B, 9) int16: grid cell gy*25+gx of each present enemy slot that lies in the 25x25 window (-1 otherwise), from the
    (already fair) vector; used to rebuild grid channel 5 (grid4.GridBuilder)"""
    B = vec.shape[0]
    mx, my = own_xy(vec, W, H)
    en = vec[:, EN0:EN0 + EN_K * EN_F].reshape(B, EN_K, EN_F)
    ex = mx[:, None] + np.rint(en[:, :, 1].astype(np.float64) * R); ey = my[:, None] + np.rint(en[:, :, 2].astype(np.float64) * R)
    gx = np.floor_divide(ex, TILE) - np.floor_divide(mx, TILE)[:, None] + 12
    gy = np.floor_divide(ey, TILE) - np.floor_divide(my, TILE)[:, None] + 12
    ok = (en[:, :, 0] > 0.5) & (gx >= 0) & (gx < 25) & (gy >= 0) & (gy < 25)
    return np.where(ok, gy * 25 + gx, -1).astype(np.int16)
