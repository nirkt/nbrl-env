"""Sim-vs-labels test (labels only, no models): what the real 960 x 540 frame shows vs the sim's guesses in obs_v5.
  source env.sh; python3 sim_vs_labels.py [N=1500] [DS=~/nb68/datasets_v5]   -> prints, writes hud_ui.npy
Per frame: exact camera from the labels json, player = det 'player' box bottom centre (as the adapter). Tiles are binned
like the sim (obs_v5._cam): rx = tile centre - camera x (screen centre column), ry = tile centre - player y, bin
floor(r / TILE) + G // 2. (a) view window: share of tiles with any of the adapter's 6 base samples on screen, per bin;
(b) HUD: share whose on-screen samples are all seg 'ui' (the adapter then says UNKNOWN); (c) wall bleed of the full
adapter (estimated camera, as test_adapter_v5) per bin / class pair, vs the job's static map."""
import json, os, sys, collections
import numpy as np
from PIL import Image
import adapter_v5 as A
from obs_v5 import SEG, UNKNOWN, G
import test_adapter_v5 as T   # truth_map, JOBS (same argv: N, DS)

N, DS, TILE = T.N if len(sys.argv) > 1 else 1500, T.DS, A.TILE
K = np.arange(G) - G // 2


def frames():
    names = []
    for sp in ("train", "val"):
        ns = sorted(os.listdir(f"{DS}/yolo/seg/{sp}"))
        names += [(sp, n[:-4]) for n in ns[:: max(1, len(ns) // (N // 2))]][: N // 2]
    return names


def main():
    on, ui, cnt = np.zeros((G, G)), np.zeros((G, G)), np.zeros((G, G))
    bl_n, bl_k = np.zeros((G, G)), np.zeros((G, G))
    pair = collections.Counter(); pair_n = collections.Counter()
    zs, ext, camoff, fr_bleed, fr_z, clamp, skipped = [], [], [], [], [], [], 0
    for sp, nm in frames():
        lab = json.load(open(f"{DS}/labelled/labels/{nm}.json"))
        seg = np.array(Image.open(f"{DS}/yolo/seg/{sp}/{nm}.png"))
        t = np.loadtxt(f"{DS}/yolo/labels/{sp}/{nm}.txt", ndmin=2).reshape(-1, 5)
        det = np.c_[t[:, 0], (t[:, 1] - t[:, 3] / 2) * A.W, (t[:, 2] - t[:, 4] / 2) * A.H, (t[:, 1] + t[:, 3] / 2) * A.W, (t[:, 2] + t[:, 4] / 2) * A.H]
        if not (det[:, 0] == 0).any(): skipped += 1; continue
        m = np.array(lab["camera"]); gh = m[[0, 1, 3]][:, [0, 1, 3]] * np.array([1, -1, 1]); ex = (gh, np.linalg.inv(gh))
        z = np.linalg.lstsq(A.CAM[:, :3], m[:, 3] - A.CAM[:, 3], rcond=None)[0][2]; zs.append(z)
        px, py = A.anchor(det, ex)
        camx = A.ground(A.W / 2, A.H / 2, ex[1])[0]; camoff.append(camx - px)
        # frame extents in world units: x half-width at the top / player / bottom row, y of the top / bottom edge vs the player
        vp = A.screen(px, py, gh)[1][0]
        hw = [(np.ptp(A.ground(np.array([0.0, A.W]), np.full(2, v), ex[1])[0])) / 2 for v in (0.0, vp, A.H)]
        ext.append(hw + [A.ground(A.W / 2, 0.0, ex[1])[1] - py, A.ground(A.W / 2, A.H, ex[1])[1] - py])
        # tiles around the player in exact world coordinates
        cx = (np.floor(px / TILE) + K + .5) * TILE; cy = (np.floor(py / TILE) + K + .5) * TILE
        u, v = A.screen(cx[None, :, None] + A.SX, cy[:, None, None] + A.SY, gh)
        ins = (u >= 0) & (u < A.W) & (v >= 0) & (v < A.H)
        raw = seg[np.clip(v, 0, A.H - 1).astype(int), np.clip(u, 0, A.W - 1).astype(int)]
        anyon = ins.any(-1); allui = anyon & ((raw == UNKNOWN) | ~ins).all(-1)
        bx = np.floor((cx - camx) / TILE).astype(int) + G // 2; by = np.floor((cy - py) / TILE).astype(int) + G // 2
        ok = ((bx >= 0) & (bx < G))[None] & ((by >= 0) & (by < G))[:, None]   # tiles whose camera-relative bin is in the grid
        B_ = np.broadcast_to(by[:, None], ok.shape)[ok], np.broadcast_to(bx[None], ok.shape)[ok]
        np.add.at(cnt, B_, 1); np.add.at(on, B_, anyon[ok]); np.add.at(ui, B_, allui[ok])
        # (c) bleed: full adapter (estimated zoom / phase), truth = static map at the exact player tile
        tm = T.truth_map(T.JOBS[nm[:6]])
        if tm is None: continue
        zz, *ph = A.tile_phase(seg); c = A.cam(zz)
        if None in ph: continue
        g = A.grid(seg, A.anchor(det, c), ph, c)
        tx, ty = np.floor(px / TILE).astype(int) + K, np.floor(py / TILE).astype(int) + K
        Hm, Wm = tm.shape
        inb = (ty[:, None] >= 0) & (ty[:, None] < Hm) & (tx[None] >= 0) & (tx[None] < Wm)
        tru = np.where(inb, tm[np.clip(ty, 0, Hm - 1)[:, None], np.clip(tx, 0, Wm - 1)[None]], SEG.index("out_of_bounds"))
        front = np.full_like(tru, UNKNOWN); front[:-1] = tru[1:]
        cand = anyon & (g != UNKNOWN) & ~A.TALL[tru] & A.TALL[front]   # truth not tall, the tile in front tall
        bled = cand & A.TALL[g]
        np.add.at(bl_n, B_, cand[ok]); np.add.at(bl_k, B_, bled[ok]); clamp.append((px, camx, Wm * TILE))
        for f, b, k in zip(front[cand], tru[cand], bled[cand]): pair_n[SEG[f], SEG[b]] += 1; pair[SEG[f], SEG[b]] += k
        if cand.sum() >= 5: fr_bleed.append(bled.sum() / cand.sum()); fr_z.append(z)
    zs, ext, camoff = np.array(zs), np.array(ext), np.array(camoff)
    q = lambda a: " / ".join(f"{np.percentile(a, p):.0f}" for p in (5, 25, 50, 75, 95))
    print(f"frames {len(zs)} (skipped no player box {skipped}); percentiles 5/25/50/75/95:")
    print(f"  zoom z: {q(zs)}; camera x - player x: {q(camoff)} (|.| > 50: {(np.abs(camoff) > 50).mean():.1%})")
    P, C, Wd = np.array(clamp).T   # does the camera follow the player x, clamped at c units from the map's side edges (sim: 3000)?
    err = {c: np.median(np.abs(np.clip(P, c, np.maximum(Wd - c, c)) - C)) for c in range(0, 6001, 100)}
    cb = min(err, key=err.get); e = np.abs(np.clip(P, cb, np.maximum(Wd - cb, cb)) - C)
    print(f"  camera clamp fit: c = {cb} (sim 3000): median error {err[cb]:.0f}, p90 {np.percentile(e, 90):.0f}, > 100 units {(e > 100).mean():.1%}; c 3000 median {err[3000]:.0f}")
    for i, n in enumerate(("half-width top row", "half-width player row", "half-width bottom row", "top edge y - player y", "bottom edge y - player y")):
        print(f"  {n}: {q(ext[:, i])}")
    po = on / np.maximum(cnt, 1)
    rows = np.nonzero((po >= .5).any(1))[0]; cols = np.nonzero((po >= .5).any(0))[0]
    print(f"  tiles seen in >= 50 % of frames: ry bins {rows[0] - G // 2}..{rows[-1] - G // 2}, rx bins {cols[0] - G // 2}..{cols[-1] - G // 2} (x 300 units)")
    print("  P(on screen) by ry bin (rows) x rx bin -12..12, x10:")
    for r in range(G): print(f"   {r - G // 2:+3d} " + "".join(f"{min(9, int(p * 10))}" for p in po[r]))
    pu = ui / np.maximum(on, 1)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hud_ui.npy"), pu.astype(np.float32))
    print("  P(all on-screen samples are ui | on screen), x10 (hud_ui.npy):")
    for r in range(G): print(f"   {r - G // 2:+3d} " + "".join("." if on[r, c_] < 20 else f"{min(9, int(pu[r, c_] * 10))}" for c_ in range(G)))
    print(f"bleed: {bl_k.sum() / max(bl_n.sum(), 1):.1%} of {int(bl_n.sum())}; per frame (>= 5 cand, n {len(fr_bleed)}): {q(np.array(fr_bleed) * 100)} %; corr with zoom {np.corrcoef(fr_bleed, fr_z)[0, 1]:+.2f}")
    pb = bl_k / np.maximum(bl_n, 1)
    print("  by ry bin (all rx): " + ", ".join(f"{r - G // 2:+d} {pb[r].sum() and bl_k[r].sum() / bl_n[r].sum():.0%}" for r in range(G) if bl_n[r].sum() >= 100))
    ax = bl_n.sum(0); print("  by |rx| bin: " + ", ".join(f"{a} {bl_k[:, [G // 2 + a, G // 2 - 1 - a]].sum() / ax[[G // 2 + a, G // 2 - 1 - a]].sum():.0%}" for a in range(G // 2) if ax[[G // 2 + a, G // 2 - 1 - a]].sum() >= 100))
    print("  front / behind: " + ", ".join(f"{f}>{b} {pair[f, b] / n:.0%} n{n}" for (f, b), n in pair_n.most_common(12)))


if __name__ == "__main__":
    main()
