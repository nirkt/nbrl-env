"""Test of adapter_v5 on v5 val frames: ground-truth seg masks + det labels in, grid compared with the job's real tile map.
  source env.sh; python3 test_adapter_v5.py [N=300] [DS=~/nb68/datasets_v5]
Truth: the frame's exact camera (labels json) + the map from maps.csv (job apk) -> tile class around the same player anchor
(det 'player' box). Static map: destroyed walls / storm are not in it (seg poison counts as floor)."""
import csv, json, os, sys, collections
import numpy as np
from PIL import Image
import adapter_v5 as A
from obs_v5 import SEG, UNKNOWN, G, _tile_lut
from nbpc import diversity as D

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
DS = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/nb68/datasets_v5")
JOBS = {j["id"]: j for j in json.load(open(f"{DS}/plan.json"))["jobs"]}
CODE = {r["TileCode"]: i % 1000000 + 1 for _, (i, r) in D.table("tiles.csv").items() if r.get("TileCode")}
LUT = _tile_lut()
_maps = {}


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.reader(f))
    return [dict(zip(r[0], x)) for x in r[2:]]


def truth_map(j):
    """(H, W) seg class of the job's map"""
    csvd = f"{j['apk']}/assets/csv_logic" if os.path.exists(f"{j['apk']}/assets/csv_logic/maps.csv") else os.path.join(D.ASSETS, "csv_logic")
    loc = j["map"] or j["location"]   # job "map" is a locations.csv row name, as in framegen posecap (D.lid / its Map)
    name = next(r["Map"] for r in rows(f"{csvd}/locations.csv") if r[next(iter(r))] == loc)
    if (csvd, name) not in _maps:
        data, cur = [], None
        for r in rows(f"{csvd}/maps.csv"):
            if r["Map"]: cur = r["Map"]
            if cur == name: data.append(r["Data"])
            elif data: break
        _maps[csvd, name] = None if not data else LUT[np.array([[CODE["."] if ch.isdigit() else CODE.get(ch, 0) for ch in row] for row in data])]
    return _maps[csvd, name]


def main():
    names = sorted(os.listdir(f"{DS}/yolo/seg/val"))
    names = [n[:-4] for n in names[:: max(1, len(names) // N)]][:N]
    tall = A.TALL
    stat = {b: collections.Counter() for b in (False, True)}
    perc = {b: collections.defaultdict(collections.Counter) for b in (False, True)}
    pherr, resid, pv, zerr, nent, hp, skipped = [], [], [], [], [], [], 0
    ADP = A.Adapter()
    for nm in names:
        lab = json.load(open(f"{DS}/labelled/labels/{nm}.json"))
        seg = np.array(Image.open(f"{DS}/yolo/seg/val/{nm}.png"))
        t = np.loadtxt(f"{DS}/yolo/labels/val/{nm}.txt", ndmin=2).reshape(-1, 5)
        det = np.c_[t[:, 0], (t[:, 1] - t[:, 3] / 2) * A.W, (t[:, 2] - t[:, 4] / 2) * A.H, (t[:, 1] + t[:, 3] / 2) * A.W, (t[:, 2] + t[:, 4] / 2) * A.H]
        pl = det[det[:, 0] == 0]
        if not len(pl): skipped += 1; continue
        pv.append(pl[0, 4])
        img = np.array(Image.open(next(f"{DS}/yolo/images/val/{nm}{e}" for e in (".jpg", ".png") if os.path.exists(f"{DS}/yolo/images/val/{nm}{e}"))).convert("RGB"))
        o = ADP(seg, det, img)   # whole call: shapes, entities kept in the sim window
        assert o["self"].shape == (1, 8) and o["ent"].shape == (1, 39, 16) and o["grid"].shape == (1, G, G)
        nent.append(o["ent"][0, :, 0].sum()); hp.append(o["self"][0, 0])
        m = np.array(lab["camera"]); gh = m[[0, 1, 3]][:, [0, 1, 3]] * np.array([1, -1, 1]); gi = np.linalg.inv(gh)
        T = truth_map(JOBS[nm[:6]])
        if T is None: skipped += 1; continue   # map name not in maps.csv (some 'other' jobs)
        z, *ph = A.tile_phase(seg); c = A.cam(z); ex = (gh, gi)
        zt = np.linalg.lstsq(A.CAM[:, :3], m[:, 3] - A.CAM[:, 3], rcond=None)[0][2]; zerr.append(abs(z - zt))
        anc_t = A.anchor(det, ex); anc = A.anchor(det, c)   # same player pixel: exact world / adapter world0
        # world0 -> world shift on the ground around the player (one shift if the zoom is right)
        px, py = A.screen(anc_t[0] + np.array([-2400, 2400, 0, 0]), anc_t[1] + np.array([0, 0, -2400, 2400]), gh)
        sx, sy = np.subtract(A.ground(px, py, gi), A.ground(px, py, c[1]))
        resid.append(max(np.ptp(sx), np.ptp(sy)))
        shift = anc_t[0] - anc[0], anc_t[1] - anc[1]
        true_ph = (-shift[0]) % 300, (-shift[1]) % 300
        if None not in ph: pherr.append([abs((p - q + 150) % 300 - 150) for p, q in zip(ph, true_ph)])
        k = np.arange(G) - G // 2
        tx = np.floor(anc_t[0] / 300).astype(int) + k
        ty = np.floor(anc_t[1] / 300).astype(int) + k
        Hm, Wm = T.shape
        inb = (ty[:, None] >= 0) & (ty[:, None] < Hm) & (tx[None] >= 0) & (tx[None] < Wm)
        tru = np.where(inb, T[np.clip(ty, 0, Hm - 1)[:, None], np.clip(tx, 0, Wm - 1)[None]], SEG.index("out_of_bounds"))
        for b in (False, True):
            g = A.grid(seg, anc, ph if None not in ph else (0.0, 0.0), c, b).copy()
            g[g == SEG.index("poison")] = SEG.index("floor")
            vis = A.grid(seg, anc_t, (0.0, 0.0), ex, False) != UNKNOWN   # cells the frame shows at all (exact camera)
            ok = (g == tru)
            st = stat[b]
            st["vis"] += vis.sum(); st["unk"] += (vis & (g == UNKNOWN)).sum(); st["right"] += (vis & ok).sum()
            front_tall = np.zeros_like(vis); front_tall[:-1] = tall[tru[1:]]
            bleed = vis & (g != UNKNOWN) & tall[g] & ~tall[tru] & front_tall
            st["bleed"] += bleed.sum(); st["notall_behind"] += (vis & ~tall[tru] & front_tall).sum()
            for cl in np.unique(tru[vis]):
                q = vis & (tru == cl); perc[b][SEG[cl]]["n"] += q.sum(); perc[b][SEG[cl]]["ok"] += (q & ok).sum()
                perc[b][SEG[cl]]["unk"] += (q & (g == UNKNOWN)).sum()
    pe = np.array(pherr)
    print(f"frames {len(names) - skipped} (skipped, no player box or map: {skipped}); zoom error median {np.median(zerr):.0f}, p90 {np.percentile(zerr, 90):.0f}; shift spread +-8 tiles median {np.median(resid):.0f}, p90 {np.percentile(resid, 90):.0f} units")
    print(f"Adapter(): entities per frame mean {np.mean(nent):.1f}, own HP bar read (uncalibrated) median {np.median(hp):.2f}")
    print(f"player box bottom v: median {np.median(pv):.1f} px, p10-p90 {np.percentile(pv, 10):.0f}-{np.percentile(pv, 90):.0f}")
    print(f"tile phase error x/y: median {np.median(pe[:, 0]):.1f}/{np.median(pe[:, 1]):.1f}, p95 {np.percentile(pe[:, 0], 95):.0f}/{np.percentile(pe[:, 1], 95):.0f}, >75 units {(pe > 75).any(1).mean():.1%} of frames")
    for b in (False, True):
        c = stat[b]; known = c["vis"] - c["unk"]
        print(f"behind_unknown={b}: right {c['right'] / c['vis']:.1%} of visible tiles ({c['right'] / known:.1%} of known), "
              f"unknown {c['unk'] / c['vis']:.1%}; wall bleed (behind a tall tile, truth not tall, read tall) "
              f"{c['bleed'] / max(c['notall_behind'], 1):.1%} of {c['notall_behind']}")
        print("  " + ", ".join(f"{k} {v['ok'] / v['n']:.0%}/{v['unk'] / v['n']:.0%}u n{v['n']}" for k, v in sorted(perc[b].items(), key=lambda kv: -kv[1]['n'])))


if __name__ == "__main__":
    main()
