"""Calibrate adapter_v5.bar_fill (HP / ammo bars) against the game's state dump. CPU only, no new capture: the old debug
captures ~/nb68/dbg_n/_captures/j00362, j00476 kept chunk0000.states.npz (hp, maxHp, ammo per brawler each saved tick) and
their labels (~/nb68/dbg_n/labelled) tie each bar box to its owner player.
  source env.sh; python3 calib_bars.py [DIR=~/nb68/dbg_n]"""
import glob, json, os, sys, collections
import numpy as np
from PIL import Image
import adapter_v5 as A
sys.path.insert(0, os.path.expanduser("~/nb68/bulletlabels/vendor"))
import state2

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/nb68/dbg_n")
CF = {n: i for i, n in enumerate(state2.C_F)}


def states(cap):
    m = json.load(open(f"{D}/_captures/{cap}/chunk0000.json"))["meta"]
    z = np.load(f"{D}/_captures/{cap}/chunk0000.states.npz"); s, off = z["states"], z["offsets"]
    return {r["tick"]: s[off[i]:off[i + 1]] for i, r in enumerate(m) if i + 1 < len(off)}


def rows():
    """(kind, team, truth fill, box, img name) per bar"""
    out, S = [], {}
    for f in sorted(glob.glob(f"{D}/labelled/labels/*.json")):
        lab = json.load(open(f)); cap = lab["capture"]
        if not os.path.exists(f"{D}/_captures/{cap}/chunk0000.states.npz"): continue
        if cap not in S: S[cap] = states(cap)
        raw = S[cap].get(lab["tick"]) if lab["tick"] in S[cap] else S[cap].get(lab["tick"] - 1)
        if raw is None: continue
        st = state2.parse(raw); pl, ch = st["players"], st["chars"]
        by = {int(c[CF["id"]]): c for c in ch}
        ammo = collections.defaultdict(list)
        for i in lab["instances"]:
            c = by.get(i.get("object_id"))   # the bar's object_id = its brawler's char id
            if c is None or i["kind"] not in ("health", "ammo"): continue
            if i["kind"] == "health": out.append(("health", i["team"], c[CF["hp"]] / max(c[CF["maxHp"]], 1), i["bbox"], lab["image"]))
            else: ammo[i["owner_player"]].append(i)
        for p, bs in ammo.items():   # one box per ammo slot, left to right; slot k full when ammo >= (k + 1) * 1000
            a = by[bs[0]["object_id"]][CF["ammo"]] / 1000
            for k, i in enumerate(sorted(bs, key=lambda i: i["bbox"][0])):
                out.append(("ammo", i["team"], float(np.clip(a - k, 0, 1)), i["bbox"], lab["image"]))
    return out


def fill_right(img, b, share=0.3):
    """candidate: bars fill from the left, so fill = rightmost column whose pixels are mostly bright + saturated (the
    white HP digits on the bar are unsaturated and do not cut it short), as a share of the box width"""
    x1, y1, x2, y2 = (int(round(t)) for t in b[1:5])
    p = img[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].astype(int)
    if not p.size: return 0.0
    col = ((p.max(-1) > 100) & (p.max(-1) - p.min(-1) > 60)).mean(0) >= share
    return float((np.nonzero(col)[0].max() + 1) / len(col)) if col.any() else 0.0


READ = {"bar_fill": A.bar_fill, "right": fill_right, "ammo_side": lambda im, b: A.bar_fill(im, b, "ammo")}


def main():
    R = rows(); imgs = {}
    res = collections.defaultdict(list)
    for kind, team, t, b, im in R:
        if im not in imgs: imgs[im] = np.array(Image.open(f"{D}/labelled/{im}").convert("RGB"))
        res[kind, team].append((t, *[f(imgs[im], [0, *b]) for f in READ.values()]))
    print(f"bars {len(R)} from {len(imgs)} frames; image size {next(iter(imgs.values())).shape}")
    for k, v in sorted(res.items()):
        t, *ps = np.array(v).T
        print(f"  {k[0]:6s} {k[1]:5s} n {len(t):4d} truth mean {t.mean():.2f} (full {np.mean(t > .99):.0%}): " + "; ".join(
            f"{n} MAE {np.abs(p - t).mean():.3f} corr {np.corrcoef(t, p)[0, 1] if t.std() and p.std() else 0:+.2f} bias {np.mean(p - t):+.2f}" for n, p in zip(READ, ps)))
    return R, imgs, res


if __name__ == "__main__":
    main()
