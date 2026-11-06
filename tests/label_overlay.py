"""P2 label check on PC frames: one client-mode instance (real client + nbgl), all-native battle (the local player's AI
plays), at selected steps the 256x115 observation (the phone's preprocess) with the labels rendered from the state through
the camera projection (learn/labels.py: client positions, client flags): ground maps tinted (wall red, water blue, bush
green, storm purple, off-map grey) and entity markers (self white, enemy red, enemy projectile orange, own projectile
cyan, box yellow, cube magenta, bomb black). Writes a sheet of x3-scaled panels (frame | frame + labels) per step.
  python3 tests/label_overlay.py --out ../runs/p2_labels"""
import os, sys, json, argparse
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py"))
from nbpc.instance import Instance
from nbpc import proto as P
import state2 as S2, proj, labels as LB

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=31337)
ap.add_argument("--steps", default="5,40,80,120,160,200,240,280"); ap.add_argument("--ai-local", type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
want = [int(x) for x in a.steps.split(",")]
inst = Instance(0, tag=f"lov{os.getpid()}", gl=True); inst.start(); inst.wait_ready()
Pp = proj.load_params()
r = inst.reset(seed=a.seed, native_mask=0xffffffff, flags=P.F_STATE | P.F_PIXELS, pix=(256, 115), pix_obs=True, intro_ai=True)
M = S2.MapInfo(r.map_w, r.map_h, inst.map_tiles().tolist())
COL = {"wall": (255, 0, 0), "water": (0, 80, 255), "bush": (0, 255, 0), "gas": (160, 0, 255), "offmap": (128, 128, 128)}
ECOL = {"self": (255, 255, 255), "enemy": (255, 0, 0), "proj_enemy": (255, 140, 0), "proj_own": (0, 255, 255), "box": (255, 255, 0),
        "cube": (255, 0, 255), "bomb": (0, 0, 0)}
rows = []; panels = []
for k in range(max(want) + 1):
    if k in want:
        f = inst.pixels_obs().copy(); v = S2.StateView(inst.state_bytes(), {}, M, None); cam = (r.cam_x, r.cam_y)
        ml, hm, pts, sc = LB.render_all(v, r.local_index, cam, 256, 115, Pp, sigma=1.5)
        o = f.astype(np.float32).copy()
        for kk, c in COL.items():
            m = ml[kk]; o[m] = 0.6 * o[m] + 0.4 * np.array(c)
        im0 = Image.fromarray(f).resize((768, 345), Image.NEAREST); im1 = Image.fromarray(o.astype(np.uint8)).resize((768, 345), Image.NEAREST)
        d = ImageDraw.Draw(im1)
        for kk, lst in pts.items():
            for (u, vv, _) in lst:
                x, y = u * 3, vv * 3; rr = 5 if kk in ("self", "enemy", "box") else 3
                d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=ECOL[kk], width=2)
        d.text((4, 4), f"step {k} tick {r.tick} hp {sc['hp_frac']:.2f} ammo {sc['ammo']:.1f} left {sc['brawlers_left']}", fill=(255, 255, 255))
        pan = Image.new("RGB", (768 * 2 + 8, 345)); pan.paste(im0, (0, 0)); pan.paste(im1, (776, 0)); pan.save(os.path.join(a.out, f"overlay_step{k:03d}.png"))
        panels.append(pan)
        rows.append({"step": k, "tick": r.tick, "cam": cam, "n": {kk: len(lst) for kk, lst in pts.items()}, "scal": sc})
    r = inst.step(actions=None, ctl_mask=0, native_mask=0xffffffff, flags=P.F_STATE | P.F_PIXELS, pix=(256, 115), pix_obs=True)
    if r.done: break
inst.close()
sheet = Image.new("RGB", (panels[0].width, sum(p.height + 6 for p in panels)))
y = 0
for p in panels: sheet.paste(p, (0, y)); y += p.height + 6
sheet.save(os.path.join(a.out, "overlay_sheet.png"))
json.dump(rows, open(os.path.join(a.out, "overlay.json"), "w"), indent=1)
print(json.dumps(rows[:3]))
