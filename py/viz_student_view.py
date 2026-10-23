#!/usr/bin/env python3
"""What the pixel student sees and understands (held-out frames, data/heldout_px): per example one image with
  row 1: the 4 stacked 256x115 input frames (oldest -> newest; HUD mask applied), as the student receives them (x2);
  row 2: newest frame (x3) | + TRUE labels (rendered from the game state through the camera projection) |
         + what the STUDENT decodes from the pixels alone (its 64x29 maps / heat-map peaks) and its action (aim arrow).
Ground classes tinted: wall red, water blue, bush green, storm purple, off-map grey. Entities (circles): self white,
enemy red, enemy bullet orange, own bullet cyan, box yellow, power cube magenta, bomb black. Runs on the CPU.
  python3 viz_student_view.py --ckpt ../runs/dag4_d/student_last.pt --out ../runs/p2_viz --n 6"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn"))
from student4 import PixelStudent4, MAP_K, ENT_K, SCAL_K, AUX_W, AUX_H, ground_table

GC = {"wall": (255, 40, 40), "water": (40, 110, 255), "bush": (0, 220, 0), "gas": (170, 0, 255), "offmap": (110, 110, 110)}
EC = [(255, 255, 255), (255, 30, 30), (255, 150, 0), (0, 255, 255), (255, 235, 0), (255, 0, 255), (0, 0, 0)]
S = 3                       # display scale of the 256x115 frame
CW, CH = 256 * S / AUX_W, 115 * S / AUX_H   # pixels per aux cell on the display


def overlay(frame, maps, peaks, title, aim=None):
    img = frame.astype(np.float32)
    big = np.array(Image.fromarray(img.astype(np.uint8)).resize((256 * S, 115 * S), Image.NEAREST), np.float32)
    for i, k in enumerate(MAP_K):
        m = np.array(Image.fromarray((maps[i] * 255).astype(np.uint8)).resize((256 * S, 115 * S), Image.NEAREST)) > 127
        big[m] = 0.62 * big[m] + 0.38 * np.array(GC[k])
    im = Image.fromarray(big.clip(0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    for e, pts in peaks.items():
        r = 9 if ENT_K[e] in ("self", "enemy", "box") else 6
        for (y, x) in pts:
            cx, cy = (x + 0.5) * CW, (y + 0.5) * CH
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=EC[e], width=3)
    if aim is not None:
        (y, x), ang, pf = aim
        cx, cy = (x + 0.5) * CW, (y + 0.5) * CH; L = 70
        d.line([cx, cy, cx + L * np.cos(np.radians(ang)), cy + L * np.sin(np.radians(ang))], fill=(255, 255, 255), width=3)
        d.text((cx + 8, cy - 22), f"P(attack) {pf:.2f}", fill=(255, 255, 255))
    d.rectangle([0, 0, 256 * S, 18], fill=(0, 0, 0)); d.text((4, 3), title, fill=(255, 255, 0))
    return im


def peaks_of(heat, thr):
    out = {}
    t = torch.as_tensor(heat)[None]
    mx = F.max_pool2d(t, 3, 1, 1)[0].numpy()
    for e in range(len(ENT_K)):
        ys, xs = np.nonzero((heat[e] == mx[e]) & (heat[e] > thr))
        out[e] = list(zip(ys.tolist(), xs.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "heldout_px", "heldout.npz"))
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--require", default=None, help="only frames with this true entity (e.g. proj_enemy)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    z = np.load(a.data); N = len(z["frames"])
    torch.set_num_threads(2)
    net = PixelStudent4(); net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]); net.eval()
    rng = np.random.default_rng(a.seed)
    heat_all = z["heat"]
    want = ENT_K.index(a.require) if a.require else ENT_K.index("enemy")
    cand = [b for b in range(N) if (heat_all[b, want] > 200).any()]
    pick = sorted(rng.choice(cand, min(a.n, len(cand)), replace=False).tolist())
    rows = []
    for j, b in enumerate(pick):
        fr = z["frames"][b]
        with torch.no_grad():
            o = net(torch.as_tensor(fr[None]))
        aux = o["aux"][0].float(); nm = len(MAP_K)
        pmaps = (torch.sigmoid(aux[:nm]) > 0.5).numpy(); pheat = torch.sigmoid(aux[nm:]).numpy()
        tmaps = z["maps"][b].astype(bool); theat = z["heat"][b].astype(np.float32) / 255.0
        sc = o["scal"][0].float()
        p_hp, p_ammo, p_sup, p_left = float(torch.sigmoid(sc[0])), float(sc[1] * 3), float(torch.sigmoid(sc[2])), float(sc[3] * 10)
        t_hp, t_ammo, t_sup, t_left = float(z["scal"][b][0]), float(z["scal"][b][1] * 3), float(z["scal"][b][2]), float(z["scal"][b][3] * 10)
        lg = o["logits"]; pf = float(torch.softmax(lg[1][0].float(), -1)[1]); ang = int(lg[2][0].argmax())
        sp = peaks_of(pheat, 0.3); selfp = sp[0][0] if sp[0] else (AUX_H // 2, AUX_W // 2)
        # row 1: the 4 input frames
        top = Image.new("RGB", (4 * 512 + 3 * 6, 230), (25, 25, 25))
        for k in range(4):
            top.paste(Image.fromarray(fr[:, :, 3 * k:3 * k + 3]).resize((512, 230), Image.NEAREST), (k * 518, 0))
        dt = ImageDraw.Draw(top)
        for k, lab in enumerate(["t-3 (oldest)", "t-2", "t-1", "t (newest)"]):
            dt.rectangle([k * 518, 0, k * 518 + 120, 16], fill=(0, 0, 0)); dt.text((k * 518 + 4, 2), "input " + lab, fill=(255, 255, 0))
        newest = fr[:, :, 9:12]
        r0 = Image.fromarray(newest).resize((256 * S, 115 * S), Image.NEAREST); d0 = ImageDraw.Draw(r0)
        d0.rectangle([0, 0, 256 * S, 18], fill=(0, 0, 0)); d0.text((4, 3), "newest frame (what the camera shows)", fill=(255, 255, 0))
        r1 = overlay(newest, tmaps, peaks_of(theat, 0.5), f"TRUE labels (from the game state) | HP {t_hp:.2f} ammo {t_ammo:.1f} super {t_sup:.2f} left {t_left:.0f}")
        r2 = overlay(newest, pmaps, sp, f"STUDENT decodes (pixels only) | HP {p_hp:.2f} ammo {p_ammo:.1f} super {p_sup:.2f} left {p_left:.1f}",
                     aim=(selfp, ang, pf))
        W = 3 * 256 * S + 2 * 6; bottom = Image.new("RGB", (W, 115 * S), (25, 25, 25))
        for k, im in enumerate((r0, r1, r2)): bottom.paste(im, (k * (256 * S + 6), 0))
        canvas = Image.new("RGB", (max(W, top.width), 230 + 8 + 115 * S), (25, 25, 25))
        canvas.paste(top, (0, 0)); canvas.paste(bottom, (0, 238))
        path = os.path.join(a.out, f"student_view_{a.require or 'enemy'}_{j:02d}_frame{b}.png"); canvas.save(path)
        rows.append({"frame": int(b), "file": os.path.basename(path), "true": [t_hp, t_ammo, t_sup, t_left], "pred": [p_hp, p_ammo, p_sup, p_left],
                     "p_attack": pf, "aim_deg": ang})
    json.dump({"ckpt": a.ckpt, "examples": rows}, open(os.path.join(a.out, "student_view.json"), "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
