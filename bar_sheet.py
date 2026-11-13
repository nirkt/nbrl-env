"""Mate / enemy HP bar check on live frames (HISTORY 57): det -> enemy_health / teammate_health boxes -> upscaled crop
sheet with the old generic read and the fill-vs-track read printed, plus the column medians (judge by eye).
python bar_sheet.py DET.pt OUT.jpg FRAMES...   (PYTHONPATH=., ~/yolo_env, CUDA_VISIBLE_DEVICES=)"""
import sys, numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import adapter_v5 as A

det_m = YOLO(sys.argv[1]); out = sys.argv[2]; tiles = []
for f in sys.argv[3:]:
    rgb = np.asarray(Image.open(f).convert('RGB').resize((960, 540), Image.BILINEAR))
    r = det_m.predict(rgb[:, :, ::-1], imgsz=960, conf=0.3, device='cpu', verbose=False)[0]
    for c, *b in (np.c_[r.boxes.cls.numpy(), r.boxes.xyxy.numpy()] if len(r.boxes) else []):
        side = {"enemy_health": "enemy", "teammate_health": "mate"}.get(A.DET[int(c)])
        if not side: continue
        x1, y1, x2, y2 = (int(round(t)) for t in b)
        p = rgb[max(y1, 0):y2, max(x1, 0):x2]
        if not p.size: continue
        old, new = A.bar_fill(rgb, [c, *b]), A.bar_fill(rgb, [c, *b], side)
        cm = A.band(p.astype(int), side)
        t = Image.new('RGB', (480, 70)); t.paste(Image.fromarray(p).resize((480, 40), Image.NEAREST))
        d = ImageDraw.Draw(t); d.text((2, 42), '%s %s %s old %.2f new %.2f' % (f.split('/')[-1], side, p.shape[1], old, new), fill=(255, 255, 0))
        d.text((2, 55), ' '.join('%d,%d,%d' % tuple(cm[i]) for i in np.linspace(0, len(cm) - 1, 6).astype(int)), fill=(200, 200, 200))
        tiles.append(t)
print(len(tiles), 'bars')
n = 3; sheet = Image.new('RGB', (n * 480, (len(tiles) + n - 1) // n * 70))
for k, t in enumerate(tiles): sheet.paste(t, (k % n * 480, k // n * 70))
sheet.save(out, quality=90)
