"""Live-game check of the vision -> adapter chain (HISTORY 57): screenshots of a real match (MuMu, 1920 x 1080) ->
det + seg checkpoints (CPU) -> adapter_v5.Adapter -> per-frame summary + an overlay sheet.
python live_game_check.py FRAMES_DIR DET.pt SEG.pt OUT_DIR   (PYTHONPATH=.:$HOME/nb68/bulletlabels/vendor, ~/yolo_env)"""
import glob, os, sys, collections, numpy as np, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
from ultralytics import YOLO
import adapter_v5 as A
from obs_v5 import SEG, ENT, G

fdir, detp, segp, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
det_m = YOLO(detp)
assert list(det_m.names.values()) == A.DET, det_m.names   # det class ids must be the adapter's
seg_m = smp.FPN('tu-mobilenetv3_large_100', encoder_weights=None, classes=len(A.SEGN))
seg_m.load_state_dict(torch.load(segp, map_location='cpu')); seg_m.eval()
PAL = np.array([[128, 128, 128],[40, 40, 40], [160, 90, 40], [220, 160, 60], [30, 170, 40], [100, 200, 100], [40, 90, 220],
                [200, 200, 200]] + [[np.random.RandomState(i).randint(60, 255) for _ in range(3)] for i in range(8, 30)], np.uint8)
PAL[A.SEGN.index("ui")] = [255, 0, 255]; PAL[A.OOB] = [0, 0, 0]


def seg_of(im):   # same input as train_seg2.py: 480 x 272 RGB / 255, padded to 288 rows
    x = torch.from_numpy(np.asarray(im.resize((480, 272), Image.BILINEAR)).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.no_grad(): y = seg_m(F.pad(x, (0, 0, 0, 16)))[:, :, :272]
    return F.interpolate(y, (540, 960), mode='bilinear').argmax(1)[0].numpy().astype(np.uint8)


ad, rows, tiles = A.Adapter(), [], []
files = sorted(glob.glob(fdir + '/*.png'))
for f in files:
    im = Image.open(f).convert('RGB').resize((960, 540), Image.BILINEAR); rgb = np.asarray(im)
    r = det_m.predict(rgb[:, :, ::-1], imgsz=960, conf=0.3, device='cpu', verbose=False)[0]   # ultralytics wants BGR arrays
    det = np.c_[r.boxes.cls.numpy(), r.boxes.xyxy.numpy()] if len(r.boxes) else np.zeros((0, 5))
    seg = seg_of(im)
    o = ad(seg, det, rgb)
    cls = collections.Counter(A.DET[int(c)] for c in det[:, 0])
    g = o["grid"][0]; ent = o["ent"][0]; ne = int(ent[:, 0].sum())
    rows.append((os.path.basename(f), cls["player"], o["self"][0][0], o["self"][0][1], ne, (g == A.SEGN.index("ui")).mean(),
                 dict(cls)))
    tiles.append(np.bincount(g.ravel(), minlength=len(A.SEGN)))
    vis = Image.fromarray((rgb * 0.55 + PAL[seg] * 0.45).astype(np.uint8)); d = ImageDraw.Draw(vis)
    for c, *b in det: d.rectangle(b, outline=(255, 255, 0) if A.DET[int(c)] == "player" else (0, 255, 255)); d.text((b[0], b[1] - 10), A.DET[int(c)][:10], fill=(255, 255, 255))
    gi = Image.fromarray(PAL[g]).resize((G * 8, G * 8), Image.NEAREST); gd = ImageDraw.Draw(gi)
    for k in range(ne): gd.ellipse([(G / 2 + ent[k, A.XY] * A.R / A.TILE) * 8 - 3, (G / 2 + ent[k, A.XY + 1] * A.R / A.TILE) * 8 - 3,
                                    (G / 2 + ent[k, A.XY] * A.R / A.TILE) * 8 + 3, (G / 2 + ent[k, A.XY + 1] * A.R / A.TILE) * 8 + 3], fill=(255, 0, 0))
    gd.rectangle([G * 4 - 4, G * 4 - 4, G * 4 + 4, G * 4 + 4], outline=(255, 255, 0))
    vis.paste(gi, (960 - G * 8, 0)); d.text((4, 520), 'hp %.2f ammo %.2f ents %d' % rows[-1][2:5], fill=(255, 255, 255))
    vis.save('%s/%s.jpg' % (out, os.path.basename(f)[:-4]), quality=80)

with open(out + '/summary.txt', 'w') as fo:
    for r in rows: fo.write('%s player=%d hp=%.2f ammo=%.2f ents=%d grid_ui=%.2f %s\n' % r)
    t = np.sum(tiles, 0); fo.write('grid tiles: %s\n' % {n: round(v / t.sum(), 3) for n, v in zip(A.SEGN, t) if v})
    fo.write('player found %d / %d frames\n' % (sum(r[1] > 0 for r in rows), len(rows)))
pick = [files[i] for i in np.linspace(0, len(files) - 1, min(12, len(files))).astype(int)]
sheet = Image.new('RGB', (3 * 640, 4 * 360))
for k, f in enumerate(pick):
    sheet.paste(Image.open('%s/%s.jpg' % (out, os.path.basename(f)[:-4])).resize((640, 360)), (k % 3 * 640, k // 3 * 360))
sheet.save(out + '/sheet.jpg', quality=85)
print(open(out + '/summary.txt').read()[-1500:])
