"""Measure the vision noise the policy will see: trained det + seg models vs the labels on v5 val frames, both sides
turned into ObsV5 by adapter_v5 (same anchor / zoom / tile rules), so the numbers are in obs_v5.NOISE units.

  (WSL) . ~/yolo_env/bin/activate; cd rl; PYTHONPATH=.:$HOME/nb68/bulletlabels/vendor \
        python3 measure_noise.py --det DET.pt --seg SEG.pt [--n 1000] [--device cpu|0] [--conf 0.25]

  jitter  box centre error of matched entities (same class, nearest within 1 tile), sigma per axis, 10-tile units
  drop    truth entities with no match / truth entities
  fp      unmatched predictions / empty entity slots (39 - truth count)
  seg     tiles known in both with a different class / tiles known in both (plus known <-> UNKNOWN flips)
latency is not measurable offline. Frames without a player box in the labels are skipped (adapter anchor)."""
import argparse, glob, os, random, sys, collections
import numpy as np
from PIL import Image

DS = os.path.expanduser("~/nb68/datasets_v5/yolo")
SW, SH, SHP = 480, 272, 288   # seg model input (train_seg2.py): 480 x 272 padded to 288


def load_seg(path, device="cpu"):
    import torch, segmentation_models_pytorch as smp
    net = smp.FPN("tu-mobilenetv3_large_100", encoder_weights=None, classes=19)
    net.load_state_dict(torch.load(path, map_location="cpu")); return net.to(device).eval()


def seg_predict(net, imgs, device="cpu"):
    """list of (540, 960, 3) RGB uint8 -> list of (540, 960) uint8 class ids (the model's 480 x 272 output, nearest up)"""
    import torch, torch.nn.functional as F
    x = torch.stack([torch.from_numpy(np.asarray(Image.fromarray(i).resize((SW, SH), Image.BILINEAR)).copy()) for i in imgs])
    x = F.pad(x.permute(0, 3, 1, 2).float() / 255, (0, 0, 0, SHP - SH)).to(device)
    with torch.no_grad():
        p = net(x).argmax(1)[:, :SH].byte().cpu().numpy()
    return [np.asarray(Image.fromarray(q).resize((960, 540), Image.NEAREST)) for q in p]


def gt_det(nm, split="val"):
    t = np.loadtxt(f"{DS}/labels/{split}/{nm}.txt", ndmin=2).reshape(-1, 5)
    return np.c_[t[:, 0], (t[:, 1] - t[:, 3] / 2) * 960, (t[:, 2] - t[:, 4] / 2) * 540, (t[:, 1] + t[:, 3] / 2) * 960, (t[:, 2] + t[:, 4] / 2) * 540]


def val_names(n, seed=0):
    f = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DS}/seg/val/*.png"))
    random.Random(seed).shuffle(f); return f[:n] if n else f


def image(nm, split="val"):
    p = next(f"{DS}/images/{split}/{nm}{e}" for e in (".jpg", ".png") if os.path.exists(f"{DS}/images/{split}/{nm}{e}"))
    return np.asarray(Image.open(p).convert("RGB"))


def compare(ot, op, acc):
    """one frame: truth obs ot vs predicted obs op (adapter dicts) -> counters in acc"""
    from obs_v5 import UNKNOWN, ENT, XY, N_ENT
    gt, gp = ot["grid"][0], op["grid"][0]
    kn = (gt != UNKNOWN) & (gp != UNKNOWN)
    acc["tiles"] += kn.sum(); acc["tiles_wrong"] += (kn & (gt != gp)).sum(); acc["tiles_flip"] += ((gt != UNKNOWN) != (gp != UNKNOWN)).sum()
    et, ep = ot["ent"][0], op["ent"][0]
    ct, cp = et[:, 1:1 + len(ENT)].argmax(1), ep[:, 1:1 + len(ENT)].argmax(1)
    vt, vp = et[:, 0] > 0, ep[:, 0] > 0
    acc["slots_empty"] += N_ENT - vt.sum()
    used = np.zeros(N_ENT, bool)
    for i in np.nonzero(vt)[0]:
        d = np.hypot(*(ep[:, XY:XY + 2] - et[i, XY:XY + 2]).T) + 1e9 * (~vp | used | (cp != ct[i]))
        j = d.argmin(); acc["n", ENT[ct[i]]] += 1
        if d[j] < 0.1:   # 1 tile in 10-tile units
            used[j] = True; acc.setdefault("err", []).append(ep[j, XY:XY + 2] - et[i, XY:XY + 2])
            acc.setdefault(("errc", ENT[ct[i]]), []).append(d[j])
        else: acc["drop", ENT[ct[i]]] += 1
    for j in np.nonzero(vp & ~used)[0]: acc["fp", ENT[cp[j]]] += 1


def report(acc, frames):
    from obs_v5 import ENT
    e = np.array(acc.get("err", [])).reshape(-1, 2)
    nt = sum(acc["n", c] for c in ENT); nd = sum(acc["drop", c] for c in ENT); nf = sum(acc["fp", c] for c in ENT)
    jit = float(e.std(0).mean()) if len(e) else float("nan")
    m = {"jitter": jit, "drop": nd / max(nt, 1), "fp": nf / max(acc["slots_empty"], 1), "seg": acc["tiles_wrong"] / max(acc["tiles"], 1)}
    print(f"frames {frames}; entities {nt}, matched {len(e)}; tiles known in both {acc['tiles']}, known<->UNKNOWN flips {acc['tiles_flip'] / max(acc['tiles'], 1):.2%}")
    print("measured: " + ", ".join(f"{k} {v:.4f}" for k, v in m.items()) + "   (jitter = sigma per axis, 10-tile units)")
    for c in ENT:
        if acc["n", c] or acc["fp", c]:
            ec = acc.get(("errc", c), [])
            print(f"  {c:22s} n {acc['n', c]:5d}  drop {acc['drop', c] / max(acc['n', c], 1):.1%}  fp {acc['fp', c]:4d}  "
                  f"median err {np.median(ec) * 10 if ec else float('nan'):.2f} tiles")
    print("suggested obs_v5.NOISE (0 .. 2 x measured): " + ", ".join(f'"{k}": (0.0, {2 * v:.3f})' for k, v in m.items()))
    return m


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--det", required=True); a.add_argument("--seg", required=True)
    a.add_argument("--n", type=int, default=1000); a.add_argument("--device", default="cpu"); a.add_argument("--conf", type=float, default=0.25)
    a = a.parse_args()
    if a.device == "cpu": os.environ["HIP_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = ""   # never touch the GPU
    import adapter_v5 as A
    from ultralytics import YOLO
    dev = "cpu" if a.device == "cpu" else f"cuda:{a.device}"
    det, seg = YOLO(a.det), load_seg(a.seg, dev)
    acc = collections.defaultdict(int)
    names = [n for n in val_names(0) if (gt_det(n)[:, 0] == 0).any()][:a.n]
    for b in range(0, len(names), 16):
        nb = names[b:b + 16]; imgs = [image(n) for n in nb]
        sp = seg_predict(seg, imgs, dev)
        rs = det.predict([i[..., ::-1] for i in imgs], imgsz=(544, 960), conf=a.conf, device=a.device, verbose=False)   # numpy = BGR
        for n, img, s, r in zip(nb, imgs, sp, rs):
            pd = np.c_[r.boxes.cls.cpu().numpy(), r.boxes.xyxy.cpu().numpy()]
            st = np.array(Image.open(f"{DS}/seg/val/{n}.png"))
            compare(A.Adapter()(st, gt_det(n)), A.Adapter()(s, pd), acc)
    report(acc, len(names))


if __name__ == "__main__":
    main()
