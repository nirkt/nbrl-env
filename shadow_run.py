"""Offline shadow run (HISTORY 58): the trained policy on recorded live frames vs on the simulator, CPU only.
  sim:    python shadow_run.py sim CKPT OUT.npz [DECISIONS=3000]        (source env.sh: boots 2 host instances, s3 setup)
  live:   python shadow_run.py live CKPT OUT.npz DET.pt SEG.pt FRAMES_DIR [OVERLAY_DIR]   (~/yolo_env + vendor / nirkt paths)
  report: python shadow_run.py report SIM.npz LIVE.npz [LIVE2.npz ...]
npz = self (N, 8), ent (N, 32, F), grid (N, 25, 25), act (N, 4) = move 17 (0 stop, 1 + k = k * 22.5 deg) | fire 3 | aim deg
| distance 8, pmove (N, 17), pfire (N, 3) = the policy's probabilities, name (N,) = frame file (live)"""
import sys, os, glob, numpy as np, torch
import obs_v5 as O
from model import Policy, HID

torch.set_grad_enabled(False)
COND = O.cond("Sniper", np.ones(len(O.SLIDERS)), 1)   # s3 trained Piper ("Sniper") solo; slider_scale 0 (weights unused)


def load(ck):
    net = Policy().eval(); net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"], strict=False)
    return net


def step(net, o, h, cond):
    from models4 import dist_logp_ent
    s = torch.as_tensor(np.concatenate([o["self"], cond], 1), dtype=torch.float32)
    lg, _, h = net(s, torch.as_tensor(o["ent"], dtype=torch.float32), torch.as_tensor(o["grid"]).to(torch.uint8), h)
    _, _, a = dist_logp_ent(lg)
    return a.numpy(), torch.softmax(lg[0], -1).numpy(), torch.softmax(lg[1], -1).numpy(), h


def save(path, rows, **extra):
    np.savez_compressed(path, **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]}, **extra)


def sim(ck, out, n=3000):
    import env_v5 as E
    from nbpc import diversity as D
    net = load(ck); loc = "Survival604"
    tag = {"loc": loc, "chr": ["Sniper"] * 10, "team_size": 1, "groups": [0] * 10}
    fn = lambda rng, i: {"div": D.make_div(location=loc, chr=tag["chr"]), "tag": tag, "end": "agents",
                         "seed": int(rng.integers(10_000_000, 2 ** 31 - 1)), "ctl": list(range(10))}
    E.Env.slider_scale = 0.0
    env = E.Env(2, fn, ticks=2, fair=True, max_ticks=4000, seed=1, tag="shadow%d" % os.getpid())
    ob = O.ObsV5(noise=O.NOISE, seed=1); hid, rows, got = {}, [], 0
    try:
        while got < n:
            obs, ids, tr = env.poll(min_ready=1, max_wait=0.003)
            for aid, _, done, *_ in tr:
                if done: hid.pop(aid, None)
            if not len(ids): continue
            o = ob.build(obs["vec"], obs["grid"], env.last_meta, ids)
            h = torch.as_tensor(np.stack([hid.get(int(x), np.zeros(HID, np.float32)) for x in ids]))
            a, pm, pf, h2 = step(net, o, h, env.cond(ids))
            env.send(ids, a)
            for x, hh in zip(ids, h2.numpy()): hid[int(x)] = hh
            rows.append({"self": o["self"], "ent": o["ent"], "grid": o["grid"], "act": a, "pmove": pm, "pfire": pf}); got += len(ids)
    finally:
        env.close()
    save(out, rows); print("sim: %d decisions -> %s" % (got, out))


def live(ck, out, detp, segp, fdir, ovl=None):
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    import adapter_v5 as A
    net = load(ck); det_m = YOLO(detp)
    seg_m = smp.FPN('tu-mobilenetv3_large_100', encoder_weights=None, classes=len(A.SEGN))
    seg_m.load_state_dict(torch.load(segp, map_location='cpu')); seg_m.eval()
    ad, h, rows = A.Adapter(), torch.zeros(1, HID), []
    if ovl: os.makedirs(ovl, exist_ok=True)
    for f in sorted(glob.glob(fdir + '/*.png')):
        im = Image.open(f).convert('RGB').resize((960, 540), Image.BILINEAR); rgb = np.asarray(im)
        r = det_m.predict(rgb[:, :, ::-1], imgsz=960, conf=0.3, device='cpu', verbose=False)[0]
        det = np.c_[r.boxes.cls.numpy(), r.boxes.xyxy.numpy()] if len(r.boxes) else np.zeros((0, 5))
        x = torch.from_numpy(np.asarray(im.resize((480, 272), Image.BILINEAR)).copy()).permute(2, 0, 1).float()[None] / 255
        seg = F.interpolate(seg_m(F.pad(x, (0, 0, 0, 16)))[:, :, :272], (540, 960), mode='bilinear').argmax(1)[0].numpy().astype(np.uint8)
        o = ad(seg, det, rgb)
        a, pm, pf, h = step(net, o, h, COND[None])
        rows.append({"self": o["self"], "ent": o["ent"], "grid": o["grid"], "act": a, "pmove": pm, "pfire": pf,
                     "name": np.array([os.path.basename(f)]), "player": np.array([(det[:, 0] == A.DET.index("player")).sum()])})
        if ovl:   # move arrow (cyan) + aim ray if firing (red: attack, yellow: super) from the screen centre
            d = ImageDraw.Draw(im); cx, cy = 480, 300; mv, fi, aim = int(a[0, 0]), int(a[0, 1]), float(a[0, 2])
            if mv: t = np.deg2rad((mv - 1) * 22.5); d.line([cx, cy, cx + 80 * np.cos(t), cy + 80 * np.sin(t)], fill=(0, 255, 255), width=4)
            if fi: t = np.deg2rad(aim); d.line([cx, cy, cx + 140 * np.cos(t), cy + 140 * np.sin(t)], fill=[(0,) * 3, (255, 0, 0), (255, 255, 0)][fi], width=2)
            d.text((4, 520), 'move %s p%.2f fire %d p%.2f/%.2f/%.2f aim %d hp %.2f ammo %.2f ents %d' % (
                'stop' if not mv else '%d' % ((mv - 1) * 22.5), pm[0, mv], fi, *pf[0], aim, o["self"][0, 0], o["self"][0, 1], o["ent"][0, :, 0].sum()), fill=(255, 255, 255))
            im.save('%s/%s.jpg' % (ovl, os.path.basename(f)[:-4]), quality=80)
    save(out, rows); print("live: %d frames -> %s" % (len(rows), out))


def report(simp, *livep):
    S = dict(np.load(simp)); L = [dict(np.load(p)) for p in livep]
    def stats(d, tag):
        e, g, s, a = d["ent"], d["grid"], d["self"], d["act"]; pres = e[..., 0] > 0.5; n = len(s)
        cls = {c: (pres & (e[..., 1 + i] > 0.5)).sum(1).mean() for i, c in enumerate(O.ENT)}
        dist = np.hypot(e[..., O.XY], e[..., O.XY + 1])[pres]
        gs = np.bincount(g.ravel().astype(int), minlength=O.UNKNOWN + 1) / g.size
        hp_e = e[..., O.HP][pres & (e[..., 1] > 0.5)]
        mv = np.bincount(a[:, 0], minlength=17) / n; fi = np.bincount(a[:, 1], minlength=3) / n
        print("== %s: %d decisions" % (tag, n))
        print("  ents/frame %.2f; per class: %s" % (pres.sum(1).mean(), {c: round(float(v), 2) for c, v in cls.items() if v > 0.005}))
        print("  ent dist (R units) p10/50/90 %s; enemy HP mean %.2f (n %d)" % (np.round(np.percentile(dist, [10, 50, 90]), 2) if len(dist) else '-', hp_e.mean() if len(hp_e) else -1, len(hp_e)))
        names = list(O.SEG) + ["unknown"]
        from model import _CAND
        foe = pres & ((e[..., 1] > .5) | (e[..., 1 + O.ENT.index("power_box")] > .5)); off = foe.copy(); off[:, _CAND] = False
        print("  enemies / boxes in slots the aim pointer cannot see: %d of %d" % (off.sum(), foe.sum()))
        print("  grid shares: %s" % {names[i]: round(float(v), 3) for i, v in enumerate(gs) if v > 0.004})
        print("  self mean: %s; bush %.2f" % (np.round(s.mean(0), 2).tolist(), s[:, 5].mean()))
        print("  hp p10/50/90 %s ammo p10/50/90 %s" % (np.round(np.percentile(s[:, 0], [10, 50, 90]), 2), np.round(np.percentile(s[:, 1], [10, 50, 90]), 2)))
        print("  move: stop %.2f, dir hist %s; fire none/attack/super %s; mean p(max move) %.2f" % (
            mv[0], np.round(mv[1:], 2).tolist(), np.round(fi, 2).tolist(), d["pmove"].max(1).mean()))
    stats(S, "sim " + simp)
    for p, d in zip(livep, L):
        k = d["self"][:, 0] > 0; stats({x: v[k] for x, v in d.items()}, "live %s, own-HP-bar frames only (%d of %d)" % (p, k.sum(), len(k)))
    for p, d in zip(livep, L):
        print("-- per frame %s" % p)
        for k in range(len(d["act"])):
            a = d["act"][k]; e = d["ent"][k]; ne = int(e[:, 0].sum()); nen = int(((e[:, 0] > .5) & (e[:, 1] > .5)).sum())
            print("  %s pl %d ents %d enemies %d hp %.2f ammo %.2f bush %d | move %s fire %d aim %d" % (
                d["name"][k], d["player"][k], ne, nen, d["self"][k, 0], d["self"][k, 1], d["self"][k, 5],
                'stop' if not a[0] else int((a[0] - 1) * 22.5), a[1], a[2]))


if __name__ == "__main__":
    m, *r = sys.argv[1:]
    {"sim": lambda: sim(r[0], r[1], *[int(x) for x in r[2:]]), "live": lambda: live(*r), "report": lambda: report(*r)}[m]()
