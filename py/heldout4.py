#!/usr/bin/env python3
"""P2 held-out decoding set + metrics.

collect: client mode, player 0 driven by its native AI (native_ctl: observed, not controlled), seeds 40,000,000.. (never
  used for training or eval), every --every-th step: the 4-frame stack, the labels (dagger4.render_labels: 5 ground maps,
  7 entity heat maps, 5 scalars), the fair state vector -> ../data/heldout_px/heldout.npz
metrics: a student checkpoint on that set, pixels only: ground-map IoU; entity detection (true peaks matched to the nearest
  predicted peak > 0.3 within 4 observation px = one 64x29 cell; median error), enemies split into in-bush / at the screen
  edge / other; projectiles; HUD scalars vs trivial baselines; example images (frame | true maps | predicted maps).
  python3 heldout4.py collect --episodes 12 --out ../data/heldout_px
  python3 heldout4.py metrics --ckpt ../runs/dag4_a/student_last.pt --data ../data/heldout_px --out ../runs/dag4_a/heldout
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn"))


# the Showdown map (fixed: the same hash 3654183511 in all 1,000 demo battles; data/map_showdown.json)
import numpy as _np, os as _os
MAP_TILES = _np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "map_showdown.npy"))

def collect(a):
    import proj, state2 as S2
    from nbpc.maenv import MAEnv
    from dagger4 import render_labels
    P = proj.load_params(); os.makedirs(a.out, exist_ok=True)
    queue = list(range(40_000_000, 40_000_000 + a.episodes))
    bf = lambda rng, i: ({"seed": queue.pop(0), "ctl": [0], "end": "player0", "tag": "ho", "native_ctl": True} if queue else None)
    env = MAEnv(min(a.n, a.episodes), bf, pixels=True, tag=f"ho{os.getpid()}", verbose=True)
    M = S2.MapInfo(60, 60, MAP_TILES.tolist())
    F, MP, HT, SC, V, META, cnt = [], [], [], [], [], [], {}
    while True:
        obs, ids, tr = env.poll(min_ready=1, max_wait=0.01)
        if len(ids):
            for b, aid in enumerate(ids):
                k = cnt.get(int(aid), 0); cnt[int(aid)] = k + 1
                if k % a.every: continue
                sb, cam, li, p = env.last_state[b]
                if cam is None: continue
                maps, heat, scal, n_en = render_labels(sb, cam, p, M, P)
                F.append(obs["pixels"][b]); MP.append(maps); HT.append(heat); SC.append(scal); V.append(obs["vec"][b].astype(np.float16))
                META.append([int(aid), k, *env.last_meta[b, :3].tolist()])
            env.send(ids, np.zeros((len(ids), 4), np.int64))
        if all(s.state == "idle" for s in env.I): break
    env.close()
    np.savez(os.path.join(a.out, "heldout.npz"), frames=np.stack(F), maps=np.stack(MP), heat=np.stack(HT), scal=np.stack(SC), vec=np.stack(V),
             meta=np.array(META, np.int64))
    print("saved", len(F), "frames")


def metrics(a):
    import torch, torch_rocm_fix  # noqa: F401
    import torch.nn.functional as Fn
    from student4 import PixelStudent4, MAP_K, ENT_K, AUX_W, AUX_H
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(a.data, "heldout.npz")); dev = torch.device(a.device)
    if a.device == "cpu": torch.set_num_threads(3)
    net = PixelStudent4().to(dev).eval(); net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    N = len(z["frames"]); A, S = [], []
    with torch.no_grad():
        for s0 in range(0, N, 64):
            o = net(torch.as_tensor(z["frames"][s0:s0 + 64], device=dev))
            A.append(o["aux"].float().cpu()); S.append(o["scal"].float().cpu())
    aux = torch.cat(A); sc = torch.cat(S); nm = len(MAP_K)
    maps = torch.as_tensor(z["maps"]).bool(); heat = torch.as_tensor(z["heat"]).float() / 255.0; scal = torch.as_tensor(z["scal"])
    res = {"frames": N, "iou": {}, "ent": {}, "scal": {}}
    pm = torch.sigmoid(aux[:, :nm]) > 0.5
    for i, k in enumerate(MAP_K):
        inter = (pm[:, i] & maps[:, i]).sum().item(); uni = (pm[:, i] | maps[:, i]).sum().item()
        res["iou"][k] = round(inter / uni, 4) if uni else None
    ph = torch.sigmoid(aux[:, nm:])
    tpk = (heat == Fn.max_pool2d(heat, 3, 1, 1)) & (heat > 0.5)
    ppk = (ph == Fn.max_pool2d(ph, 3, 1, 1)) & (ph > 0.3)
    bush = maps[:, MAP_K.index("bush")]
    for i, k in enumerate(ENT_K):
        bi, yi, xi = torch.nonzero(tpk[:, i], as_tuple=True)
        if len(bi) == 0: res["ent"][k] = None; continue
        pb, py, px = torch.nonzero(ppk[:, i], as_tuple=True)
        d = []
        for b, y, x in zip(bi.tolist(), yi.tolist(), xi.tolist()):
            m = pb == b
            dd = float(torch.sqrt(((py[m] - y) ** 2 + (px[m] - x) ** 2).float()).min()) * 4.0 if m.any() else float("inf")
            d.append((dd, bool(bush[b, y, x]), x <= 8 or x >= AUX_W - 9 or y <= 1 or y >= AUX_H - 2))
        arr = np.array([t[0] for t in d]); fin = np.isfinite(arr)
        n_pred = int(ppk[:, i].sum()); n_true = len(d)
        row = {"n": n_true, "found_4px": round(float((arr <= 4).mean()), 4), "median_err_px": round(float(np.median(arr[fin])), 2) if fin.any() else None,
               "pred_peaks_per_true": round(n_pred / max(1, n_true), 3)}
        if k == "enemy":
            for name, sel in (("in_bush", [t[1] for t in d]), ("edge", [t[2] for t in d]), ("open_center", [not t[1] and not t[2] for t in d])):
                sel = np.array(sel)
                if sel.any(): row[name] = {"n": int(sel.sum()), "found_4px": round(float((arr[sel] <= 4).mean()), 4)}
        res["ent"][k] = row
    hp = torch.sigmoid(sc[:, 0]); sup = torch.sigmoid(sc[:, 2])
    res["scal"] = {"hp_frac_mae": round(float((hp - scal[:, 0]).abs().mean()), 4), "hp_frac_const_mae": round(float((scal[:, 0] - scal[:, 0].median()).abs().mean()), 4),
                   "ammo_mae_shots": round(float(((sc[:, 1] - scal[:, 1]).abs() * 3).mean()), 4), "ammo_const_mae": round(float(((scal[:, 1] - scal[:, 1].median()).abs() * 3).mean()), 4),
                   "super_frac_mae": round(float((sup - scal[:, 2]).abs().mean()), 4), "super_const_mae": round(float((scal[:, 2] - scal[:, 2].median()).abs().mean()), 4),
                   "brawlers_left_acc": round(float(((sc[:, 3] * 10).round() == (scal[:, 3] * 10).round()).float().mean()), 4),
                   "brawlers_left_majority": round(float(((scal[:, 3] * 10).round() == (scal[:, 3] * 10).round().mode().values).float().mean()), 4),
                   "in_gas_acc": round(float(((sc[:, 4] > 0) == (scal[:, 4] > 0.5)).float().mean()), 4), "in_gas_base": round(float((scal[:, 4] > 0.5).float().mean()), 4)}
    json.dump(res, open(os.path.join(a.out, "heldout_metrics.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    # example images: frame | true (maps tinted + heat) | predicted
    cols = {"wall": (255, 0, 0), "water": (0, 80, 255), "bush": (0, 200, 0), "gas": (160, 0, 255), "offmap": (90, 90, 90)}
    ecol = [(255, 255, 255), (255, 0, 0), (255, 140, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0)]
    def render(mp, ht):
        img = np.zeros((AUX_H, AUX_W, 3), np.float32)
        for i, k in enumerate(MAP_K): img[mp[i]] = 0.5 * img[mp[i]] + 0.5 * np.array(cols[k])
        for i in range(len(ENT_K)):
            h = ht[i][..., None]; img = img * (1 - h) + h * np.array(ecol[i])
        return Image.fromarray(img.clip(0, 255).astype(np.uint8)).resize((512, 232), Image.NEAREST)
    rng = np.random.default_rng(0)
    enemy_frames = [b for b in range(N) if heat[b, 1].max() > 0.5]
    pick = list(rng.choice(enemy_frames, min(6, len(enemy_frames)), replace=False)) if enemy_frames else list(range(min(6, N)))
    for j, b in enumerate(pick):
        fr = Image.fromarray(z["frames"][b][:, :, -3:]).resize((512, 230), Image.NEAREST)
        t = render(z["maps"][b].astype(bool), z["heat"][b] / 255.0)
        p = render((torch.sigmoid(aux[b, :nm]) > 0.5).numpy(), ph[b].numpy())
        out = Image.new("RGB", (512 * 3 + 16, 232)); out.paste(fr, (0, 0)); out.paste(t, (520, 0)); out.paste(p, (1040, 0))
        out.save(os.path.join(a.out, f"viz_{j:02d}_frame{b}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "metrics"])
    ap.add_argument("--episodes", type=int, default=12); ap.add_argument("--every", type=int, default=3); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt"); ap.add_argument("--data"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else metrics(a)
