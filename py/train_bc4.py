#!/usr/bin/env python3
"""BC v4 (pctrain round P1): behavioural cloning of the native AI (or "native + lead") on fair observations of all ten
demonstrators (collect_demos4.py shards). Differences to the learn round's train_bc.py (research F1/D7 fixes):
  * fixed split by battle seed: seed % 10 == 0 -> validation (no leakage across data sets, F1.2);
  * fire weight 1 (no up-weighting of attack/super labels: calibrated fire probability, D7);
  * fair input (nbpc.fair at collection: only what the observer sees; per-observer 'visible', E1/F1.1);
  * the super distance executes the bin centre (maenv.action_to_native, F1.5);
  * pointer aim head, optionally with intercept-angle candidates (--pointer-lead: models.intercept_angles, D2/D5);
  * all data on the GPU (vector float16 + exact metadata; the grid is rebuilt per batch by grid4.GridBuilder).
Writes ../runs/<run>/{config.json, progress.csv, bc_best.pt, bc_last.pt}.

  python3 train_bc4.py --data ../data/demos4_native --run bc4_native --epochs 8
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_rocm_fix  # noqa: F401
from models4 import StatePolicy, NVEC3
from grid4 import GridBuilder


def load_dirs(dirs, max_battles=None):
    V, A, M, E = [], [], [], []; mp = None
    for d in dirs:
        mj = json.load(open(os.path.join(d, "map.json")))
        if mp is None:
            mp = (np.load(os.path.join(d, "map.npy")), mj)
        assert mj["hash"] == mp[1]["hash"], "different maps"
        for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            z = np.load(f); V.append(z["vec"]); A.append(z["act"]); M.append(z["meta"]); E.append(z["ecell"])
    V = np.concatenate(V); A = np.concatenate(A); M = np.concatenate(M); E = np.concatenate(E)
    if max_battles:
        keep = np.isin(M[:, 0], np.unique(M[:, 0])[:max_battles]); V, A, M, E = V[keep], A[keep], M[keep], E[keep]
    return V, A, M, E, mp


def loss_terms(logits, act, bins, sigma):
    """per-sample loss terms (B, 4): move CE, fire CE, aim CE (Gaussian target, firing samples), distance CE (supers)"""
    B = act.shape[0]
    lm = F.cross_entropy(logits[0], act[:, 0], reduction="none")
    lf = F.cross_entropy(logits[1].float(), act[:, 1], reduction="none")
    fire = act[:, 1] > 0
    la = torch.zeros(B, device=act.device); ld = torch.zeros(B, device=act.device)
    if fire.any():
        d = (bins[None] - act[fire, 2][:, None].float()).abs(); d = torch.minimum(d, 360 - d)
        tgt = torch.softmax(-0.5 * (d / sigma) ** 2, -1)
        la[fire] = -(tgt * torch.log_softmax(logits[2][fire].float(), -1)).sum(-1)
    sp = act[:, 1] == 2
    if sp.any():
        ld[sp] = F.cross_entropy(logits[3][sp].float(), act[sp, 3], reduction="none")
    return torch.stack([lm, lf, la, ld], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--aim-sigma", type=float, default=3.0)
    ap.add_argument("--pointer-lead", type=int, default=1); ap.add_argument("--max-battles", type=int, default=None)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", a.run); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda")
    V, A, M, E, (tiles, mj) = load_dirs(a.data, a.max_battles)
    is_val = (M[:, 0] % 10) == 0
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    nb = len(np.unique(M[:, 0])); nvb = len(np.unique(M[is_val, 0]))
    print(f"samples {len(A)} train {len(tr)} val {len(va)} battles {nb} (val {nvb})", flush=True)
    json.dump({"rule": "seed % 10 == 0", "val_battles": nvb, "battles": nb, "train_samples": int(len(tr)), "val_samples": int(len(va))},
              open(os.path.join(out, "split.json"), "w"))
    Vt = torch.from_numpy(V).to(dev); At = torch.from_numpy(A.astype(np.int64)).to(dev)
    Xt = torch.from_numpy(M[:, 3].astype(np.int64)).to(dev); Yt = torch.from_numpy(M[:, 4].astype(np.int64)).to(dev)
    Tt = torch.from_numpy(M[:, 5].astype(np.int64)).to(dev); Et = torch.from_numpy(E).to(dev)
    gb = GridBuilder(tiles, int(mj["W"]), int(mj["H"]), dev)
    net = StatePolicy(436, (6, 25, 25), nvec=NVEC3, pointer_aim=True, pointer_lead=bool(a.pointer_lead)).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_ep = (len(tr) + a.batch - 1) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_ep, pct_start=0.05)
    bins = torch.arange(360, device=dev).float()
    trt = torch.from_numpy(tr).to(dev); vat = torch.from_numpy(va).to(dev)

    def batch(j):
        return Vt[j].float(), gb.build(Xt[j], Yt[j], Tt[j], Et[j]), At[j]

    log = None; best = 1e9
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tl = torch.zeros(4, device=dev); n = 0
        perm = trt[torch.randperm(len(trt), device=dev)]
        for s0 in range(0, len(perm), a.batch):
            j = perm[s0:s0 + a.batch]; v, g, act = batch(j)
            logits, _ = net(v, g)
            L = loss_terms(logits, act, bins, a.aim_sigma); loss = L.sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tl += L.detach().sum(0); n += len(j)
        net.eval(); vl = torch.zeros(4, device=dev); vn = 0
        st = {"mv_ok": 0, "mv_n": 0, "mv22": 0, "mvm": 0, "pf_sum": 0.0, "f_lab": 0, "att_hit": 0, "att_n": 0, "fp": 0, "neg": 0, "ae": []}
        with torch.no_grad():
            for s0 in range(0, len(vat), 8192):
                j = vat[s0:s0 + 8192]; v, g, act = batch(j)
                logits, _ = net(v, g); L = loss_terms(logits, act, bins, a.aim_sigma); vl += L.sum(0); vn += len(j)
                pm = logits[0].argmax(-1); st["mv_ok"] += int((pm == act[:, 0]).sum()); st["mv_n"] += len(j)
                mv = act[:, 0] > 0; d = (pm[mv] - act[mv, 0]).abs(); d = torch.minimum(d, 16 - d)
                st["mv22"] += int(((pm[mv] > 0) & (d <= 1)).sum()); st["mvm"] += int(mv.sum())
                pf = torch.softmax(logits[1].float(), -1); st["pf_sum"] += float(pf[:, 1:].sum()); st["f_lab"] += int((act[:, 1] > 0).sum())
                pa = logits[1].argmax(-1)
                pos = act[:, 1] == 1; st["att_hit"] += int((pa[pos] == 1).sum()); st["att_n"] += int(pos.sum())
                neg = act[:, 1] == 0; st["fp"] += int((pa[neg] > 0).sum()); st["neg"] += int(neg.sum())
                f = act[:, 1] > 0
                if f.any():
                    e = (logits[2][f].argmax(-1) - act[f, 2]).abs(); st["ae"].append(torch.minimum(e, 360 - e).cpu().numpy())
        ae = np.concatenate(st["ae"]) if st["ae"] else np.zeros(1)
        tlv = (tl / n).cpu().numpy(); vlv = (vl / vn).cpu().numpy()
        row = {"epoch": ep, "train_loss": float(tlv.sum()), "val_loss": float(vlv.sum()), "val_move": float(vlv[0]), "val_fire": float(vlv[1]),
               "val_aim": float(vlv[2]), "val_dist": float(vlv[3]), "move_acc": st["mv_ok"] / st["mv_n"], "move_within_22deg": st["mv22"] / max(1, st["mvm"]),
               "p_fire_mean": st["pf_sum"] / vn, "fire_label_rate": st["f_lab"] / vn, "attack_recall": st["att_hit"] / max(1, st["att_n"]),
               "false_fire": st["fp"] / max(1, st["neg"]), "aim_median_err": float(np.median(ae)), "aim_within_5deg": float((ae <= 5).mean()),
               "sec": round(time.time() - t0, 1)}
        if log is None:
            log = csv.DictWriter(open(os.path.join(out, "progress.csv"), "w"), fieldnames=list(row)); log.writeheader()
        log.writerow(row)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        ck = {"model": net.state_dict(), "vec_dim": 436, "grid_shape": [6, 25, 25], "epoch": ep, "metrics": row, "args": vars(a),
              "nvec": NVEC3, "pointer_aim": True, "pointer_lead": bool(a.pointer_lead), "fair": True}
        torch.save(ck, os.path.join(out, "bc_last.pt"))
        if row["val_loss"] < best:
            best = row["val_loss"]; torch.save(ck, os.path.join(out, "bc_best.pt"))
    print("done; best val loss", best, flush=True)


if __name__ == "__main__":
    main()
