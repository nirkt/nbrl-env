#!/usr/bin/env python3
"""S2 behavioural cloning of the native bot AI on the privileged state (s1_build.py samples).

  python3 work/train_bc.py --data work/datasets/s1b_bc.npz --run bc1 --epochs 20
Train / held-out split by episode seed (--val-frac of the seeds). Loss: sum of per-head cross-entropies (gadget head is
constant 0 in the data and is trained too so the policy never presses it). Optional up-weighting of the rare non-none
attack/super labels (--fire-weight). Writes work/runs/<run>/{config.json, progress.csv, bc_best.pt, bc_last.pt}.
"""
import argparse, csv, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_rocm_fix  # noqa: F401  (HIP device count on this host)
from models import StatePolicy, NVEC, is_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--run", default="bc1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--fire-weight", type=float, default=2.0, help="loss weight of non-none attack/super labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--pointer-aim", action="store_true", help="v3: aim head = mixture over enemy/box angles + free component (models.PointerAim)")
    ap.add_argument("--aim-sigma", type=float, default=3.0, help="v3: Gaussian label smoothing of the aim head (degrees)")
    ap.add_argument("--data-weight", type=int, nargs="*", default=None, help="per --data file: repeat its train samples this many times")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", a.run); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    V, G, A, M, SRC = [], [], [], [], []; nvec = None
    for fi, f in enumerate(a.data):
        d = np.load(f); V.append(d["vec"]); G.append(d["grid"]); A.append(d["act"]); M.append(d["meta"]); SRC.append(np.full(len(d["act"]), fi))
        nv = d["nvec"].tolist() if "nvec" in d.files else list(NVEC)
        assert nvec is None or nv == nvec, "mixed action spaces"; nvec = nv
    V = np.concatenate(V); G = np.concatenate(G); A = np.concatenate(A).astype(np.int64); M = np.concatenate(M); SRC = np.concatenate(SRC)
    v3 = is_v3(nvec)
    seeds = np.unique(M[:, 0]); rng = np.random.default_rng(a.seed)
    val_seeds = set(rng.choice(seeds, max(1, int(len(seeds) * a.val_frac)), replace=False).tolist())
    is_val = np.isin(M[:, 0], list(val_seeds))
    tr, va = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
    if a.data_weight:
        tr = np.concatenate([np.repeat(tr[SRC[tr] == fi], w) for fi, w in enumerate(a.data_weight)])
    print(f"samples {len(A)} train {len(tr)} val {len(va)} episodes {len(seeds)} (val {len(val_seeds)})", flush=True)
    json.dump({"val_seeds": sorted(int(s) for s in val_seeds)}, open(os.path.join(out, "split.json"), "w"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = StatePolicy(V.shape[1], G.shape[1:], nvec=nvec, pointer_aim=a.pointer_aim).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    Vt = torch.from_numpy(V); Gt = torch.from_numpy(G); At = torch.from_numpy(A)

    def batches(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), a.batch):
            j = torch.from_numpy(idx[i:i + a.batch])
            yield (Vt[j].to(dev, non_blocking=True).float(), Gt[j].to(dev, non_blocking=True).float(), At[j].to(dev))

    if v3:
        bins = torch.arange(nvec[2], device=dev).float()

    def loss_fn(logits, act):
        if v3:
            # move: CE; fire: CE (fires up-weighted); aim: CE against a circular Gaussian around the exact angle, only
            # where the controller fired; super distance: CE only on supers
            fire = act[:, 1]
            w = torch.where(fire > 0, torch.full_like(fire, a.fire_weight, dtype=torch.float32), torch.ones_like(fire, dtype=torch.float32))
            tot = F.cross_entropy(logits[0], act[:, 0]) + (F.cross_entropy(logits[1], fire, reduction="none") * w).mean()
            f = fire > 0
            if f.any():
                d = (bins[None] - act[f, 2][:, None].float()).abs(); d = torch.minimum(d, nvec[2] - d)
                tgt = torch.softmax(-0.5 * (d / a.aim_sigma) ** 2, -1)
                tot = tot + (-(tgt * torch.log_softmax(logits[2][f], -1)).sum(-1)).sum() / len(fire)
            sp = fire == 2
            if sp.any():
                tot = tot + F.cross_entropy(logits[3][sp], act[sp, 3], reduction="sum") / len(fire)
            return tot
        tot = 0.0
        for h, lg in enumerate(logits):
            y = act[:, h]
            w = torch.ones_like(y, dtype=torch.float32)
            if h in (1, 2):
                w = torch.where(y > 0, torch.full_like(w, a.fire_weight), w)
            tot = tot + (F.cross_entropy(lg, y, reduction="none") * w).mean()
        return tot

    log = csv.writer(open(os.path.join(out, "progress.csv"), "w")); hdr = None
    best = 1e9
    for ep in range(a.epochs):
        net.train(); t0 = time.time(); tl, n = 0.0, 0
        for v, g, act in batches(tr, True):
            logits, _ = net(v, g)
            loss = loss_fn(logits, act)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tl += loss.item() * len(act); n += len(act)
        sched.step()
        # held-out metrics
        net.eval(); vl, vn = 0.0, 0
        correct = np.zeros(4); cnt = 0; fire_hit = np.zeros(2); fire_n = np.zeros(2); none_fp = np.zeros(2); none_n = np.zeros(2)
        move_ang_ok = 0; move_n = 0; aim_err = []
        with torch.no_grad():
            for v, g, act in batches(va, False):
                logits, _ = net(v, g)
                vl += loss_fn(logits, act).item() * len(act); vn += len(act)
                pred = torch.stack([lg.argmax(-1) for lg in logits], -1)
                correct += (pred == act).float().sum(0).cpu().numpy(); cnt += len(act)
                if v3:
                    for k, val in enumerate((1, 2)):    # attack / super recall at argmax, false fires on none-steps
                        pos = act[:, 1] == val
                        fire_n[k] += pos.sum().item(); fire_hit[k] += (pred[pos, 1] == val).sum().item()
                        neg = act[:, 1] == 0; none_n[k] += neg.sum().item(); none_fp[k] += (pred[neg, 1] == val).sum().item()
                    f = act[:, 1] > 0
                    e = (pred[f, 2] - act[f, 2]).abs(); aim_err.append(torch.minimum(e, nvec[2] - e).cpu().numpy())
                else:
                  for k, h in enumerate((1, 2)):
                    pos = act[:, h] > 0
                    fire_n[k] += pos.sum().item(); fire_hit[k] += (pred[pos, h] == act[pos, h]).sum().item()
                    neg = ~pos; none_n[k] += neg.sum().item(); none_fp[k] += (pred[neg, h] > 0).sum().item()
                mv = act[:, 0] > 0
                d = (pred[mv, 0] - act[mv, 0]).abs(); d = torch.minimum(d, 16 - d)
                move_ang_ok += ((pred[mv, 0] > 0) & (d <= 1)).sum().item(); move_n += mv.sum().item()
        row = {"epoch": ep, "train_loss": tl / n, "val_loss": vl / vn, "acc_move": correct[0] / cnt, "acc_attack": correct[1] / cnt,
               "acc_super": correct[2] / cnt, "attack_recall": fire_hit[0] / max(1, fire_n[0]), "super_recall": fire_hit[1] / max(1, fire_n[1]),
               "attack_fp": none_fp[0] / max(1, none_n[0]), "super_fp": none_fp[1] / max(1, none_n[1]),
               "move_within_22deg": move_ang_ok / max(1, move_n), "sec": time.time() - t0, "lr": sched.get_last_lr()[0]}
        if v3:
            ae = np.concatenate(aim_err) if aim_err else np.zeros(0)
            row.update({"aim_median_err_deg": float(np.median(ae)) if len(ae) else -1.0, "aim_within_5deg": float((ae <= 5).mean()) if len(ae) else -1.0,
                        "aim_within_11deg": float((ae <= 11.25).mean()) if len(ae) else -1.0})
        if hdr is None:
            hdr = list(row); log.writerow(hdr)
        log.writerow([row[k] for k in hdr])
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        ck = {"model": net.state_dict(), "vec_dim": V.shape[1], "grid_shape": list(G.shape[1:]), "epoch": ep, "metrics": row, "args": vars(a), "nvec": nvec, "pointer_aim": a.pointer_aim}
        torch.save(ck, os.path.join(out, "bc_last.pt"))
        if row["val_loss"] < best:
            best = row["val_loss"]; torch.save(ck, os.path.join(out, "bc_best.pt"))
    print("done; best val loss", best)


if __name__ == "__main__":
    main()
