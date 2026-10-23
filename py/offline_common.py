#!/usr/bin/env python3
"""Shared pieces of the research-round offline experiments (D2, D3, E1, D1):
- the v3 BC samples of the 300 s1b demos as memory-mapped arrays (work/data/s1b_bc3/*.npy, extracted from the learn
  round's s1b_bc3.npz: vec float16 (N,436), grid uint8 (N,6,25,25), act int16 (N,4), meta int64 (N,5) =
  (seed, step, player, char id, tick))
- the train / held-out split of runs/bc3p (30 held-out seeds) so every variant is compared on the same episodes
- per-player outcomes (work/evidence/outcomes_s1b.csv) -> per-sample final rank
- the 'fair' transform (E1): enemies the player could not see (the visibility rule of nbpc/fair.py) are removed from the vector
  (remaining visible ones compacted to the front, 'visible' feature set to 1) and from grid channel 5
"""
import json, os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "work", "data", "s1b_bc3")
EN_OFF, EN_K, EN_F = 22, 9, 16


def load_bc3(mmap=True):
    m = "r" if mmap else None
    return {k: np.load(os.path.join(DATA, k + ".npy"), mmap_mode=m) for k in ("vec", "grid", "act", "meta")}


def split_bc3p(meta):
    val = set(json.load(open(os.path.join(ROOT, "inputs", "data", "runs", "bc3p", "split.json")))["val_seeds"])
    is_val = np.isin(meta[:, 0], list(val))
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def sample_ranks(meta):
    import pandas as pd
    o = pd.read_csv(os.path.join(ROOT, "work", "evidence", "outcomes_s1b.csv"))
    key = {(int(s), int(p)): int(r) for s, p, r in zip(o.seed, o.p, o["rank"])}
    return np.array([key[(int(s), int(p))] for s, p in zip(meta[:, 0], meta[:, 2])], np.int64)


class FairTransform:
    """GPU batch transform: remove enemies the controlled player cannot see (masks from nbpc/fair.py)."""

    def __init__(self, path, dev):
        f = np.load(path)
        self.vis = torch.from_numpy(f["vis"])
        cells = f["hid_cells"]                     # (M, 3) sample, gy, gx (sorted by sample)
        order = np.argsort(cells[:, 0], kind="stable"); cells = cells[order]
        self.cells = cells
        self.indptr = np.searchsorted(cells[:, 0], np.arange(len(f["vis"]) + 1))
        self.dev = dev

    def __call__(self, idx, v, g):
        B = v.shape[0]
        vis = self.vis[idx].to(self.dev)                                  # (B, 9)
        en = v[:, EN_OFF:EN_OFF + EN_K * EN_F].view(B, EN_K, EN_F)
        key = (~vis).float() * 10 + torch.arange(EN_K, device=self.dev).float()[None] * 0.01   # visible first, keep order
        order = key.argsort(-1)
        en = torch.gather(en, 1, order[..., None].expand(B, EN_K, EN_F))
        keep = torch.gather(vis, 1, order).float()[..., None]
        en = en * keep
        en[..., 11] = keep[..., 0]                                         # 'visible' = 1 for every kept enemy
        v = torch.cat([v[:, :EN_OFF], en.reshape(B, -1), v[:, EN_OFF + EN_K * EN_F:]], 1)
        # grid channel 5: subtract hidden enemies
        sel = [np.arange(self.indptr[i], self.indptr[i + 1]) for i in idx]
        if sum(len(s) for s in sel):
            rows = np.concatenate([np.full(len(s), b) for b, s in enumerate(sel)]); cs = self.cells[np.concatenate(sel)]
            g = g.clone()
            ii = (torch.from_numpy(rows).long().to(self.dev), torch.full((len(rows),), 5, dtype=torch.long, device=self.dev),
                  torch.from_numpy(cs[:, 1]).long().to(self.dev), torch.from_numpy(cs[:, 2]).long().to(self.dev))
            g.index_put_(ii, torch.full((len(rows),), -1.0, device=self.dev), accumulate=True)
            g[:, 5].clamp_(min=0)
        return v, g


def batches(D, idx, bs, dev, shuffle, rng=None, extra=None):
    idx = (rng or np.random).permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        j = np.sort(idx[i:i + bs])
        v = torch.from_numpy(np.asarray(D["vec"][j], np.float32)).to(dev)
        g = torch.from_numpy(np.asarray(D["grid"][j], np.float32)).to(dev)
        a = torch.from_numpy(np.asarray(D["act"][j], np.int64)).to(dev)
        yield j, v, g, a
