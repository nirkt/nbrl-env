#!/usr/bin/env python3
"""S3 pixel student: CNN over a stack of k=4 256x115 frames, trained by DAgger-style distillation from the state
teacher, with 'understanding' decoders that must reconstruct the S0 labels from the pixels alone.

Model (PixelStudent):
  encoder  conv 12->32 (5, s2) 128x58 -> 64 (3, s2) 64x29 [F4] -> 96 (3, s2) 32x15 -> 128 (3, s2) 16x8 [F16]
  aux      (pixels only) U-Net style decoder F16 -> 64x29 (+ skip F4): 5 ground maps (wall, water, bush, gas, offmap) and
           7 entity heat maps (self, enemy, proj_enemy, proj_own, box, cube, bomb); scalars (hp_frac, ammo/3, super_frac,
           brawlers_left/10, in_gas) from the pooled pixel features
  policy   pixel features (512) [+ privileged state features (256) * keep, keep ~ Bernoulli(1 - p_drop) per sample]
           -> trunk 512 -> heads move 17 / attack 18 / super 18 / gadget 2 + value
Labels at 64x29 come from labels.py (ground lookup + projected client positions), sigma 1.0 px for heat maps.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Heads, StatePolicy, nvec_of

AUX_H, AUX_W = 29, 64
MAP_K = ["wall", "water", "bush", "gas", "offmap"]
ENT_K = ["self", "enemy", "proj_enemy", "proj_own", "box", "cube", "bomb"]
SCAL_K = ["hp_frac", "ammo", "super_frac", "brawlers_left", "in_gas"]
SCAL_SCALE = {"hp_frac": 1.0, "ammo": 1 / 3.0, "super_frac": 1.0, "brawlers_left": 0.1, "in_gas": 1.0}


def ground_table(P=None):
    """(AUX_H, AUX_W, 2): world ground offset (X, Y) relative to the camera anchor of every 64x29 aux-cell centre (S0
    inverse projection); differences of two cells = their world offset, independent of where the camera is"""
    import proj
    P = P or proj.load_params()
    us = (np.arange(AUX_W) + 0.5) * proj.FULL[0] / AUX_W
    vs = (np.arange(AUX_H) + 0.5) * proj.FULL[1] / AUX_H
    u, v = np.meshgrid(us, vs)
    th, D, Fc = P["th"], P["D"], P["F"]
    dv = v - P["v0"]
    Y = dv * D / (Fc * np.cos(th) + dv * np.sin(th))
    X = (u - P["u0"]) * (D - Y * np.sin(th)) / Fc
    return np.stack([X, Y], -1).astype(np.float32)


class PixelPointerAim(nn.Module):
    """v3 aim from what the student SEES: candidates = the top peaks of its own decoded enemy (6) and box (4) heat maps;
    the world angle from its decoded self position to each peak goes through the inverse camera projection (sub-cell
    centroid in a 3x3 window); then the same mixture as models.PointerAim (Gaussian bump per candidate at its angle +
    learned offset/width, plus a free 360-bin component)."""

    def __init__(self, hdim, nbins=360, k_enemy=6, k_box=4):
        super().__init__()
        self.nbins, self.ke, self.kb = nbins, k_enemy, k_box
        self.register_buffer("gt", torch.from_numpy(ground_table()))          # (H, W, 2)
        self.register_buffer("bins", torch.arange(nbins).float() * (360.0 / nbins))
        self.cand = nn.Sequential(nn.Linear(hdim + 6, 256), nn.ReLU(), nn.Linear(256, 3))
        self.free_score = nn.Linear(hdim, 1)
        self.free = nn.Linear(hdim, nbins)
        nn.init.zeros_(self.free.weight); nn.init.zeros_(self.free.bias)

    def _peaks(self, hm, k):
        """hm (B, H, W) probabilities -> top-k local maxima: value (B,k), world (X, Y) (B,k,2) by 3x3 centroid"""
        B, H, W = hm.shape
        mx = F.max_pool2d(hm[:, None], 3, 1, 1)[:, 0]
        val, idx = torch.where(hm == mx, hm, torch.zeros_like(hm)).flatten(1).topk(k, -1)
        w = F.unfold(hm[:, None], 3, padding=1)                                  # (B, 9, H*W)
        g = F.unfold(self.gt.permute(2, 0, 1)[None].expand(B, 2, H, W), 3, padding=1).view(B, 2, 9, H * W)
        wi = torch.gather(w, 2, idx[:, None].expand(B, 9, k))                     # (B, 9, k)
        gi = torch.gather(g, 3, idx[:, None, None].expand(B, 2, 9, k))            # (B, 2, 9, k)
        xy = (gi * wi[:, None]).sum(2) / wi.sum(1, keepdim=True).clamp_min(1e-6)  # (B, 2, k)
        return val, xy.permute(0, 2, 1)

    def forward(self, h, heat):
        """heat: sigmoid entity heat maps (B, 7, H, W) in ENT_K order"""
        sv, sxy = self._peaks(heat[:, ENT_K.index("self")], 1)
        ev, exy = self._peaks(heat[:, ENT_K.index("enemy")], self.ke)
        bv, bxy = self._peaks(heat[:, ENT_K.index("box")], self.kb)
        val = torch.cat([ev, bv], 1); rel = torch.cat([exy, bxy], 1) - sxy        # (B, K), (B, K, 2) world units
        B, K = val.shape
        typ = torch.cat([torch.tensor([1.0, 0.0], device=h.device).expand(B, self.ke, 2), torch.tensor([0.0, 1.0], device=h.device).expand(B, self.kb, 2)], 1)
        dist = rel.norm(dim=-1, keepdim=True)
        ang = torch.rad2deg(torch.atan2(rel[..., 1], rel[..., 0])) % 360.0
        cf = torch.cat([val[..., None], typ, dist / 3000.0, rel / 3000.0], -1)    # (B, K, 6)
        o = self.cand(torch.cat([h[:, None].expand(B, K, h.shape[-1]), cf], -1))
        present = (val > 0.3) & (sv > 0.3)
        score = o[..., 0].masked_fill(~present, -1e9)
        mu = ang + 25.0 * torch.tanh(o[..., 1])
        sig = F.softplus(o[..., 2]) + 0.7
        d = (self.bins[None, None] - mu[..., None]).abs() % 360.0; d = torch.minimum(d, 360.0 - d)
        bump = torch.log_softmax(-0.5 * (d / sig[..., None]) ** 2, -1)
        mix = torch.log_softmax(torch.cat([score, self.free_score(h)], -1), -1)
        comp = torch.cat([bump, torch.log_softmax(self.free(h), -1)[:, None]], 1)
        return torch.logsumexp(mix[..., None] + comp, 1)


class PixelStudent(nn.Module):
    def __init__(self, k=4, vec_dim=436, grid_shape=(6, 25, 25), privileged=True, nvec=None, pointer_aim=False):
        super().__init__()
        c = 3 * k
        self.e1 = nn.Sequential(nn.Conv2d(c, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())        # 29x64
        self.e2 = nn.Sequential(nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(), nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU())      # 8x16
        self.d1 = nn.Sequential(nn.Conv2d(128 + 64, 96, 3, 1, 1), nn.ReLU(), nn.Conv2d(96, 64, 3, 1, 1), nn.ReLU())
        self.aux_out = nn.Conv2d(64, len(MAP_K) + len(ENT_K), 1)
        self.pool = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 16, 512), nn.ReLU())
        self.scal = nn.Linear(512, len(SCAL_K))
        self.privileged = privileged
        if privileged:
            sp = StatePolicy(vec_dim, grid_shape)
            self.p_grid, self.p_gfc, self.p_vfc = sp.grid, sp.gfc, sp.vfc
            self.p_fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(512 + (256 if privileged else 0), 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.heads = Heads(512, nvec)
        self.pointer_aim = bool(pointer_aim)
        if self.pointer_aim:
            self.aim_ptr = PixelPointerAim(512, self.heads.nvec[2])

    def encode(self, x):
        """x: (B, H, W, 3k) uint8 -> F4, F16, pooled features"""
        x = x.permute(0, 3, 1, 2).float() / 255.0
        f4 = self.e1(x)
        f16 = self.e2(f4)
        return f4, f16, self.pool(f16)

    def aux(self, f4, f16):
        up = F.interpolate(f16, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        return self.aux_out(self.d1(torch.cat([up, f4], 1)))

    def forward(self, x, vec=None, grid=None, keep=None, want_aux=True):
        f4, f16, h = self.encode(x)
        parts = [h]
        if self.privileged:
            if vec is None or keep is None:
                parts.append(torch.zeros(h.shape[0], 256, device=h.device))
            else:
                pf = self.p_fc(torch.cat([self.p_vfc(vec), self.p_gfc(self.p_grid(grid))], -1))
                parts.append(pf * keep[:, None])
        ht = self.trunk(torch.cat(parts, -1))
        logits, v = self.heads(ht)
        out = {"logits": logits, "value": v, "scal": self.scal(h)}
        if want_aux or self.pointer_aim:
            out["aux"] = self.aux(f4, f16)
        if self.pointer_aim:
            heat = torch.sigmoid(out["aux"][:, len(MAP_K):]).detach()          # aim from what it decodes (no gradient into the decoders)
            logits = list(logits); logits[2] = self.aim_ptr(ht, heat); out["logits"] = tuple(logits)
        return out


def aux_targets(view, pidx, cam, P=None):
    """label arrays for one frame at AUX res: maps (5, 29, 64) uint8, heat (7, 29, 64) float16, scal (5,) float32"""
    import labels as LB
    ml, hm, pts, sc = LB.render_all(view, pidx, cam, AUX_W, AUX_H, P, sigma=1.0)
    maps = np.stack([ml[k] for k in MAP_K]).astype(np.uint8)
    heat = np.stack([hm[k] for k in ENT_K]).astype(np.float16)
    scal = np.array([sc[k] * SCAL_SCALE[k] for k in SCAL_K], np.float32)
    return maps, heat, scal, pts


def shift_batch(fr, maps, heat, k):
    """augmentation: one random shift per batch by (dy, dx) cells of 4 obs px, frames (B,115,256,C) and label maps
    (B,c,29,64) together; vacated = 0"""
    dy, dx = np.random.randint(-k, k + 1, size=2)

    def sh(t, ay, ax, py, px):
        t = torch.roll(t, shifts=(int(py), int(px)), dims=(ay, ax))
        if py > 0: t.narrow(ay, 0, int(py)).zero_()
        if py < 0: t.narrow(ay, t.shape[ay] + int(py), -int(py)).zero_()
        if px > 0: t.narrow(ax, 0, int(px)).zero_()
        if px < 0: t.narrow(ax, t.shape[ax] + int(px), -int(px)).zero_()
        return t
    return sh(fr, 1, 2, dy * 4, dx * 4), sh(maps, 2, 3, dy, dx), sh(heat, 2, 3, dy, dx)


def aux_loss(out, maps, heat, scal):
    a = out["aux"]
    nm = len(MAP_K)
    lm = F.binary_cross_entropy_with_logits(a[:, :nm], maps.float())
    # heat maps: weighted BCE (positives are rare)
    ht = heat.float()
    w = 1.0 + 20.0 * ht
    lh = (F.binary_cross_entropy_with_logits(a[:, nm:], ht, reduction="none") * w).mean()
    ls = F.mse_loss(torch.sigmoid(out["scal"][:, [0, 2, 4]]), scal[:, [0, 2, 4]]) + F.mse_loss(out["scal"][:, [1, 3]], scal[:, [1, 3]])
    return lm, lh, ls


def distill_loss(logits, teacher_logits):
    """sum over heads of KL(teacher || student)"""
    tot = 0.0
    for s, t in zip(logits, teacher_logits):
        pt = torch.log_softmax(t, -1); ps = torch.log_softmax(s, -1)
        tot = tot + (pt.exp() * (pt - ps)).sum(-1).mean()
    return tot


class PixelTorchPolicy:
    """evaluate.py adapter: pixels only (privileged input dropped)."""

    def __init__(self, path, deterministic=False):
        import torch_rocm_fix  # noqa
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PixelStudent(k=ck.get("k", 4), vec_dim=ck.get("vec_dim", 436), grid_shape=tuple(ck.get("grid_shape", (6, 25, 25))),
                                privileged=ck.get("privileged", True), nvec=nvec_of(ck), pointer_aim=ck.get("pointer_aim", False)).to(self.dev).eval()
        self.net.load_state_dict(ck["model"]); self.det = deterministic

    def reset(self): pass

    def act(self, obs, info):
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs if not isinstance(obs, dict) else obs["pixels"]), device=self.dev)[None]
            out = self.net(x, want_aux=False)
            lg = out["logits"]
            a = [l.argmax(-1) for l in lg] if self.det else [torch.distributions.Categorical(logits=l).sample() for l in lg]
        return np.array([int(t.item()) for t in a])
