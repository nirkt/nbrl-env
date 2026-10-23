"""P2 pixel student (pctrain round): sees 4 stacked 256x115 RGB frames (the phone's preprocess, HUD mask applied) and is trained
to (a) act like the state teacher (KL to the teacher's four v3 heads) and (b) decode the game state from the pixels:
5 ground maps (wall, water, bush, storm, off-map) and 7 entity heat maps (self, enemy, enemy projectile, own projectile,
box, power cube, bomb) at 64x29 (labels.py: the state projected through proj.py), plus HUD scalars (own HP fraction, ammo,
super fraction, brawlers left, in storm).

  encoder  12 -> 32 (5x5 s2) 58x128 -> 64 (s2) 29x64 [F4] + res -> 128 (s2) 15x32 [F8] + res -> 192 (s2) 8x16 [F16]
  decoder  F16 -> up + F8 -> up + F4 -> 12 maps (29x64)
  pooled   F16 -> 1x1 64 -> 512 (pixel features h_pix)
  policy   [h_pix, priv * keep] -> trunk 512 -> move 17 / fire 3 / aim 360 / distance 8 (+ value, unused)
           priv = the teacher-style state encoder (fair vector + grid; side input, dropped with probability p per sample;
           keep = 0 at deployment = pixels only)
  aim      PixelPointerAim (learn round, E4): Gaussian bumps at the angles of the student's OWN decoded enemy/box heat-map peaks
           (inverse camera projection, exact geometry) + learned offset/width + a free 360-bin component; the decoded maps are
           detached (the policy loss does not deform them).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn"))
from models4 import Heads, StatePolicy, NVEC3

AUX_H, AUX_W = 29, 64
MAP_K = ["wall", "water", "bush", "gas", "offmap"]
ENT_K = ["self", "enemy", "proj_enemy", "proj_own", "box", "cube", "bomb"]
SCAL_K = ["hp_frac", "ammo", "super_frac", "brawlers_left", "in_gas"]
SCAL_SCALE = {"hp_frac": 1.0, "ammo": 1 / 3.0, "super_frac": 1.0, "brawlers_left": 0.1, "in_gas": 1.0}


def ground_table():
    import proj
    P = proj.load_params()
    us = (np.arange(AUX_W) + 0.5) * proj.FULL[0] / AUX_W
    vs = (np.arange(AUX_H) + 0.5) * proj.FULL[1] / AUX_H
    u, v = np.meshgrid(us, vs)
    th, D, Fc = P["th"], P["D"], P["F"]
    dv = v - P["v0"]
    Y = dv * D / (Fc * np.cos(th) + dv * np.sin(th))
    X = (u - P["u0"]) * (D - Y * np.sin(th)) / Fc
    return np.stack([X, Y], -1).astype(np.float32)


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, 1, 1); self.b = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))


class PixelPointerAim(nn.Module):
    def __init__(self, hdim, nbins=360, k_enemy=6, k_box=4):
        super().__init__()
        self.nbins, self.ke, self.kb = nbins, k_enemy, k_box
        self.register_buffer("gt", torch.from_numpy(ground_table()))
        self.register_buffer("bins", torch.arange(nbins).float() * (360.0 / nbins))
        self.cand = nn.Sequential(nn.Linear(hdim + 6, 256), nn.ReLU(), nn.Linear(256, 3))
        self.free_score = nn.Linear(hdim, 1); self.free = nn.Linear(hdim, nbins)
        nn.init.zeros_(self.free.weight); nn.init.zeros_(self.free.bias)

    def _peaks(self, hm, k):
        B, H, W = hm.shape
        mx = F.max_pool2d(hm[:, None], 3, 1, 1)[:, 0]
        val, idx = torch.where(hm == mx, hm, torch.zeros_like(hm)).flatten(1).topk(k, -1)
        w = F.unfold(hm[:, None], 3, padding=1)
        g = F.unfold(self.gt.permute(2, 0, 1)[None].expand(B, 2, H, W), 3, padding=1).view(B, 2, 9, H * W)
        wi = torch.gather(w, 2, idx[:, None].expand(B, 9, k))
        gi = torch.gather(g, 3, idx[:, None, None].expand(B, 2, 9, k))
        xy = (gi * wi[:, None]).sum(2) / wi.sum(1, keepdim=True).clamp_min(1e-6)
        return val, xy.permute(0, 2, 1)

    def forward(self, h, heat):
        sv, sxy = self._peaks(heat[:, ENT_K.index("self")], 1)
        ev, exy = self._peaks(heat[:, ENT_K.index("enemy")], self.ke)
        bv, bxy = self._peaks(heat[:, ENT_K.index("box")], self.kb)
        val = torch.cat([ev, bv], 1); rel = torch.cat([exy, bxy], 1) - sxy
        B, K = val.shape
        typ = torch.zeros(B, K, 2, device=h.device); typ[:, :self.ke, 0] = 1; typ[:, self.ke:, 1] = 1
        dist = rel.norm(dim=-1, keepdim=True)
        ang = torch.rad2deg(torch.atan2(rel[..., 1], rel[..., 0])) % 360.0
        cf = torch.cat([val[..., None], typ, dist / 3000.0, rel / 3000.0], -1)
        o = self.cand(torch.cat([h[:, None].expand(B, K, h.shape[-1]), cf], -1))
        present = (val > 0.3) & (sv > 0.3)
        score = o[..., 0].masked_fill(~present, -1e9)
        mu = ang + 25.0 * torch.tanh(o[..., 1]); sig = F.softplus(o[..., 2]) + 0.7
        d = (self.bins[None, None] - mu[..., None]).abs() % 360.0; d = torch.minimum(d, 360.0 - d)
        bump = torch.log_softmax(-0.5 * (d / sig[..., None]) ** 2, -1)
        mix = torch.log_softmax(torch.cat([score, self.free_score(h)], -1), -1)
        comp = torch.cat([bump, torch.log_softmax(self.free(h), -1)[:, None]], 1)
        return torch.logsumexp(mix[..., None] + comp, 1)


class PixelStudent4(nn.Module):
    def __init__(self, k=4, privileged=True, pointer_aim=True):
        super().__init__()
        c = 3 * k; self.k = k
        self.s1 = nn.Sequential(nn.Conv2d(c, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), Res(64))   # 29x64
        self.s2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), Res(128))                                        # 15x32
        self.s3 = nn.Sequential(nn.Conv2d(128, 192, 3, 2, 1), nn.ReLU(), Res(192))                                       # 8x16
        self.u1 = nn.Sequential(nn.Conv2d(192 + 128, 128, 3, 1, 1), nn.ReLU())
        self.u2 = nn.Sequential(nn.Conv2d(128 + 64, 96, 3, 1, 1), nn.ReLU(), nn.Conv2d(96, 64, 3, 1, 1), nn.ReLU())
        self.aux_out = nn.Conv2d(64, len(MAP_K) + len(ENT_K), 1)
        self.squeeze = nn.Sequential(nn.Conv2d(192, 64, 1), nn.ReLU())
        self.pool = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 16, 512), nn.ReLU())
        self.scal = nn.Linear(512, len(SCAL_K))
        self.privileged = privileged
        if privileged:
            sp = StatePolicy(436, (6, 25, 25))
            self.p_grid, self.p_gfc, self.p_vfc = sp.grid, sp.gfc, sp.vfc
            self.p_fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(512 + (256 if privileged else 0), 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU())
        self.heads = Heads(512, NVEC3)
        self.pointer_aim = pointer_aim
        if pointer_aim:
            self.aim_ptr = PixelPointerAim(512)

    def encode(self, x):
        x = x.permute(0, 3, 1, 2).float() / 255.0
        f4 = self.s1(x); f8 = self.s2(f4); f16 = self.s3(f8)
        return f4, f8, f16, self.pool(self.squeeze(f16))

    def aux(self, f4, f8, f16):
        u = F.interpolate(f16, size=f8.shape[-2:], mode="bilinear", align_corners=False)
        u = self.u1(torch.cat([u, f8], 1))
        u = F.interpolate(u, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        return self.aux_out(self.u2(torch.cat([u, f4], 1)))

    def forward(self, x, vec=None, grid=None, keep=None):
        f4, f8, f16, h = self.encode(x)
        parts = [h]
        if self.privileged:
            if vec is None or keep is None:
                parts.append(torch.zeros(h.shape[0], 256, device=h.device, dtype=h.dtype))
            else:
                g = grid.to(vec.dtype) if grid.dtype != vec.dtype else grid
                pf = self.p_fc(torch.cat([self.p_vfc(vec), self.p_gfc(self.p_grid(g))], -1))
                parts.append(pf * keep[:, None])
        ht = self.trunk(torch.cat(parts, -1))
        logits, v = self.heads(ht)
        aux = self.aux(f4, f8, f16)
        if self.pointer_aim:
            heat = torch.sigmoid(aux[:, len(MAP_K):]).detach()
            logits = list(logits); logits[2] = self.aim_ptr(ht, heat); logits = tuple(logits)
        return {"logits": logits, "value": v, "aux": aux, "scal": self.scal(h)}


def distill_kl(s_logits, t_logits):
    """KL(teacher || student) of the factored v3 policy (masked like the policy: aim only when the teacher fires)"""
    kls = []
    for s, t in zip(s_logits, t_logits):
        pt = torch.log_softmax(t.float(), -1); ps = torch.log_softmax(s.float(), -1)
        kls.append((pt.exp() * (pt - ps)).sum(-1))
    pf = torch.softmax(t_logits[1].float(), -1)
    return kls[0] + kls[1] + (1 - pf[:, 0]) * kls[2] + pf[:, 2] * kls[3]


def aux_loss(out, maps, heat, scal):
    a = out["aux"].float(); nm = len(MAP_K)
    lm = F.binary_cross_entropy_with_logits(a[:, :nm], maps.float())
    ht = heat.float()
    lh = (F.binary_cross_entropy_with_logits(a[:, nm:], ht, reduction="none") * (1.0 + 20.0 * ht)).mean()
    sc = out["scal"].float()
    ls = F.mse_loss(torch.sigmoid(sc[:, [0, 2, 4]]), scal[:, [0, 2, 4]]) + F.mse_loss(sc[:, [1, 3]], scal[:, [1, 3]])
    return lm, lh, ls
