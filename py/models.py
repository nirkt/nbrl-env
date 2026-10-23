#!/usr/bin/env python3
"""Policies for the learn round.

StatePolicy (S2 teacher): input = state2.encode of the controlled player: vec (436,) + egocentric tile grid (6, 25, 25).
  grid CNN (6->32->64, stride 2) -> 256; vec MLP -> 256; trunk 512 -> heads move 17 / attack 18 / super 18 / gadget 2
  (logits) + value.
PixelPolicy (S3/S4 student): input = stacked 256x115 RGB frames (H, W, 3k) uint8 -> Nature-style CNN trunk -> the same
  heads + value + auxiliary decoders (label maps at 1/4 resolution, entity heatmaps, scalars) for 'understanding'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NVEC = [17, 18, 18, 2]        # v2: move, attack (none / tap / 16 aimed), super (none / tap / 16 aimed), gadget
# v3 (free aim, the user's request 2026-10-01): move (none / 16 joystick directions), fire (none / attack / super),
# aim (360 one-degree bins, x right / y down; used when fire > 0), super distance (8 bins of 2600/8; used when fire = 2)
NVEC3 = [17, 3, 360, 8]


def nvec_of(ck):
    """the head sizes a checkpoint was trained with (old checkpoints: v2)"""
    return list(ck.get("nvec", NVEC)) if isinstance(ck, dict) else list(NVEC)


def is_v3(nvec):
    return list(nvec) == NVEC3


class Heads(nn.Module):
    def __init__(self, d, nvec=None):
        super().__init__()
        self.nvec = list(nvec or NVEC)
        self.pi = nn.Linear(d, sum(self.nvec))
        self.v = nn.Linear(d, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0); nn.init.zeros_(self.v.bias)

    def forward(self, h):
        return torch.split(self.pi(h), self.nvec, dim=-1), self.v(h).squeeze(-1)


# state2 vector layout (state2.flatten): self 22 | enemy 9 x 16 | projectile 16 x 12 | box 6 x 5 | item 8 x 6
EN_OFF, EN_K, EN_F = 22, 9, 16
BOX_OFF, BOX_K, BOX_F = 22 + 9 * 16 + 16 * 12, 6, 5


def candidates_from_vec(vec):
    """aim candidates of the state vector: 9 enemies + 6 boxes -> (present (B,15), angle deg (B,15), features (B,15,18))"""
    B = vec.shape[0]
    en = vec[:, EN_OFF:EN_OFF + EN_K * EN_F].reshape(B, EN_K, EN_F)
    bx = vec[:, BOX_OFF:BOX_OFF + BOX_K * BOX_F].reshape(B, BOX_K, BOX_F)
    bxf = torch.cat([bx, torch.zeros(B, BOX_K, EN_F - BOX_F, device=vec.device)], -1)
    feat = torch.cat([en, bxf], 1)
    typ = torch.cat([torch.tensor([1.0, 0.0], device=vec.device).expand(B, EN_K, 2), torch.tensor([0.0, 1.0], device=vec.device).expand(B, BOX_K, 2)], 1)
    present = feat[:, :, 0] > 0.5
    ang = torch.rad2deg(torch.atan2(feat[:, :, 2], feat[:, :, 1])) % 360.0
    return present, ang, torch.cat([feat, typ], -1)


def intercept_angles(vec, present, ang, cf, bullet=197.0):
    """research round 2026-10-02 (D5): for enemy candidates, the angle to the linear-lead intercept point of a bullet at
    `bullet` units/tick (state features: dx, dy / 3000, vx, vy / 40 per tick); boxes keep their current angle."""
    is_en = cf[..., 16] > 0.5
    d = cf[..., 1:3] * 3000.0
    v = cf[..., 3:5] * 40.0 * is_en[..., None].float()
    a = (v * v).sum(-1) - bullet * bullet
    b = 2.0 * (d * v).sum(-1)
    c = (d * d).sum(-1)
    t = (-b - torch.sqrt((b * b - 4 * a * c).clamp(min=0))) / (2 * a)
    p = d + v * t.clamp(min=0, max=40)[..., None]
    lead = torch.rad2deg(torch.atan2(p[..., 1], p[..., 0])) % 360.0
    return torch.where(is_en & present, lead, ang)


class PointerAim(nn.Module):
    """free 360-bin aim distribution built as a mixture: for every candidate (visible enemy / box) a circular Gaussian
    bump at its exact angle + a learned offset (lead, up to +/-25 deg) with a learned width, plus a free component over
    all 360 bins; mixture weights from the policy features. Output = log-probabilities (valid logits)."""

    def __init__(self, hdim, nbins=360, cdim=18, lead=False):
        super().__init__()
        self.nbins = nbins; self.lead = bool(lead)
        self.cand = nn.Sequential(nn.Linear(hdim + cdim, 256), nn.ReLU(), nn.Linear(256, 3))
        self.free_score = nn.Linear(hdim, 1)
        self.free = nn.Linear(hdim, nbins)
        nn.init.zeros_(self.free.weight); nn.init.zeros_(self.free.bias)
        self.register_buffer("bins", torch.arange(nbins).float() * (360.0 / nbins))

    def forward(self, h, vec):
        present, ang, cf = candidates_from_vec(vec)
        if self.lead:
            ang = intercept_angles(vec, present, ang, cf)
        B, K, _ = cf.shape
        o = self.cand(torch.cat([h[:, None].expand(B, K, h.shape[-1]), cf], -1))
        score = o[..., 0].masked_fill(~present, -1e9)
        mu = ang + 25.0 * torch.tanh(o[..., 1])
        sig = F.softplus(o[..., 2]) + 0.7
        d = (self.bins[None, None] - mu[..., None]).abs() % 360.0; d = torch.minimum(d, 360.0 - d)
        bump = torch.log_softmax(-0.5 * (d / sig[..., None]) ** 2, -1)                       # (B, K, nbins)
        mix = torch.log_softmax(torch.cat([score, self.free_score(h)], -1), -1)              # (B, K+1)
        comp = torch.cat([bump, torch.log_softmax(self.free(h), -1)[:, None]], 1)            # (B, K+1, nbins)
        return torch.logsumexp(mix[..., None] + comp, 1)


class StatePolicy(nn.Module):
    def __init__(self, vec_dim=436, grid_shape=(6, 25, 25), hidden=512, nvec=None, pointer_aim=False, pointer_lead=False):
        super().__init__()
        c = grid_shape[0]
        self.grid = nn.Sequential(nn.Conv2d(c, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            gd = self.grid(torch.zeros(1, *grid_shape)).shape[1]
        self.gfc = nn.Sequential(nn.Linear(gd, 256), nn.ReLU())
        self.vfc = nn.Sequential(nn.Linear(vec_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.heads = Heads(hidden, nvec)
        self.pointer_aim = bool(pointer_aim)
        if self.pointer_aim:
            assert is_v3(self.heads.nvec), "pointer aim needs the v3 heads"
            self.aim = PointerAim(hidden, self.heads.nvec[2], lead=pointer_lead)

    def features(self, vec, grid):
        return self.trunk(torch.cat([self.vfc(vec), self.gfc(self.grid(grid))], dim=-1))

    def forward(self, vec, grid):
        h = self.features(vec, grid)
        logits, v = self.heads(h)
        if self.pointer_aim:
            logits = list(logits); logits[2] = self.aim(h, vec); logits = tuple(logits)
        return logits, v


def make_state_policy(ck, dev=None):
    """build the StatePolicy a checkpoint was trained with (heads, pointer aim) and load its weights"""
    net = StatePolicy(ck["vec_dim"], tuple(ck["grid_shape"]), nvec=nvec_of(ck), pointer_aim=ck.get("pointer_aim", False),
                      pointer_lead=ck.get("pointer_lead", False))
    net.load_state_dict(ck["model"])
    return net.to(dev) if dev is not None else net


def dist_logp_ent(logits, actions=None, masks=None):
    """categorical heads -> (log-prob of actions, entropy, sampled actions). v3 heads: the aim head only counts when
    the fire head fires and the distance head only for a super (their values do nothing otherwise)."""
    lps, ents, samples = [], [], []
    for i, lg in enumerate(logits):
        d = torch.distributions.Categorical(logits=lg)
        a = d.sample() if actions is None else actions[:, i]
        samples.append(a); lps.append(d.log_prob(a)); ents.append(d.entropy())
    lp, en = torch.stack(lps, -1), torch.stack(ents, -1)
    if len(logits) == 4 and logits[1].shape[-1] == 3:
        fire = samples[1].float()
        m = torch.stack([torch.ones_like(fire), torch.ones_like(fire), (fire > 0).float(), (fire == 2).float()], -1)
        lp, en = lp * m, en * m
    return lp.sum(-1), en.sum(-1), torch.stack(samples, -1)


def to_torch_state(vec, grid, dev):
    v = torch.as_tensor(vec, dtype=torch.float32, device=dev)
    g = torch.as_tensor(grid, dtype=torch.float32, device=dev)
    if v.dim() == 1:
        v, g = v[None], g[None]
    return v, g
