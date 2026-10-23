"""Rebuild the encoder's 6 x 25 x 25 egocentric grid on the GPU from compact metadata (pctrain round: BC/DAgger shards store
the 436-vector in float16 + the observer's exact position / tick + the visible enemies' grid cells instead of 3,750 grid
bytes per sample). Channels (state2.encode): wall, water, bush, storm-or-offmap, offmap, enemy count. The map is the
battle's tile map (fixed for this Showdown map: checked per shard by map hash)."""
import numpy as np
import torch
import state2 as S2


class GridBuilder:
    def __init__(self, map_tiles, W, H, dev):
        m = S2.MapInfo(W, H, list(map_tiles))
        self.W, self.H, self.dev = int(W), int(H), dev
        self.wall = torch.as_tensor(m.wall, device=dev); self.water = torch.as_tensor(m.water, device=dev)
        self.bush = torch.as_tensor(m.bush, device=dev)
        ys, xs = np.mgrid[0:H, 0:W]
        self.ring = torch.as_tensor(np.minimum(np.minimum(xs, ys), np.minimum(W - 1 - xs, H - 1 - ys)), device=dev)
        self.off = torch.arange(-12, 13, device=dev)

    def build(self, x, y, tick, ecell):
        """x, y, tick (B,) int tensors (world units / ticks); ecell (B, 9) int16 (gy*25+gx of visible enemies, -1 none)
        -> (B, 6, 25, 25) uint8"""
        B = x.shape[0]; W, H = self.W, self.H
        tx = torch.div(x, 300, rounding_mode="floor"); ty = torch.div(y, 300, rounding_mode="floor")
        xs = tx[:, None] + self.off[None]; ys = ty[:, None] + self.off[None]                         # (B, 25)
        inside = ((xs >= 0) & (xs < W))[:, None, :] & ((ys >= 0) & (ys < H))[:, :, None]           # (B, 25y, 25x)
        cx = xs.clamp(0, W - 1)[:, None, :].expand(B, 25, 25); cy = ys.clamp(0, H - 1)[:, :, None].expand(B, 25, 25)
        step = torch.clamp(tick - 440, min=0) // 100
        g = torch.zeros(B, 6, 25, 25, dtype=torch.uint8, device=self.dev)
        g[:, 0] = (self.wall[cy, cx] & inside).to(torch.uint8)
        g[:, 1] = (self.water[cy, cx] & inside).to(torch.uint8)
        g[:, 2] = (self.bush[cy, cx] & inside).to(torch.uint8)
        g[:, 3] = ((self.ring[cy, cx] < step[:, None, None]) | ~inside).to(torch.uint8)
        g[:, 4] = (~inside).to(torch.uint8)
        e = ecell.long(); ok = e >= 0
        if ok.any():
            bi = torch.arange(B, device=self.dev)[:, None].expand_as(e)[ok]; c = e[ok]
            flat = g[:, 5].reshape(B, 625)
            flat.index_put_((bi, c), torch.ones_like(c, dtype=torch.uint8), accumulate=True)
            g[:, 5] = flat.reshape(B, 25, 25)
        return g
