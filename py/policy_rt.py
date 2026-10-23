"""Fast batched inference for the multi-agent loop: the actor (+ optional critic) forward, Gumbel-max sampling of the four
v3 heads and the masked log-probability (models.dist_logp_ent semantics: the aim head counts only when firing, the
distance head only for a super) captured in one CUDA graph per batch-size bucket. Inputs are padded to the bucket.
3.8 ms per eager call (launch-bound, any batch) -> ~1.3 ms per replay (P0 measurement)."""
import numpy as np
import torch


def sample_heads(logits, fire_shift=0.0, deterministic=False):
    acts, lps = [], []
    for k, l in enumerate(logits):
        l = l.float()
        if k == 1 and fire_shift:
            l = torch.cat([l[:, :1], l[:, 1:] - fire_shift], -1)
        lsm = torch.log_softmax(l, -1)
        if deterministic:
            a = torch.argmax(lsm, -1)
        else:
            u = torch.rand_like(lsm).clamp_(1e-12, 1.0)
            a = torch.argmax(lsm - torch.log(-torch.log(u)), -1)
        acts.append(a); lps.append(lsm.gather(-1, a[:, None])[:, 0])
    A = torch.stack(acts, -1); L = torch.stack(lps, -1)
    fire = A[:, 1]
    m = torch.stack([torch.ones_like(fire), torch.ones_like(fire), (fire > 0), (fire == 2)], -1).to(L.dtype)
    return A, (L * m).sum(-1)


class GraphPolicy:
    def __init__(self, actor, critic=None, buckets=(16, 64, 160), dev=None, fire_shift=0.0, deterministic=False, use_graph=True):
        self.dev = dev or next(actor.parameters()).device
        self.actor, self.critic, self.fs, self.det = actor, critic, float(fire_shift), bool(deterministic)
        self.buckets = sorted(buckets); self.use_graph = use_graph and self.dev.type == "cuda"
        self.g = {}
        if self.use_graph:
            for b in self.buckets:
                self._capture(b)

    def _fwd(self, v, g):
        lg, _ = self.actor(v, g)
        A, L = sample_heads(lg, self.fs, self.det)
        V = self.critic(v, g)[1] if self.critic is not None else torch.zeros_like(L)
        return torch.cat([A.to(torch.float32), L[:, None], V[:, None].float()], -1)

    def _capture(self, b):
        v = torch.zeros(b, 436, device=self.dev); g = torch.zeros(b, 6, 25, 25, device=self.dev, dtype=torch.uint8)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3): self._fwd(v, g)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            out = self._fwd(v, g)
        pin = torch.empty(b, 6, dtype=torch.float32).pin_memory()
        self.g[b] = (graph, v, g, out, pin)

    def refresh(self):
        """re-capture after the weights were replaced by new tensors (load_state_dict copies in place: not needed)"""
        self.g = {}
        if self.use_graph:
            for b in self.buckets: self._capture(b)

    @torch.no_grad()
    def act(self, vec, grid):
        """vec (B, 436) float32, grid (B, 6, 25, 25) uint8 -> actions (B, 4) int64, logp (B,), value (B,)"""
        B = len(vec)
        if B == 0:
            return np.zeros((0, 4), np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32)
        if not self.use_graph or B > self.buckets[-1]:
            v = torch.as_tensor(vec, device=self.dev); g = torch.as_tensor(grid, device=self.dev)
            o = self._fwd(v, g).cpu().numpy()
        else:
            b = next(x for x in self.buckets if x >= B)
            graph, v, g, out, pin = self.g[b]
            v[:B].copy_(torch.from_numpy(vec)); g[:B].copy_(torch.from_numpy(grid))
            if B < b:
                v[B:].zero_(); g[B:].zero_()
            graph.replay()
            pin[:B].copy_(out[:B]); o = pin[:B].numpy().copy()
        return o[:, :4].astype(np.int64), o[:, 4].astype(np.float32), o[:, 5].astype(np.float32)
