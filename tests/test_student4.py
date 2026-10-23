"""student4 / dagger4 offline checks (synthetic data, no game): shapes, finite losses, gradients reach the encoder,
the privileged side input is used only when keep = 1, distill_kl == 0 for identical logits, the replay + shift +
grid-builder training step runs."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_rocm_fix  # noqa
from student4 import PixelStudent4, distill_kl, aux_loss
import dagger4 as D
from grid4 import GridBuilder
dev = torch.device("cuda"); torch.manual_seed(0); res = {}
net = PixelStudent4().to(dev)
B = 6; px = torch.randint(0, 255, (B, 115, 256, 12), dtype=torch.uint8, device=dev)
vec = torch.randn(B, 436, device=dev); grid = (torch.rand(B, 6, 25, 25, device=dev) > 0.9).to(torch.uint8)
o0 = net(px, vec, grid, torch.zeros(B, device=dev)); o1 = net(px, vec, grid, torch.ones(B, device=dev)); on = net(px)
res["shapes"] = [tuple(x.shape) for x in o0["logits"]] + [tuple(o0["aux"].shape), tuple(o0["scal"].shape)]
res["keep0_equals_pixels_only"] = all(torch.allclose(a, b, atol=1e-5) for a, b in zip(o0["logits"], on["logits"]))
res["keep1_differs"] = not all(torch.allclose(a, b, atol=1e-5) for a, b in zip(o0["logits"], o1["logits"]))
res["kl_self_zero"] = float(distill_kl(o1["logits"], [x.detach() for x in o1["logits"]]).abs().max())
tl = [torch.randn_like(x) for x in o1["logits"]]
maps = (torch.rand(B, 5, 29, 64, device=dev) > 0.7).to(torch.uint8); heat = torch.rand(B, 7, 29, 64, device=dev) * (torch.rand(B, 7, 29, 64, device=dev) > 0.98)
scal = torch.rand(B, 5, device=dev)
lm, lh, ls = aux_loss(o1, maps, heat, scal); kl = distill_kl(o1["logits"], tl).mean()
(kl + lm + lh + ls).backward()
res["losses_finite"] = all(torch.isfinite(t).item() for t in (kl, lm, lh, ls))
res["enc_grad"] = float(net.s1[0].weight.grad.abs().sum()) > 0; res["priv_grad"] = float(net.p_fc[0].weight.grad.abs().sum()) > 0
# one dagger-style training step from the replay
MAP = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "map_showdown.npy"))
rep = D.Replay(64, dev); gb = GridBuilder(MAP, 60, 60, dev); last = None
for t in range(40):
    prev = last or [None, None, None]
    j = rep.add(px[t % B, :, :, -3:], prev, maps[0], (heat[0] * 255).to(torch.uint8), scal[0], torch.randn(388, device=dev).half(), vec[0].half(),
                torch.tensor([9000, 9000, 300], device=dev), torch.full((9,), -1, dtype=torch.int16, device=dev))
    last = [j, j, j] if prev[0] is None else prev[1:] + [j]
idx = rep.sample(16); fr = rep.stack(idx); m2, h2 = rep.maps[idx], rep.heat[idx].float() / 255
fr, m2, h2 = D.shift(fr, m2, h2, 2)
g2 = gb.build(rep.meta[idx, 0], rep.meta[idx, 1], rep.meta[idx, 2], rep.ecell[idx])
o = net(fr, rep.vec[idx].float(), g2, torch.ones(len(idx), device=dev))
l = distill_kl(o["logits"], torch.split(rep.tlog[idx].float(), [17, 3, 360, 8], -1)).mean() + sum(aux_loss(o, m2, h2, rep.scal[idx]))
l.backward(); res["replay_step_loss"] = round(float(l), 3); res["params_M"] = round(sum(p.numel() for p in net.parameters()) / 1e6, 2)
res["PASS"] = res["keep0_equals_pixels_only"] and res["keep1_differs"] and res["kl_self_zero"] < 1e-4 and res["losses_finite"] and res["enc_grad"] and res["priv_grad"]
print(res)
