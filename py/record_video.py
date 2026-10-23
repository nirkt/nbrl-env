#!/usr/bin/env python3
"""MP4 of the pixel student playing (pixels only), with what it DECODES drawn over what it SEES: the 256x115 observation
(x4), its predicted ground classes (wall red, water blue, bush green, storm purple), its entity heat-map peaks (self white,
enemy red, enemy bullet orange, own bullet cyan, box yellow, power cube magenta), its aim arrow when it attacks, and its
decoded HUD (HP, ammo, super, brawlers left). Plays episodes (1 vs 9 native, client mode) until --wins winning games are
recorded (or --max-episodes); the best other game is kept as a fallback. One video frame per env step (0.2 s of game
time) at 5 fps = real time.
  python3 record_video.py --ckpt ../runs/dag4_d/student_last.pt --out ../runs/video --wins 1"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_rocm_fix  # noqa: F401
from nbpc.maenv import MAEnv
from student4 import PixelStudent4, MAP_K, ENT_K, AUX_W, AUX_H
from policy_rt import sample_heads

GC = {"wall": (255, 40, 40), "water": (40, 110, 255), "bush": (0, 220, 0), "gas": (170, 0, 255)}
EC = [(255, 255, 255), (255, 30, 30), (255, 150, 0), (0, 255, 255), (255, 235, 0), (255, 0, 255), (0, 0, 0)]
SC = 4; W, H = 256 * SC, 115 * SC; BAR = 28


def render(fr, maps, peaks, hud, act, step, fire_p):
    big = np.array(Image.fromarray(fr).resize((W, H), Image.NEAREST), np.float32)
    for i, k in enumerate(MAP_K):
        if k not in GC: continue
        m = np.array(Image.fromarray(maps[i].astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        big[m] = 0.7 * big[m] + 0.3 * np.array(GC[k])
    im = Image.new("RGB", (W, H + BAR), (0, 0, 0)); im.paste(Image.fromarray(big.clip(0, 255).astype(np.uint8)), (0, BAR))
    d = ImageDraw.Draw(im); cw, ch = W / AUX_W, H / AUX_H
    for e, pts in peaks.items():
        r = 12 if ENT_K[e] in ("self", "enemy", "box") else 8
        for (y, x) in pts:
            cx, cy = (x + 0.5) * cw, BAR + (y + 0.5) * ch
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=EC[e], width=3)
    if act[1] > 0 and peaks.get(0):
        y, x = peaks[0][0]; cx, cy = (x + 0.5) * cw, BAR + (y + 0.5) * ch; L = 110; a = np.radians(act[2])
        d.line([cx, cy, cx + L * np.cos(a), cy + L * np.sin(a)], fill=(255, 255, 0) if act[1] == 1 else (255, 0, 255), width=4)
    d.text((6, 7), f"t={step * 0.2:5.1f}s  student decodes: HP {hud[0]:.2f}  ammo {hud[1]:.1f}  super {hud[2]:.2f}  brawlers left {hud[3]:.0f}"
           f"   P(attack) {fire_p:.2f}   action: {'ATTACK' if act[1] == 1 else ('SUPER' if act[1] == 2 else '-')}", fill=(255, 255, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--wins", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=60); ap.add_argument("--n", type=int, default=7); ap.add_argument("--seed0", type=int, default=50_000_000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    net = PixelStudent4().to(dev).eval(); net.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["model"])
    queue = list(range(a.seed0, a.seed0 + a.max_episodes)); issued = []
    def bf(rng, i):
        if not queue: return None
        sd = queue.pop(0); issued.append(sd)
        return {"seed": sd, "ctl": [0], "end": "player0", "tag": "video"}
    env = MAEnv(a.n, bf, pixels=True, tag=f"vid{os.getpid()}", verbose=True)
    bufs = {}; done_eps = []; wins = []; nseen = 0; best = None
    try:
        while len(wins) < a.wins:
            obs, ids, tr = env.poll(min_ready=1, max_wait=0.005)
            if len(ids):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = net(torch.as_tensor(obs["pixels"], device=dev))
                lg = [x.float() for x in o["logits"]]; A, _ = sample_heads(lg); A = A.cpu().numpy()
                aux = o["aux"].float(); nm = len(MAP_K)
                maps = (torch.sigmoid(aux[:, :nm]) > 0.5).cpu().numpy(); heat = torch.sigmoid(aux[:, nm:])
                pk = (heat == F.max_pool2d(heat, 3, 1, 1)) & (heat > 0.4); pk = pk.cpu().numpy()
                sc = o["scal"].float(); hud = torch.stack([torch.sigmoid(sc[:, 0]), sc[:, 1] * 3, torch.sigmoid(sc[:, 2]), sc[:, 3] * 10], -1).cpu().numpy()
                fp = torch.softmax(lg[1], -1)[:, 1].cpu().numpy()
                for b, aid in enumerate(ids):
                    peaks = {e: list(zip(*np.nonzero(pk[b, e]))) for e in range(len(ENT_K))}
                    bufs.setdefault(int(aid), []).append((obs["pixels"][b, :, :, -3:].copy(), maps[b], peaks, hud[b], A[b].tolist(), float(fp[b])))
                env.send(ids, A)
            for r in env.episodes[nseen:]:
                aid = next((k for k in bufs if issued[(k >> 4) - 1] == r["seed"]), None) if bufs else None
                fr = bufs.pop(aid, []) if aid is not None else []
                done_eps.append({"seed": r["seed"], "rank": r["rank"], "steps": len(fr)})
                print(f"[video] episode seed {r['seed']}: rank {r['rank']} ({len(fr)} steps); wins so far {len(wins) + (r['rank'] == 1)}", flush=True)
                if r["rank"] == 1: wins.append((r["seed"], fr))
                elif r["rank"] is not None and (best is None or r["rank"] < best[1]): best = (r["seed"], r["rank"], fr)
            nseen = len(env.episodes)
            if all(s.state == "idle" for s in env.I): break
    finally:
        env.close()
    todo = [(sd, 1, fr) for sd, fr in wins] or ([best] if best else [])
    for sd, rk, fr in todo:
        path = os.path.join(a.out, f"student_seed{sd}_rank{rk}.mp4")
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + BAR}", "-r", "5", "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path], stdin=subprocess.PIPE)
        for k, (f, m, p, h, act, fp) in enumerate(fr):
            ff.stdin.write(np.asarray(render(f, m, p, h, act, k, fp)).tobytes())
        end = render(fr[-1][0], fr[-1][1], fr[-1][2], fr[-1][3], fr[-1][4], len(fr) - 1, fr[-1][5]); dd = ImageDraw.Draw(end)
        dd.rectangle([W // 2 - 160, H // 2 - 30, W // 2 + 160, H // 2 + 30], fill=(0, 0, 0)); dd.text((W // 2 - 140, H // 2 - 8), f"FINAL RANK {rk}  (seed {sd})", fill=(255, 255, 0))
        for _ in range(15): ff.stdin.write(np.asarray(end).tobytes())
        ff.stdin.close(); ff.wait(); print("wrote", path, len(fr), "steps", flush=True)
    json.dump(done_eps, open(os.path.join(a.out, "episodes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
