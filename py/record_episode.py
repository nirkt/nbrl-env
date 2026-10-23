#!/usr/bin/env python3
"""Record eval episodes in client mode (real client + renderer) for watching: player 0 = the policy (state:CKPT acts on
the fair state, pixel:CKPT on pixels, native = the native AI), 9 native bots, the given eval seeds. Writes per episode a
frame sheet (every --every-th 256x115 observation, x2, annotated with step, HP, the action) and a JSON trace.
  python3 record_episode.py --policy state:../runs/bc4_native_s0/bc_best.pt --seeds 3,7 --out ../runs/rec/bc4"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.maenv import MAEnv
from eval_suite import load_policy, PixelPolicy

MOVES = ["-"] + [f"{int(k * 22.5)}" for k in range(16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True); ap.add_argument("--seeds", default="3,7"); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    kind = a.policy.split(":")[0]
    pol = load_policy(a.policy)[0] if kind == "state" else (PixelPolicy(a.policy.split(":", 1)[1]) if kind == "pixel" else None)
    seeds = [int(s) for s in a.seeds.split(",")]; queue = list(seeds); issued = []
    def bf(rng, i):   # player 0 = an agent (native: observed but driven by its own AI, native_ctl)
        if not queue: return None
        sd = queue.pop(0); issued.append(sd)
        return {"seed": sd, "ctl": [0], "end": "player0", "tag": "rec", "native_ctl": pol is None}
    env = MAEnv(len(seeds), bf, pixels=True, tag=f"rec{os.getpid()}", verbose=True)
    traces = {}; frames = {}
    while True:
        obs, ids, tr = env.poll(min_ready=1, max_wait=0.01)
        if len(ids):
            if kind == "state":
                act, _, _ = pol.act(obs["vec"], obs["grid"])
            elif kind == "pixel":
                act = pol.act(obs["pixels"])
            else:
                act = np.zeros((len(ids), 4), np.int64)
            for b, aid in enumerate(ids):
                aid = int(aid); st = env.last_state[b]; meta = env.last_meta[b]
                traces.setdefault(aid, []).append({"x": int(meta[0]), "y": int(meta[1]), "tick": int(meta[2]), "act": act[b].tolist(),
                                                    "hp": float(obs["vec"][b, 0])})
                k = len(traces[aid]) - 1
                if k % a.every == 0:
                    frames.setdefault(aid, []).append((k, obs["pixels"][b, :, :, -3:].copy(), float(obs["vec"][b, 0]), act[b].tolist()))
            env.send(ids, act)
        if all(s.state == "idle" for s in env.I):
            break
    env.close()
    per = 4 * 6   # frames per sheet: 4 columns x 6 rows
    for aid, fr in frames.items():
        seed = issued[(aid >> 4) - 1] if 0 < (aid >> 4) <= len(issued) else aid
        for c0 in range(0, len(fr), per):
            part = fr[c0:c0 + per]; rws = (len(part) + 3) // 4
            sheet = Image.new("RGB", (4 * 512, rws * 232), (20, 20, 20))
            for j, (k, f, hp, ac) in enumerate(part):
                im = Image.fromarray(f).resize((512, 230), Image.NEAREST); d = ImageDraw.Draw(im)
                d.text((4, 4), f"step {k} hp {hp:.2f} mv {MOVES[ac[0]]} fire {ac[1]} aim {ac[2]}", fill=(255, 255, 0))
                sheet.paste(im, ((j % 4) * 512, (j // 4) * 232))
            sheet.save(os.path.join(a.out, f"seed{seed}_sheet{c0 // per:02d}.png"))
        json.dump({"seed": seed, "trace": traces[aid]}, open(os.path.join(a.out, f"seed{seed}_trace.json"), "w"))
    json.dump([r for r in env.episodes], open(os.path.join(a.out, "episodes.json"), "w"), indent=1)
    print(json.dumps([{k: r[k] for k in ("seed", "rank", "ticks", "steps")} for r in env.episodes]))


if __name__ == "__main__":
    main()
