#!/usr/bin/env python3
"""Template for a training loop. Copy this file to train/<your_name>.py and put your learning code where marked.

It plays Piper (player 0) against 9 native AI through the environment (nbpc.MAEnv), collects each step's observation,
action and reward, and reports the rank of every finished game. The random policy in example_policy.py is the starting
point; replace it with your own model and update.

    python3 train/train_loop.py --episodes 20 --n 4
"""
import argparse, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nbpc.maenv import MAEnv                 # noqa: E402
from example_policy import RandomPolicy      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20, help="games to play")
    ap.add_argument("--n", type=int, default=4, help="game instances (processes); each one runs one game at a time")
    ap.add_argument("--ticks", type=int, default=4, help="game ticks per decision (4 = 5 decisions per second)")
    ap.add_argument("--seed0", type=int, default=10_000_000, help="training seeds start here (evaluation uses 1..500)")
    a = ap.parse_args()

    seeds = iter(range(a.seed0, a.seed0 + 10 ** 9))

    def battle_fn(rng, i):
        # one game per call; "ctl": the players whose actions we send; "end": stop when player 0 is out
        return {"seed": next(seeds), "ctl": [0], "end": "player0", "tag": "train"}

    env = MAEnv(a.n, battle_fn, ticks=a.ticks, fair=True, max_ticks=4000, verbose=False)
    policy = RandomPolicy(seed=0)
    t0 = time.time()
    try:
        while len(env.episodes) < a.episodes:
            obs, ids, transitions = env.poll(min_ready=1, max_wait=0.003)
            if len(ids):
                actions, _ = policy.act(obs["vec"], obs["grid"], ids)
                env.send(ids, actions)
            # ---- learning goes here ------------------------------------------------------------------------
            # transitions: [(agent_id, reward, done, terms, rank), ...] for the steps that just finished.
            # Store (obs, actions, reward) per agent id and update your model when you have enough of them.
            # ------------------------------------------------------------------------------------------------
    finally:
        env.close()
    ranks = np.array([r["rank"] for r in env.episodes if r.get("player") == 0 and r.get("rank") is not None], float)
    print(f"{len(ranks)} games, mean rank {ranks.mean():.2f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
