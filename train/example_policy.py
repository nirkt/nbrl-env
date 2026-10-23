"""A policy is any object with act(vec, grid) -> (actions, ...): actions is an int array (B, 4), one row per agent.

  vec  (B, 436) float32   the agent's fair observation as a vector (see README, Observations)
  grid (B, 6, 25, 25) uint8  the grid around the agent (see README, Observations)

Action columns (v3): move 0..16 (0 = stop, 1..16 = joystick direction), fire 0 none / 1 attack / 2 super,
aim 0..359 (degrees), super distance 0..7 (only used with fire 2). This file is a random policy: the baseline to beat.
"""
import numpy as np


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, vec, grid, ids=None):
        n = len(vec)
        out = np.zeros((n, 4), np.int64)
        out[:, 0] = self.rng.integers(0, 17, n)
        out[:, 1] = self.rng.integers(0, 3, n)
        out[:, 2] = self.rng.integers(0, 360, n)
        out[:, 3] = self.rng.integers(0, 8, n)
        return out, None
