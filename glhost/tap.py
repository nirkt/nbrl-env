#!/usr/bin/env python3
"""Inject touches into a running nbgl/nbhost pair (same path as mouse clicks in the window).
  python3 tap.py /dev/shm/nbgl x y [hold_seconds]      (coordinates in game-surface pixels)"""
import mmap, os, struct, sys, time
path, x, y = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); hold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
fd = os.open(path, os.O_RDWR); m = mmap.mmap(fd, 4096 * 5)
EV_W, EV_BASE, NEV = 64, 80, 1024
def push(action):
    w = struct.unpack_from('<Q', m, EV_W)[0]
    struct.pack_into('<iiii', m, EV_BASE + 16 * (w % NEV), action, x, y, 0)
    struct.pack_into('<Q', m, EV_W, w + 1)
push(0); time.sleep(hold); push(1)
print('tapped', x, y)
