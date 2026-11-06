#!/usr/bin/env python3
"""Pixel labels aligned with an observation frame, from the privileged state (learn round S0).

Maps (H, W) for an image of size (W, H) that covers the full 2400x1080 screen (obs 256x115, capture 300x135, ...):
  ground classes by inverse projection of every pixel centre onto the ground plane (z = 0) - proj.py camera:
    wall, water, bush, gas, offmap   (tile footprints; walls/bushes are drawn with height, the label is the footprint)
  entity heat maps (gaussian at the projected 3D position (x, y, z), sigma in pixels), only for objects the client
  renders (state2 'client' flag): self, enemy, proj_enemy, proj_own, box, cube, bomb
Scalars: hp_frac, ammo (0..3), super_frac, brawlers_left, in_gas.
"""
import numpy as np
import proj
import state2 as S

MAP_NAMES = ["wall", "water", "bush", "gas", "offmap"]
ENT_NAMES = ["self", "enemy", "proj_enemy", "proj_own", "box", "cube", "bomb"]
HUD_BOXES_FULL = [(0, 0, 240, 1080), (2160, 0, 2400, 1080)]     # black side bars (world drawn 16:9 in the middle)


def ground_lookup(W, H, cam, P, full=proj.FULL):
    """world (x, y) on the ground plane of each pixel centre of a WxH image of the full screen."""
    us = (np.arange(W) + 0.5) * full[0] / W
    vs = (np.arange(H) + 0.5) * full[1] / H
    u, v = np.meshgrid(us, vs)
    th, D = P["th"], P["D"]; Fx, Fy = P.get("Fx", P["F"]), P.get("Fy", P["F"])
    dv = v - P["v0"]
    Y = dv * D / (Fy * np.cos(th) + dv * np.sin(th))
    X = (u - P["u0"]) * (D - Y * np.sin(th)) / Fx
    ax, ay = proj.anchor_world(cam)
    return X + ax + P.get("ox", 0.0), Y + ay + P.get("oy", 0.0)


def map_labels(view, cam, W, H, P):
    m = view.m
    gx, gy = ground_lookup(W, H, cam, P)
    tx = np.floor(gx / S.TILE).astype(np.int64); ty = np.floor(gy / S.TILE).astype(np.int64)
    inside = (tx >= 0) & (tx < m.W) & (ty >= 0) & (ty < m.H)
    cx = np.clip(tx, 0, m.W - 1); cy = np.clip(ty, 0, m.H - 1)
    gas = m.gas_mask(view.tick)
    out = {"wall": m.wall[cy, cx] & inside, "water": m.water[cy, cx] & inside, "bush": m.bush[cy, cx] & inside,
           "gas": (gas[cy, cx] & inside), "offmap": ~inside}
    # the black side bars are not world
    for (x0, y0, x1, y1) in HUD_BOXES_FULL:
        c0, c1 = int(x0 * W / proj.FULL[0]), int(np.ceil(x1 * W / proj.FULL[0]))
        for k in out:
            out[k][:, c0:c1] = (k == "offmap")
    return out


def entity_points(view, pidx, cam, P, W, H, client_only=True):
    """{class: [(u, v, id)]} in image pixels (WxH image of the full screen)."""
    s = view.s; ch = s["chars"]
    pts = {k: [] for k in ENT_NAMES}
    sx, sy = W / proj.FULL[0], H / proj.FULL[1]

    def add(k, x, y, z, oid, cpos=None):
        if cpos is not None and cpos[0] != S.NIL:          # the client object's rendered position
            x, y, z = cpos
        u, v = proj.project(x, y, max(0, z), cam, P)
        pts[k].append((float(u) * sx, float(v) * sy, int(oid)))
    for i in range(len(ch)):
        r = ch[i]
        if r[18] != 0 or r[6] <= 0 or (client_only and r[19] != 1):
            continue
        if view.box[i]:
            add("box", r[3], r[4], 0, r[0], r[20:23])
        elif view.brawler[i]:
            add("self" if int(r[2]) == pidx else "enemy", r[3], r[4], max(0, r[5]), r[0], r[20:23])
    for r in s["proj"]:
        if client_only and r[14] != 1:
            continue
        add("proj_own" if int(r[2]) == pidx else "proj_enemy", r[3], r[4], max(0, r[5]), r[0], r[15:18])
    for r in s["items"]:
        if client_only and r[8] != 1:
            continue
        # items are labelled at their ground point (the shadow): their z (bombs in flight up to ~1900) is not drawn
        # as a 3D height on this client's scale (device overlay s0_labels_2/c_leap0), unlike characters/projectiles
        cp = r[9:12].copy(); cp[2] = 0
        if r[1] == S.CUBE_ITEM:
            add("cube", r[3], r[4], 0, r[0], cp)
        elif r[1] in (S.BOMB_ITEM, S.BOMB_BOX_ITEM):
            add("bomb", r[3], r[4], 0, r[0], cp)
    return pts


def heatmaps(pts, W, H, sigma=1.5):
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    out = {}
    for k, lst in pts.items():
        m = np.zeros((H, W), np.float32)
        for (u, v, _) in lst:
            if -3 * sigma <= u < W + 3 * sigma and -3 * sigma <= v < H + 3 * sigma:
                m = np.maximum(m, np.exp(-((xs + 0.5 - u) ** 2 + (ys + 0.5 - v) ** 2) / (2 * sigma * sigma)))
        out[k] = m
    return out


def scalars(view, pidx):
    me = view.player_char(pidx)
    if me is None:
        return {"hp_frac": 0.0, "ammo": 0.0, "super_frac": 0.0, "brawlers_left": len(view.alive_players()), "in_gas": 0.0}
    return {"hp_frac": float(me[6]) / max(1, me[7]), "ammo": max(0, int(me[13])) / 1000.0, "super_frac": view.super_frac(pidx),
            "brawlers_left": len(view.alive_players()),
            "in_gas": float(S.in_gas(me[3], me[4], view.tick, view.m.W, view.m.H))}


def render_all(view, pidx, cam, W, H, P=None, sigma=1.5):
    P = P or proj.load_params()
    ml = map_labels(view, cam, W, H, P)
    pts = entity_points(view, pidx, cam, P, W, H)
    hm = heatmaps(pts, W, H, sigma)
    return ml, hm, pts, scalars(view, pidx)
