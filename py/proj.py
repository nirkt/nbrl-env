#!/usr/bin/env python3
"""World -> screen projection of this client's battle camera (learn round S0).

Camera anchor (stage units, BattleScreen +0xbf8/+0xbfc): settled value = (0.16*x, 0.12*y - 48) of the followed
character, clamped to the map (device: [548.07, 2331.93] x [140.07, 1983.93] on this map). The anchor in world units
is A = (camx / 0.16, camy / 0.12). A world point p = (x, y, z) relative to A, (X, Y, Z) = p - A, is projected by a tilted
pinhole camera (pitch th from vertical, distance D, focal F in full-res screen pixels, principal point (u0, v0)):
    depth = D - Y sin(th) - Z cos(th)
    u = u0 + F * X / depth
    v = v0 + F * (Y cos(th) - Z sin(th)) / depth
Coordinates u, v are FULL-RES screen pixels (2400 x 1080); divide by 8 for the 300x135 capture, by 2400/256 for obs.
Parameters are fitted by work/fit_proj.py from device calibration captures (device_runs/s0_calib_*).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "proj_params.json")
FULL = (2400, 1080)


def load_params(path=None):
    """expertbot round: the PC-client fit (proj_params_pc.json) when it exists, else the phone fit"""
    if path is None:
        pc = os.path.join(HERE, "proj_params_pc.json")
        path = pc if os.path.exists(pc) and os.environ.get("NB_PROJ", "pc") == "pc" else PARAMS_FILE
    return json.load(open(path))


def anchor_world(cam):
    return cam[0] / 0.16, cam[1] / 0.12


def project(x, y, z, cam, P):
    """world arrays -> (u, v) full-res pixels. cam = (camx, camy) stage units from the agent."""
    ax, ay = anchor_world(cam)
    X = np.asarray(x, np.float64) - ax - P.get("ox", 0.0)
    Y = np.asarray(y, np.float64) - ay - P.get("oy", 0.0)
    Z = np.asarray(z, np.float64)
    th = P["th"]
    d = P["D"] - Y * np.sin(th) - Z * np.cos(th)
    # expertbot round: separate horizontal / vertical focal lengths (PC fit, learn/proj_params_pc.json); F otherwise
    u = P["u0"] + P.get("Fx", P["F"]) * X / d
    v = P["v0"] + P.get("Fy", P["F"]) * (Y * np.cos(th) - Z * np.sin(th)) / d
    return u, v


def to_obs(u, v, obs_size=(256, 115)):
    return np.asarray(u) * obs_size[0] / FULL[0], np.asarray(v) * obs_size[1] / FULL[1]


def unproject(u, v, z, cam, P):
    """full-res pixel (u, v) of a point at height z -> world (x, y); exact inverse of project"""
    th, D = P["th"], P["D"]; Fx, Fy = P.get("Fx", P["F"]), P.get("Fy", P["F"])
    dv = np.asarray(v, np.float64) - P["v0"]; z = np.asarray(z, np.float64)
    Y = (dv * (D - z * np.cos(th)) + Fy * z * np.sin(th)) / (Fy * np.cos(th) + dv * np.sin(th))
    X = (np.asarray(u, np.float64) - P["u0"]) * (D - Y * np.sin(th) - z * np.cos(th)) / Fx
    ax, ay = anchor_world(cam)
    return X + ax + P.get("ox", 0.0), Y + ay + P.get("oy", 0.0)
