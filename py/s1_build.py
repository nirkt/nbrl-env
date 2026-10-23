#!/usr/bin/env python3
"""S1: convert recorded episodes (s1_collect.py) into BC samples in our action space, one sample per (step, living
player): the egocentric state at the START of the step (what the controller saw when it decided) and the action it took
during the step's 4 ticks.

Action conversion (nb_env action space MultiDiscrete([17, 18, 18, 2]) = move, attack, super, gadget):
  move   : displacement of the brawler over the step; >= MOVE_MIN units -> 1 + 16-direction bucket, else 0 (none).
           During a leap (agent SF job) the move is 0: the leap is not joystick motion.
  attack : first fire record (isSuper 0, not blocked) of the brawler in the step -> 2 + bucket(target - position), else 0;
           v2 (QUICK): a shot whose target is within 500 units of the auto-aim target (the nearest living enemy brawler in
           range, else the nearest box in range) becomes the tap (attack 1 = quick-fire, the client's precise auto-aim)
  super  : first fire record with isSuper 1 -> 2 + bucket(target - position) (aimed; the AI never taps), else 0
           one right-thumb action per step: super wins over attack (like nb_env.action_to_intent)
  gadget : 0 (no gadget in this loadout)
Directions: bucket = round(atan2(dy, dx) / 22.5 deg) mod 16, x right / y down (nb_env / agent DIRS).

  python3 work/s1_build.py work/datasets/s1b --out work/datasets/s1b_bc.npz [--max-episodes N]
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state2 as S

MOVE_MIN = 40.0
QUICK = True          # v2: shots at the auto-aim target become the tap (attack 1), see convert_episode
QUICK_R, RANGE = 500.0, 3000.0


def bucket(dx, dy):
    return int(np.round(np.arctan2(dy, dx) / (np.pi / 8))) % 16


def auto_aim_hit(v, me, c):
    """True if the fire target (c[3], c[4]) is where the client's auto-aim tap would shoot: the nearest living enemy
    brawler within range, or, when no brawler is in range, the nearest box in range (within QUICK_R units of it)."""
    ch = v.s["chars"]; mx, my = float(me[3]), float(me[4])
    best = None
    for pool in (v.brawler, v.box):
        cand = [ch[i] for i in np.nonzero(pool & (ch[:, 6] > 0))[0] if int(ch[i][2]) != int(me[2]) or pool is v.box]
        cand = [q for q in cand if np.hypot(q[3] - mx, q[4] - my) <= RANGE]
        if cand:
            best = min(cand, key=lambda q: np.hypot(q[3] - mx, q[4] - my)); break
    return best is not None and np.hypot(best[3] - c[3], best[4] - c[4]) <= QUICK_R


def load_episode(path):
    d = np.load(path, allow_pickle=False)
    sb = d["state_bytes"].tobytes(); offs = d["state_offs"]
    ex = [json.loads(e) for e in d["extras"]]
    M = S.MapInfo(int(d["map_wh"][0]), int(d["map_wh"][1]), d["map_tiles"].tolist())
    views, prev = [], None
    for k in range(len(offs) - 1):
        v = S.StateView(sb[offs[k]:offs[k + 1]], ex[k], M, prev); views.append(v); prev = v
    out = {"views": views, "rec": d["rec"], "map": M, "extras": ex}
    if "frames" in d.files:
        out["frames"] = d["frames"]; out["frame_steps"] = d["frame_steps"]; out["cams"] = d["cams"]
    if "ai_step" in d.files:
        out["ai_step"] = d["ai_step"]
    return out


def convert_episode(ep, with_stats=False, version=2):
    """-> list of (step, pidx, char_id, action[4]) plus stats. version 3 (free aim): action = [move, fire 0/1 attack/2
    super, aim = the exact shot angle in whole degrees (x right / y down) from the shooter's position at the record's
    tick (interpolated inside the step) to the AI's absolute target, super distance bin = ceil(8*dist/2600)-1]."""
    views, rec = ep["views"], ep["rec"]
    ticks = np.array([v.tick for v in views])
    # records per (step, char): step k covers ticks (ticks[k-1], ticks[k]]
    by = {}
    if len(rec):
        ks = np.searchsorted(ticks, rec[:, 0], side="left")
        for r, k in zip(rec, ks):
            if 1 <= k < len(views):
                by.setdefault((int(k), int(r[1])), []).append(r)
    rows = []
    st = {"fires": 0, "quick": 0, "supers": 0, "blocked": 0, "moves": 0, "none": 0, "samples": 0, "proj_births_matched": 0}
    for k in range(1, len(views)):
        v0, v1 = views[k - 1], views[k]
        pos1 = {int(r[0]): r for r in v1.s["chars"]}
        new_proj = {int(r[0]): r for r in v1.s["proj"]}
        old_proj = {int(r[0]) for r in v0.s["proj"]}
        for i in np.nonzero(v0.brawler & (v0.s["chars"][:, 6] > 0))[0]:
            r0 = v0.s["chars"][i]; cid = int(r0[0]); p = int(r0[2])
            r1 = pos1.get(cid)
            if r1 is None:
                continue
            act = [0, 0, 0, 0]
            t0, t1 = v0.tick, v1.tick
            jumping = v0.jumping(cid) is not None or v1.jumping(cid) is not None
            dx, dy = float(r1[3] - r0[3]), float(r1[4] - r0[4])
            if not jumping and dx * dx + dy * dy >= MOVE_MIN * MOVE_MIN:
                act[0] = 1 + bucket(dx, dy); st["moves"] += 1
            for c in by.get((k, cid), []):
                if c[2] != 1:
                    continue
                if c[6] == 1:
                    st["blocked"] += 1; continue
                tdx, tdy = float(c[3] - r0[3]), float(c[4] - r0[4])
                if version == 3:
                    f = min(1.0, max(0.0, (float(c[0]) - t0) / max(1.0, float(t1 - t0))))
                    px, py = r0[3] + f * (r1[3] - r0[3]), r0[4] + f * (r1[4] - r0[4])
                    tdx, tdy = float(c[3] - px), float(c[4] - py)
                    ang = int(round(np.degrees(np.arctan2(tdy, tdx)))) % 360
                    if c[5] == 1 and act[1] != 2:
                        act[1], act[2] = 2, ang
                        act[3] = min(7, max(0, int(np.ceil(np.hypot(tdx, tdy) / 2600.0 * 8)) - 1)); st["supers"] += 1
                    elif c[5] == 0 and act[1] == 0:
                        act[1], act[2] = 1, ang; st["fires"] += 1
                    continue
                if c[5] == 1 and act[2] == 0:
                    act[2] = 2 + bucket(tdx, tdy); st["supers"] += 1
                elif c[5] == 0 and act[1] == 0:
                    act[1] = 2 + bucket(tdx, tdy); st["fires"] += 1
                    if QUICK and auto_aim_hit(v0, r0, c):
                        act[1] = 1; st["quick"] = st.get("quick", 0) + 1
            if version != 3 and act[2]:
                act[1] = 0
            if act[1]:
                # did a projectile of this brawler appear in the step?
                if any(int(q[8]) == cid and int(q[0]) not in old_proj for q in new_proj.values()):
                    st["proj_births_matched"] += 1
            if act == [0, 0, 0, 0]:
                st["none"] += 1
            rows.append((k, p, cid, act)); st["samples"] += 1
    return rows, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=10 ** 9)
    ap.add_argument("--seeds", default=None, help="only these seeds: a,b-c")
    ap.add_argument("--version", type=int, default=2, help="3 = free-aim action space (models.NVEC3)")
    ap.add_argument("--takeover-only", action="store_true", help="dagger_takeover.py episodes: keep only the local player's AI-driven steps")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "ep_*.npz")))[:a.max_episodes]
    vec_n, grid_shape = S.flat_size()
    V, G, A, META = [], [], [], []
    tot = {}
    for fi, f in enumerate(files):
        seed = int(os.path.basename(f)[3:10])
        ep = load_episode(f)
        rows, st = convert_episode(ep, version=a.version)
        for k2, v in st.items(): tot[k2] = tot.get(k2, 0) + v
        if a.takeover_only:
            li = int((ep["extras"][0] or {}).get("localIndex", 0)); ai = ep.get("ai_step")
            rows = [r for r in rows if r[1] == li and ai is not None and r[0] < len(ai) and ai[r[0]] == 1]
        for (k, p, cid, act) in rows:
            enc = S.encode(ep["views"][k - 1], p)
            if enc is None:
                continue
            vec, grid = S.flatten(enc)
            V.append(vec.astype(np.float16)); G.append(np.minimum(grid, 255).astype(np.uint8)); A.append(act)
            META.append((seed, k, p, cid, ep["views"][k - 1].tick))
        if fi % 10 == 0:
            print(f"{fi+1}/{len(files)} {os.path.basename(f)} samples so far {len(A)} stats {st}", flush=True)
    V = np.stack(V); G = np.stack(G); A = np.array(A, np.int16); META = np.array(META, np.int64)
    nvec = [17, 3, 360, 8] if a.version == 3 else [17, 18, 18, 2]
    heads = ["move", "fire", "aim", "super_dist"] if a.version == 3 else ["move", "attack", "super", "gadget"]
    np.savez(a.out, vec=V, grid=G, act=A, meta=META, nvec=np.array(nvec))
    json.dump({"files": len(files), "samples": int(len(A)), "stats": tot, "vec": int(vec_n), "grid": list(grid_shape), "version": a.version, "nvec": nvec,
               "action_hist": {h: np.bincount(A[:, j], minlength=n).tolist() for j, (h, n) in enumerate(zip(heads, nvec))}},
              open(a.out + ".json", "w"), indent=1)
    print("saved", a.out, V.shape, G.shape, A.shape, tot)


if __name__ == "__main__":
    main()
