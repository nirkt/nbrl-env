#!/usr/bin/env python3
"""BC v4 demonstrations (pctrain round P1): all 10 Pipers driven by the native AI (optionally with "native + lead" aim for
the players in --lead), every living player is a demonstrator. One sample per (step, living player): the player's FAIR
observation (nbpc.fair: per-observer visibility, research E1/F1) at the START of the step and its action during the step
(s1_build.convert_episode version 3 rules, from the controller-call records NBE_F_REC and the players' positions):
  move  : displacement over the step >= 40 units -> 1 + 16-direction bucket (none during a leap)
  fire  : first record of the step not blocked; super (isSuper 1) wins over attack; aim = whole degrees from the shooter's
          position interpolated to the record's tick to the AI's absolute target (lead-corrected when the lead hook ran);
          super distance bin = ceil(8 d / 2600) - 1 (executed at the bin centre by maenv.action_to_native)
Shards (npz, uncompressed): vec float16 (N, 436), act int16 (N, 4), meta int32 (N, 8) = seed, step, player, x, y, tick,
map W, map H (exact: the grid is rebuilt on the GPU by grid4.GridBuilder), ecell int16 (N, 9) (visible enemy cells),
map.npy (tiles; one map: checked by hash), battles.jsonl (per battle and player: rank, damage, kills, cubes, hits, shots).

  python3 collect_demos4.py --battles 1000 --seed0 1000000 --out ../data/demos4_native
  python3 collect_demos4.py --battles 1000 --seed0 1100000 --lead all --out ../data/demos4_lead
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.instance import Instance
from nbpc import proto as P
from nbpc import fair as FAIR

NV, NG = P.OBS_VEC, P.OBS_GRID
MOVE_MIN = 40.0


def bucket(dx, dy):
    return int(np.round(np.arctan2(dy, dx) / (np.pi / 8))) % 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12); ap.add_argument("--battles", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=1000000); ap.add_argument("--out", required=True)
    ap.add_argument("--lead", default="none", help="none | all | half (players with a random half of the indices per battle)")
    ap.add_argument("--lead-dt", type=float, default=0.0); ap.add_argument("--shard", type=int, default=250000)
    ap.add_argument("--max-ticks", type=int, default=4000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    insts = [Instance(i, tag=f"cd{os.getpid()}", bell=True) for i in range(a.n)]
    for x in insts: x.lead_dt = a.lead_dt; x.bell_spin_us = 100
    pending, booting, t0 = list(range(a.n)), [], time.time()
    while pending or booting:
        while pending and len(booting) < 6:
            i = pending.pop(0); insts[i].start(); booting.append(i)
        for i in list(booting):
            if insts[i].ready(): booting.remove(i)
            elif not insts[i].alive(): insts[i].kill(); pending.append(i); booting.remove(i)
        time.sleep(0.2)
    print(f"{a.n} instances ready in {time.time() - t0:.0f} s", flush=True)
    seeds = list(range(a.seed0, a.seed0 + a.battles)); st = [None] * a.n
    buf = {"vec": [], "act": [], "meta": [], "ecell": []}; nshard = 0; nsamp = 0; map_hash = None
    bfile = open(os.path.join(a.out, "battles.jsonl"), "w"); done_b = 0; t_start = time.time()
    hist = {"move": np.zeros(17, np.int64), "fire": np.zeros(3, np.int64)}

    def flush(final=False):
        nonlocal nshard, buf
        if not buf["vec"] or (not final and sum(len(x) for x in buf["vec"]) < a.shard):
            return
        out = {k: np.concatenate(v) for k, v in buf.items()}
        np.savez(os.path.join(a.out, f"shard_{nshard:03d}.npz"), **out)
        print(f"shard {nshard}: {len(out['vec'])} samples", flush=True)
        nshard += 1; buf = {"vec": [], "act": [], "meta": [], "ecell": []}

    def begin(i):
        if not seeds:
            st[i] = None; return
        sd = seeds.pop(0); inst = insts[i]
        if a.lead == "all": lm = 0x3ff
        elif a.lead == "half": lm = int(sum(1 << int(p) for p in np.random.default_rng(sd).choice(10, 5, replace=False)))
        else: lm = 0
        inst.lead_mask = lm
        inst.send(P.CMD_RESET, flags=P.F_OBS | P.F_MAP | P.F_REC, seed=sd, native_mask=0xffffffff, obs_mask=0x3ff, act_tick=200, gas=1,
                  intro_ai=True)
        st[i] = {"seed": sd, "phase": "reset", "t_sent": time.time(), "prev": None, "k": 0, "lead_mask": lm}

    for i in range(a.n): begin(i)
    while any(x is not None for x in st):
        progressed = False
        for i, inst in enumerate(insts):
            s = st[i]
            if s is None: continue
            if not inst.done_seq():
                if not inst.alive() or time.time() - s["t_sent"] > 120:
                    print(f"instance {i} died/hung (seed {s['seed']}): restarting", flush=True)
                    seeds.insert(0, s["seed"]); inst.close(); inst.start(); inst.wait_ready(); begin(i)
                continue
            progressed = True; r = inst.resp
            if r.status != 0:
                print(f"instance {i} status {r.status} {r.err!r}", flush=True); seeds.insert(0, s["seed"]); begin(i); continue
            W, H = r.map_w or 60, r.map_h or 60
            if s["phase"] == "reset":
                s["phase"] = "run"; s["t0tick"] = r.tick
                if map_hash is None:
                    map_hash = int(r.map_hash); np.save(os.path.join(a.out, "map.npy"), inst.map_tiles()); json.dump({"hash": map_hash, "W": W, "H": H}, open(os.path.join(a.out, "map.json"), "w"))
                elif int(r.map_hash) != map_hash:
                    print(f"WARNING: map hash {r.map_hash} != {map_hash} (seed {s['seed']})", flush=True)
            # current observations (fair) + positions of every player
            alive = [p for p in range(10) if (r.obs_written >> p) & 1]
            cur = {}
            if alive:
                O = np.stack([inst.obs_f32[p * (NV + NG):p * (NV + NG) + NV] for p in alive]).astype(np.float32)
                G = np.zeros((len(alive), 6, 25, 25), np.float32)
                FAIR.fair_inplace(O, G, W, H); EC = FAIR.enemy_cells(O, W, H)
                for j, p in enumerate(alive):
                    cur[p] = (O[j], EC[j], int(r.pl[p].x), int(r.pl[p].y))
            jumping = {int(r.jumps[k][0]) for k in range(r.njumps)}
            if s["prev"] is not None:
                prev, ptick, pjump = s["prev"]
                rec = np.frombuffer(inst.mm, np.int32, r.rec_n * 8, P.OFF_REC).reshape(-1, 8).copy() if r.rec_n else np.zeros((0, 8), np.int32)
                for p, (o, ec, x0, y0) in prev.items():
                    pl = r.pl[p]; cid = int(pl.char_id)
                    x1, y1 = int(pl.x), int(pl.y)
                    act = [0, 0, 0, 0]
                    dx, dy = float(x1 - x0), float(y1 - y0)
                    if cid not in jumping and cid not in pjump and dx * dx + dy * dy >= MOVE_MIN * MOVE_MIN and pl.alive:
                        act[0] = 1 + bucket(dx, dy)
                    for c in rec[(rec[:, 1] == cid) & (rec[:, 2] == 1)]:
                        if c[6] == 1: continue
                        f = min(1.0, max(0.0, (float(c[0]) - ptick) / max(1.0, float(r.tick - ptick))))
                        px, py = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                        tdx, tdy = float(c[3] - px), float(c[4] - py)
                        ang = int(round(np.degrees(np.arctan2(tdy, tdx)))) % 360
                        if c[5] == 1 and act[1] != 2:
                            act[1], act[2] = 2, ang; act[3] = min(7, max(0, int(np.ceil(np.hypot(tdx, tdy) / 2600.0 * 8)) - 1))
                        elif c[5] == 0 and act[1] == 0:
                            act[1], act[2] = 1, ang
                    buf["vec"].append(o[None].astype(np.float16)); buf["act"].append(np.array([act], np.int16))
                    buf["meta"].append(np.array([[s["seed"], s["k"], p, x0, y0, ptick, W, H]], np.int32)); buf["ecell"].append(ec[None])
                    hist["move"][act[0]] += 1; hist["fire"][act[1]] += 1; nsamp += 1
            s["prev"] = (cur, r.tick, jumping); s["k"] += 1
            if r.done or r.tick - s["t0tick"] >= a.max_ticks:
                rows = []
                for p in range(10):
                    pl = r.pl[p]
                    rows.append({"player": p, "rank": int(pl.elim_rank) if pl.eliminated else 1, "lead": int((s["lead_mask"] >> p) & 1)})
                bfile.write(json.dumps({"seed": s["seed"], "ticks": r.tick - s["t0tick"], "steps": s["k"], "lead_mask": s["lead_mask"],
                                        "lead_n": int(r.reserved[5]), "lead_skip": int(r.reserved[6]), "players": rows}) + "\n"); bfile.flush()
                done_b += 1; flush()
                if done_b % 50 == 0:
                    el = time.time() - t_start
                    print(f"{done_b} battles, {nsamp} samples in {el:.0f} s ({done_b / el * 60:.0f} battles/min)", flush=True)
                begin(i); continue
            inst.send(P.CMD_STEP, ticks=4, flags=P.F_OBS | P.F_REC, native_mask=0xffffffff, ctl_mask=0, obs_mask=0x3ff); s["t_sent"] = time.time()
        if not progressed:
            time.sleep(0.0002)
    flush(final=True)
    for x in insts: x.close()
    json.dump({"battles": done_b, "samples": nsamp, "shards": nshard, "seconds": round(time.time() - t_start, 1),
               "move_hist": hist["move"].tolist(), "fire_hist": hist["fire"].tolist(), "map_hash": map_hash},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("done", done_b, "battles", nsamp, "samples", flush=True)


if __name__ == "__main__":
    main()
