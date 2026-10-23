"""T2 data: the native AI plays all 10 Pipers (local_ai) in Solo Showdown on the PC, storm on, each match to one
survivor (or max ticks), state2 dump every 4-tick step + extras, saved per episode in the phone demos' format
(inputs/data/demos/s1b_state_300ep/ep_*.npz: state_bytes, state_offs, map_tiles, map_wh, extras) so the same analyzer
(t2_stats.py) reads both. Instances run asynchronously (no batch barrier). Also logs per-episode reset time and RSS
(T3 evidence)."""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbpc.instance import Instance, InstanceDead
from nbpc import proto as P

ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=10); ap.add_argument("--episodes", type=int, default=300)
ap.add_argument("--seed0", type=int, default=100000); ap.add_argument("--max-ticks", type=int, default=4000)
ap.add_argument("--out", required=True); ap.add_argument("--extra", default=""); ap.add_argument("--max-boots", type=int, default=6)
ap.add_argument("--rec", type=int, default=1, help="1: record every controller call (NBE_F_REC, the phone's demo_record) for BC labels")
ap.add_argument("--intro-ai", type=int, default=1, help="1: the local AI also plays the intro (the phone demos: every episode after a session's first)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
man = open(os.path.join(a.out, "manifest.jsonl"), "a")
json.dump(vars(a), open(os.path.join(a.out, f"config_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w"))

def rss_mb(pid):
    try: return int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) // 1024
    except Exception: return None

insts = [Instance(i, tag=f"col{os.getpid()}", extra_args=a.extra.split()) for i in range(a.n)]
# staggered boot
pending, booting, t0 = list(range(a.n)), [], time.time()
while pending or booting:
    while pending and len(booting) < a.max_boots:
        i = pending.pop(0); insts[i].start(); booting.append(i)
    for i in list(booting):
        if insts[i].ready(): booting.remove(i)
        elif not insts[i].alive(): print("boot failed", i, flush=True); insts[i].kill(); pending.append(i); booting.remove(i)
    time.sleep(0.2)
print(f"{a.n} instances ready in {time.time() - t0:.0f} s", flush=True)
existing = set()
if os.path.exists(os.path.join(a.out, "manifest.jsonl")):
    for l in open(os.path.join(a.out, "manifest.jsonl")):
        try: existing.add(json.loads(l)["seed"])
        except Exception: pass
seeds = [s for s in range(a.seed0, a.seed0 + 100000) if s not in existing][: a.episodes]
NATIVE = 0xffffffff
st = [None] * a.n; done_eps = 0; restarts = 0; t_start = time.time()

def begin(i):
    if not seeds: st[i] = None; return
    seed = seeds.pop(0); inst = insts[i]
    inst.send(P.CMD_RESET, flags=P.F_STATE | P.F_MAP | (P.F_REC if a.rec else 0), seed=seed, native_mask=NATIVE, act_tick=200, gas=1, intro_ai=bool(a.intro_ai))
    st[i] = {"seed": seed, "phase": "reset", "t0": time.time(), "states": [], "offs": [0], "extras": [], "map": None, "t_sent": time.time(), "rec": [], "rec_lost": 0}

def extras_of(r):
    return {"localIndex": r.local_index, "cubes": {str(r.cubes[k][0]): int(r.cubes[k][1]) for k in range(r.ncubes)},
            "jumps": [list(r.jumps[k]) for k in range(r.njumps)], "tick": r.tick, "alive": r.alive, "done": bool(r.done),
            "localRank": r.local_rank, "winner": r.winner, "inGas": [int(r.pl[k].in_gas) for k in range(10)],
            "elim": [[int(r.pl[k].eliminated), int(r.pl[k].elim_rank), int(r.pl[k].elim_tick), int(r.pl[k].elim_gas)] for k in range(10)],
            "dealt": [int(r.pl[k].damage_dealt) for k in range(10)], "taken": [int(r.pl[k].damage_taken) for k in range(10)],
            "gasd": [int(r.pl[k].gas_damage) for k in range(10)], "bombd": [int(r.pl[k].bomb_damage) for k in range(10)],
            "kills": [int(r.pl[k].kills) for k in range(10)], "seed": r.seed_applied}

def save(i):
    s = st[i]; inst = insts[i]
    sb = np.frombuffer(b"".join(s["states"]), np.uint8)
    path = os.path.join(a.out, f"ep_{s['seed']:07d}.npz")
    rec = np.concatenate(s["rec"]) if s["rec"] else np.zeros((0, 8), np.int32)
    np.savez_compressed(path, state_bytes=sb, state_offs=np.array(s["offs"], np.int64), rec=rec,
                        map_tiles=np.array(s["map"][0], np.int64), map_wh=np.array(s["map"][1], np.int32),
                        extras=np.array([json.dumps(e) for e in s["extras"]]))
    row = {"seed": s["seed"], "steps": len(s["states"]), "reset_s": s["reset_s"], "dur_s": round(time.time() - s["t0"], 2),
           "rss_mb": rss_mb(inst.proc.pid), "instance": i, "generation": s["extras"][-1].get("gen"), "file": os.path.basename(path)}
    man.write(json.dumps(row) + "\n"); man.flush()

for i in range(a.n): begin(i)
while any(x is not None for x in st):
    progressed = False
    for i, inst in enumerate(insts):
        s = st[i]
        if s is None: continue
        if not inst.done_seq():
            if not inst.alive() or time.time() - s["t_sent"] > 120:
                print(f"instance {i} died/hung (seed {s['seed']}): restarting", flush=True); restarts += 1
                seeds.insert(0, s["seed"]); inst.close(); inst.start(); inst.wait_ready(); begin(i)
            continue
        progressed = True; r = inst.resp
        if r.status != 0:
            print(f"instance {i} status {r.status} {r.err!r}", flush=True); seeds.insert(0, s["seed"]); begin(i); continue
        if s["phase"] == "reset":
            s["phase"] = "run"; s["reset_s"] = round(time.time() - s["t0"], 3); s["start_tick"] = r.tick
            s["map"] = (inst.map_tiles().tolist(), [r.map_w, r.map_h])
        b = inst.state_bytes(); s["states"].append(b); s["offs"].append(s["offs"][-1] + len(b))
        ex = extras_of(r); ex["gen"] = r.generation; ex["recLost"] = int(r.rec_lost); s["extras"].append(ex)
        if a.rec and r.rec_n:
            s["rec"].append(np.frombuffer(inst.mm, np.int32, r.rec_n * 8, P.OFF_REC).reshape(-1, 8).copy())
        if r.done or r.tick - s["start_tick"] >= a.max_ticks:
            save(i); done_eps += 1
            if done_eps % 25 == 0:
                el = time.time() - t_start
                print(f"{done_eps} episodes in {el:.0f} s ({done_eps / el * 60:.1f}/min), restarts {restarts}", flush=True)
            begin(i); continue
        inst.send(P.CMD_STEP, ticks=4, flags=P.F_STATE | (P.F_REC if a.rec else 0), native_mask=NATIVE, ctl_mask=0); s["t_sent"] = time.time()
    if not progressed:
        time.sleep(0.0002)
for inst in insts: inst.close()
print(f"done: {done_eps} episodes in {time.time() - t_start:.0f} s, restarts {restarts}", flush=True)
