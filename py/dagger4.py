#!/usr/bin/env python3
"""P2: online DAgger distillation of the state teacher into the pixel student (student4.PixelStudent4), client mode.

Env: nbpc.maenv pixels=True (the real client + nbgl per instance), Solo Showdown 1 vs 9 native: player 0 is driven by
the mixture policy (teacher with probability beta, else the student; per step = DAgger's pi_i = beta pi* + (1-beta) pi_hat),
training seeds >= 20,000,000 (never the eval seeds 1..500).
Every step of player 0 stores: its newest 256x115 frame (the 4-frame stack is rebuilt from links to the same agent's
previous frames), the teacher's four head logits on the FAIR state (the distillation target), the fair state (privileged
side input: vector + exact grid metadata), and the understanding labels rendered from the state through the camera
projection (learn/labels.py at 64x29: 5 ground maps, 7 entity heat maps, 5 HUD scalars).
GPU replay: the most recent --buffer samples (FIFO). Training (between collection chunks): loss = KL(teacher || student)
(masked like the policy) + map BCE + heat BCE (positives x21) + scalar MSE, random +-2-cell shifts of frames and maps.
The ramp (design): beta (who acts) 1 -> 0 linearly over the first --beta-frac of the run; the privileged side input is
dropped per sample with probability p rising 0 -> 1 over the first --priv-frac; then pixels only. The student must
therefore learn to act from pixels while it still gets the teacher's state features early, and it is trained on states it
visits itself (DAgger) from the start.
Metrics per chunk on the FRESH chunk before training on it (held out): move agreement, attack-probability correlation with
the teacher, mean P(attack) student / teacher, enemy-on-screen statistics of the teacher's attacks (fairness check).

  python3 dagger4.py --teacher ../runs/ppo4_x/ckpt_best.pt --run dag4_a --hours 3
"""
import argparse, copy, csv, json, os, sys, time, signal
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "learn"))
import torch_rocm_fix  # noqa: F401
import proj, labels as LB
import state2 as S2
from models4 import make_state_policy
from policy_rt import sample_heads
from grid4 import GridBuilder
from student4 import PixelStudent4, distill_kl, aux_loss, MAP_K, ENT_K, SCAL_K, SCAL_SCALE, AUX_W, AUX_H
from nbpc.maenv import MAEnv

NV = 436
BAR_L, BAR_R = int(np.ceil(240 * AUX_W / 2400)), int(2160 * AUX_W / 2400)   # = labels.map_labels' off-map side bars (7, 57)


# the Showdown map (fixed: the same hash 3654183511 in all 1,000 demo battles; data/map_showdown.json)
import numpy as _np, os as _os
MAP_TILES = _np.load(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "map_showdown.npy"))

def render_labels(sbytes, cam, pidx, mapinfo, P):
    """state2 bytes + camera -> maps (5,29,64) uint8, heat (7,29,64) uint8 (x255), scal (5,) f32, enemy-on-screen flag"""
    v = S2.StateView(sbytes, {}, mapinfo, None)
    ml, hm, pts, sc = LB.render_all(v, pidx, cam, AUX_W, AUX_H, P, sigma=1.0)
    maps = np.stack([ml[k] for k in MAP_K]).astype(np.uint8)
    heat = (np.stack([hm[k] for k in ENT_K]) * 255).astype(np.uint8)
    # objects the client keeps just outside the drawn view (in the black side bars) are not visible: no entity labels there
    heat[:, :, :BAR_L] = 0; heat[:, :, BAR_R:] = 0
    scal = np.array([sc[k] * SCAL_SCALE[k] for k in SCAL_K], np.float32)
    n_en = sum(1 for (u, vv, _) in pts["enemy"] if BAR_L <= u < BAR_R and 0 <= vv < AUX_H)
    return maps, heat, scal, n_en


class Replay:
    """FIFO replay in HOST RAM (pctrain 2026-10-05: the GPU is the binding resource in client mode, ~1.7 GB VRAM per
    rendering instance): frames + labels + teacher logits + the privileged state as numpy arrays; batches are gathered on
    the CPU and copied to the GPU. Frame stacks = links to the same agent's previous 3 slots; a sample is valid while all
    its linked slots still hold the frames it was written with (generation check)."""
    def __init__(self, cap, dev):
        self.cap, self.dev, self.n, self.w = cap, dev, 0, 0
        z = lambda *s, dt=np.uint8: np.zeros(s, dt)
        self.frame = z(cap, 115, 256, 3); self.maps = z(cap, 5, AUX_H, AUX_W); self.heat = z(cap, 7, AUX_H, AUX_W)
        self.scal = z(cap, 5, dt=np.float32); self.tlog = z(cap, 388, dt=np.float16); self.vec = z(cap, NV, dt=np.float16)
        self.meta = z(cap, 3, dt=np.int64); self.ecell = z(cap, 9, dt=np.int16)
        self.ref = np.zeros((cap, 3), np.int64); self.gen = np.zeros(cap, np.int64); self.refgen = np.zeros((cap, 3), np.int64); self.g = 0

    def add(self, frame, prev_slots, maps, heat, scal, tlog, vec, meta, ecell):
        j = self.w; self.g += 1
        self.frame[j] = frame; self.gen[j] = self.g
        refs = [(p if p is not None else j) for p in prev_slots]
        self.ref[j] = refs; self.refgen[j] = [self.gen[r] for r in refs]
        self.maps[j] = maps; self.heat[j] = heat; self.scal[j] = scal; self.tlog[j] = tlog; self.vec[j] = vec
        self.meta[j] = meta; self.ecell[j] = ecell
        self.w = (j + 1) % self.cap; self.n = min(self.n + 1, self.cap)
        return j

    def sample(self, B):
        idx = np.random.randint(0, self.n, B * 2)
        ok = (self.gen[self.ref[idx]] == self.refgen[idx]).all(-1)
        return idx[ok][:B]

    def stack(self, idx):
        r = self.ref[idx]
        fr = np.concatenate([self.frame[r[:, 0]], self.frame[r[:, 1]], self.frame[r[:, 2]], self.frame[idx]], -1)
        return torch.from_numpy(fr).to(self.dev, non_blocking=True)

    def get(self, name, idx):
        return torch.from_numpy(getattr(self, name)[idx]).to(self.dev, non_blocking=True)

    def gather(self, idx, pin):
        """batch -> GPU: the 4-frame stacks via a multi-threaded index_select into a pinned staging buffer, the rest numpy"""
        if not hasattr(self, "_ft"):
            self._ft = torch.from_numpy(self.frame)
        B = len(idx); r = self.ref[idx]
        flat = torch.from_numpy(np.concatenate([r[:, 0], r[:, 1], r[:, 2], idx]).astype(np.int64))
        st = pin[:4 * B]; torch.index_select(self._ft, 0, flat, out=st)
        fr = st.to(self.dev, non_blocking=True).view(4, B, 115, 256, 3).permute(1, 2, 3, 0, 4).reshape(B, 115, 256, 12)
        g = {k: torch.from_numpy(getattr(self, k)[idx]).to(self.dev, non_blocking=True) for k in ("maps", "heat", "scal", "tlog", "vec", "meta", "ecell")}
        return fr, g


def shift(fr, maps, heat, k):
    dy, dx = np.random.randint(-k, k + 1, size=2)
    def sh(t, ay, ax, py, px):
        t = torch.roll(t, shifts=(int(py), int(px)), dims=(ay, ax))
        if py > 0: t.narrow(ay, 0, int(py)).zero_()
        if py < 0: t.narrow(ay, t.shape[ay] + int(py), -int(py)).zero_()
        if px > 0: t.narrow(ax, 0, int(px)).zero_()
        if px < 0: t.narrow(ax, t.shape[ax] + int(px), -int(px)).zero_()
        return t
    return sh(fr, 1, 2, dy * 4, dx * 4), sh(maps, 2, 3, dy, dx), sh(heat, 2, 3, dy, dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--run", required=True); ap.add_argument("--init", default=None)
    ap.add_argument("--hours", type=float, default=3.0); ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--buffer", type=int, default=40000); ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--reuse", type=float, default=4.0); ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--beta-frac", type=float, default=0.35)
    ap.add_argument("--priv-frac", type=float, default=0.5); ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--w-map", type=float, default=1.0); ap.add_argument("--w-heat", type=float, default=1.0)
    ap.add_argument("--w-scal", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=0, help="schedule length in env steps (0: hours * 900,000)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", a.run); os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    dev = torch.device("cuda"); P = proj.load_params()
    tck = torch.load(a.teacher, map_location="cpu", weights_only=False)
    teacher = make_state_policy(tck, dev).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    stu = PixelStudent4().to(dev)
    if a.init:
        stu.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)["model"])
    else:   # privileged branch warm-started from the teacher's state encoders
        stu.p_grid.load_state_dict(teacher.grid.state_dict()); stu.p_gfc.load_state_dict(teacher.gfc.state_dict()); stu.p_vfc.load_state_dict(teacher.vfc.state_dict())
    opt = torch.optim.AdamW(stu.parameters(), lr=a.lr, weight_decay=1e-5)
    rng = np.random.default_rng(a.seed + 11)
    total = a.total_steps or int(a.hours * 900_000)
    battle_fn = lambda r, i: {"seed": int(rng.integers(20_000_000, 2 ** 31 - 1)), "ctl": [0], "end": "player0", "tag": "dagger"}
    env = MAEnv(a.n, battle_fn, fair=True, tag=f"dag{os.getpid()}", verbose=True, pixels=True, seed=a.seed)
    mapinfo = S2.MapInfo(60, 60, MAP_TILES.tolist())
    gb = GridBuilder(MAP_TILES, 60, 60, dev)
    rep = Replay(a.buffer, dev)
    last_slots = {}          # aid -> last 3 replay slots (oldest first)
    stop = {"now": False}; signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    t0 = time.time(); steps = 0; chunk_rows = []; prog = None; ep_rows = []; nseen = 0; upd = 0
    log = open(os.path.join(out, "episodes.jsonl"), "a")
    import threading, copy as _copy
    actor = _copy.deepcopy(stu).eval()                   # the acting copy (refreshed from the training weights every 25 steps)
    wlock, rlock = threading.Lock(), threading.Lock()   # acting-copy weights / replay
    sh = {"pdrop": 0.0, "collected": 0, "trained": 0, "stop": False, "L": [], "err": None, "t_busy": 0.0}
    pin = torch.empty(4 * a.batch, 115, 256, 3, dtype=torch.uint8).pin_memory()

    def trainer():
        try:
            stu.train()
            while not sh["stop"]:
                due = int(a.reuse * sh["collected"] / a.batch) - sh["trained"]
                if due <= 0 or rep.n < 4 * a.batch:
                    time.sleep(0.005); continue
                tb = time.time()
                with rlock:
                    idx = rep.sample(a.batch)
                    if len(idx) < a.batch // 2: continue
                    fr, g = rep.gather(idx, pin)
                maps, heat = g["maps"], g["heat"].float() / 255.0
                if a.aug: fr, maps, heat = shift(fr, maps, heat, a.aug)
                mt = g["meta"]; grid = gb.build(mt[:, 0], mt[:, 1], mt[:, 2], g["ecell"])
                keep = (torch.rand(len(idx), device=dev) > sh["pdrop"]).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = stu(fr, g["vec"].float(), grid, keep)
                tl = torch.split(g["tlog"].float(), [17, 3, 360, 8], -1)
                kl = distill_kl(o["logits"], tl).mean()
                lm, lh, ls = aux_loss(o, maps, heat, g["scal"])
                loss = kl + a.w_map * lm + a.w_heat * lh + a.w_scal * ls
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(stu.parameters(), 1.0); opt.step()
                sh["L"].append([kl.item(), lm.item(), lh.item(), ls.item()]); sh["trained"] += 1; sh["t_busy"] += time.time() - tb
                if sh["trained"] % 25 == 0:
                    with wlock: actor.load_state_dict(stu.state_dict())
        except Exception as e:   # surfaced by the collection loop
            sh["err"] = e

    th = threading.Thread(target=trainer, daemon=True); th.start()
    c_stat = None; c0 = time.time(); got = 0
    try:
        while time.time() - t0 < a.hours * 3600 and not stop["now"]:
            if sh["err"] is not None: raise sh["err"]
            frac = steps / max(1, total)
            beta = max(0.0, 1.0 - frac / a.beta_frac) if a.beta_frac > 0 else 0.0
            pdrop = min(1.0, frac / a.priv_frac) if a.priv_frac > 0 else 1.0
            sh["pdrop"] = pdrop
            if c_stat is None:
                c_stat = {"t_fire": [], "s_fire": [], "agree": [], "n_en": [], "t_att": []}; c0 = time.time(); got = 0; tr0 = sh["trained"]; busy0 = sh["t_busy"]
            obs, ids, tr = env.poll(min_ready=max(1, a.n // 2), max_wait=0.006)
            if len(ids):
                B = len(ids)
                with torch.no_grad():
                    vf = torch.as_tensor(obs["vec"], device=dev); gf = torch.as_tensor(obs["grid"], device=dev)
                    tl, _ = teacher(vf, gf); tA, _ = sample_heads(tl)
                    px = torch.as_tensor(obs["pixels"], device=dev)
                    keep = torch.as_tensor((rng.random(B) > pdrop).astype(np.float32), device=dev)
                    with wlock, torch.autocast("cuda", dtype=torch.bfloat16):
                        so = actor(px, vf, gf, keep)
                    sl = [x.float() for x in so["logits"]]; sA, _ = sample_heads(sl)
                    use_t = torch.as_tensor(rng.random(B) < beta, device=dev)
                    act = torch.where(use_t[:, None], tA, sA).cpu().numpy()
                    tcat = torch.cat([x.float() for x in tl], -1).half().cpu().numpy()
                    tp = torch.softmax(tl[1].float(), -1)[:, 1].cpu().numpy(); sp = torch.softmax(sl[1], -1)[:, 1].cpu().numpy()
                    agree = (sl[0].argmax(-1) == tl[0].argmax(-1)).float().cpu().numpy(); tatt = (tA[:, 1] == 1).cpu().numpy()
                meta_t = env.last_meta[:, :3].astype(np.int64); ec_t = env.last_ecell
                vf16 = obs["vec"].astype(np.float16); px_np = obs["pixels"]
                env.send(ids, act)
                for b in range(B):
                    sb, cam, li, p = env.last_state[b]
                    if cam is None: continue
                    maps, heat, scal, n_en = render_labels(sb, cam, p, mapinfo, P)
                    aid = int(ids[b]); prev = last_slots.get(aid, [None, None, None])
                    with rlock:
                        j = rep.add(px_np[b, :, :, -3:], prev, maps, heat, scal, tcat[b], vf16[b], meta_t[b], ec_t[b])
                    # the env's stack: the episode's first frame tiled x4, then shifted -> refs [j0, j0, j0] for frame 2
                    last_slots[aid] = [j, j, j] if prev[0] is None else prev[1:] + [j]
                    c_stat["t_fire"].append(float(tp[b])); c_stat["s_fire"].append(float(sp[b])); c_stat["agree"].append(float(agree[b]))
                    c_stat["n_en"].append(n_en); c_stat["t_att"].append(int(tatt[b]))
                    got += 1; steps += 1; sh["collected"] += 1
                new = env.episodes[nseen:]; nseen = len(env.episodes)
                for r in new:
                    ep_rows.append(r); log.write(json.dumps({**r, "beta": round(beta, 3), "pdrop": round(pdrop, 3)}) + "\n")
                if len(last_slots) > 4096:
                    live = set(int(x) for x in ids); last_slots = {k: v for k, v in last_slots.items() if k in live}
            if got < a.chunk:
                continue
            # ---------------- log one chunk
            t_col = time.time() - c0; upd += 1
            L, sh["L"] = sh["L"], []
            Lm = np.mean(L, 0) if L else np.zeros(4)
            tf, sf, ne, ta = (np.array(c_stat[k]) for k in ("t_fire", "s_fire", "n_en", "t_att"))
            rec = [r for r in ep_rows[-100:] if r["rank"] is not None and not r["truncated"]]
            row = {"chunk": upd, "env_steps": steps, "minutes": round((time.time() - t0) / 60, 2), "beta": round(beta, 3), "pdrop": round(pdrop, 3),
                   "steps_per_s_collect": round(got / max(1e-9, t_col), 1), "train_steps": sh["trained"] - tr0,
                   "train_busy_frac": round((sh["t_busy"] - busy0) / max(1e-9, t_col), 3), "train_lag_steps": int(a.reuse * sh["collected"] / a.batch) - sh["trained"],
                   "kl": round(float(Lm[0]), 4), "map_bce": round(float(Lm[1]), 4), "heat_bce": round(float(Lm[2]), 4), "scal_mse": round(float(Lm[3]), 4),
                   "move_agree": round(float(np.mean(c_stat["agree"])), 3) if c_stat["agree"] else None,
                   "fire_corr": round(float(np.corrcoef(tf, sf)[0, 1]), 3) if len(tf) > 10 and tf.std() > 0 and sf.std() > 0 else None,
                   "p_att_teacher": round(float(tf.mean()), 4) if len(tf) else None, "p_att_student": round(float(sf.mean()), 4) if len(sf) else None,
                   "teacher_att_no_enemy_on_screen": round(float((ne[ta == 1] == 0).mean()), 3) if (ta == 1).any() else None,
                   "teacher_p_att_enemy_vis": round(float(tf[ne > 0].mean()), 4) if (ne > 0).any() else None,
                   "teacher_p_att_no_enemy": round(float(tf[ne == 0].mean()), 4) if (ne == 0).any() else None,
                   "rank_recent": round(float(np.mean([r["rank"] for r in rec])), 3) if rec else None, "restarts": env.stats["restarts"],
                   "buffer": rep.n, "min_avail_mb": env.stats.get("min_avail_mb"), "max_vram_mb": env.stats.get("max_vram_mb")}
            if prog is None:
                progf = open(os.path.join(out, "progress.csv"), "a"); prog = csv.DictWriter(progf, fieldnames=list(row)); prog.writeheader()
            prog.writerow(row); progf.flush(); print(json.dumps(row), flush=True); log.flush()
            c_stat = None
            if upd % 10 == 0:
                with wlock: sd = {k: v.clone() for k, v in actor.state_dict().items()}
                torch.save({"model": sd, "env_steps": steps, "chunk": upd, "teacher": a.teacher}, os.path.join(out, "student_last.pt"))
            if upd % 100 == 0:
                with wlock: sd = {k: v.clone() for k, v in actor.state_dict().items()}
                torch.save({"model": sd, "env_steps": steps, "chunk": upd, "teacher": a.teacher}, os.path.join(out, f"student_c{upd:05d}.pt"))
    finally:
        sh["stop"] = True; th.join(timeout=30)
        torch.save({"model": stu.state_dict(), "env_steps": steps, "chunk": upd, "teacher": a.teacher}, os.path.join(out, "student_last.pt"))
        env.close()

if __name__ == "__main__":
    main()
