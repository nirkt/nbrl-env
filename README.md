# rl/ — RL system for the Pyla-RL bot (handoff, current state only)

Plan (approved 2026-11-11): `~/.claude/plans/can-you-plan-a-clever-summit.md`. Showdown first; one policy with
behaviour sliders + per-brawler adapters; self-play with up to all 10 slots as agents; Claude launches runs
(owner's go before each long run).

**Decision log: `HISTORY.md`** (kept locally, not in the git repo). Every decision, run and result
gets an entry there (date, decision, why, numbers) in the same session it happens. This README stays current-state.

## Look here first (before any new reverse engineering or tooling)
The v5 work and nirkt's repo already answer most libg / game-data questions. Grep these before disassembling,
probing or writing a new helper; reading them costs minutes, re-finding costs hours:
- `out/libg_emu/logic_map.md`: libg 69 logic map (players, skills, scripted input, gas, loadout cards, what is
  "not found yet"). Offsets differ from libg 68 but the structure carries over (e.g. skill slots, super charge).
- `out/v5/HANDOFF.md` (history: grep `HANDOFF_history.md`): nb68 host findings (hypercharge meter, gadget crashes,
  bad gadgets, team jobs, maps), dataset / label pipeline, models.
- `tools/libg_qemu/`: `battle.py` (scripted battles on the libg 69 server build, ~93 ticks/s: slow but complete
  logic), `classes.py` (seg / entity class rules; `obs_v5._tile_lut` copies its seg rule), `tables.py`.
- `reference/nirkt/bulletlabels/docs/OPEN_ISSUES.md`, `native/README.md`, comments at the top of `loader/env.c` and
  `spawnfix.c`: what the libg 68 client build lacks (server-only code is stubbed) and how parts were re-implemented.
- `out/v5/classes.md` / `classes.json`: class tables. Game CSVs: `nbpc.diversity._rows(name)` / `table(name)`.
If the answer is "not found" there too, say so and ask the owner before a long search (see Pitfalls: gadgets).

## Working cheaply (tokens) without losing quality
- Grep with a narrow pattern and `head_limit`; read only the lines you need. Never print whole CSV rows, state
  dumps or disassembly: select the columns / addresses first.
- Shell output: end commands with `| tail -n` or `| grep -i "error\|built"` (the host build prints pages of
  warnings); `cut -c1-200` on source greps (env.c has very long lines).
- One run per question, and batch questions into it: boot is 60-90 s, so put the self-check, the live check and
  the benchmark in one command. Design a probe so one boot answers several cases (`~/nb68/rl/hyp_kind.py`).
- Every new module carries an assert self-check (`python -m obs_v5`, `python -m env_v5 --check`); run those before
  any live run, and let live runs assert too (teams, shapes) instead of printing data to read.
- Time-box unknowns: two probes without a lead -> stop, write the finding under Pitfalls, ask the owner.
- Keep this README current-state only; replace lines instead of appending history.

## How to run (WSL distro Ubuntu-24.04)
Always pass scripts on stdin; `wsl ... bash -lc '...$VAR...'` loses the quotes and expands `$VAR` to nothing:

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -- bash -s <<'EOF'
    source /mnt/c/<path to this folder>/env.sh; cd ~/nb68/rl
    python3 -m env_v5 --n 12 --minutes 2 --agents 10
    EOF

- `env.sh`: points nirkt's `MAEnv` at our libg 68 host (`~/nb68/bulletlabels/native/loader/nbhost_maps`), sets
  `NB_PIX_CAP`, PYTHONPATH (our `vendor/nbpc` first), activates `~/nb68/venv`. Keep it LF (CRLF breaks every variable).
- `env_v5.py`: `heroes()` (115 playable rows), `locations()` (25 Showdown maps), `make_battle_fn(agents)` (random
  brawlers / map / 0-2 mates for player 0), `Env` (MAEnv + crash log), benchmark `main`. `skip.json` (optional)
  lists brawlers / locations to leave out. Generated maps: `use_genmaps(k)` (call before `Env`) boots instances
  0..k-1 from `~/nb68/rl/genmaps/r*/game` (12 layouts, `framegen/genmap.py OUT --seed N`, one layout per instance);
  `make_battle_fn(gen=..., p_gen=0.5)` plays location `Survival612` there = the generated layout (tag `loc` = `r1xx`).
  A layout with a PayloadTrack tile (`b`) crashes the boot (rc 139): `use_genmaps` skips those.
- `live_check.py --n 2 --minutes 6 --gen 1`: one boot checks team reward, area effects and generated maps (asserts).
- `obs_v5.py`: `ObsV5().build(obs["vec"], obs["grid"], env.last_meta, ids)` -> `self` (B, 8), `ent` (B, 39, 16),
  `grid` (B, 25, 25) seg class ids with UNKNOWN outside the camera view; per-episode noise (jitter, dropout, false
  positives, seg tile errors, 1-2 decisions latency; the `NOISE` ranges are guesses until `measure_noise.py`);
  `cond(brawler, sliders, team_size)` (26,). `python -m obs_v5` = self-check, `--live` adds 300 real decisions
  (passes; 20 us per decision at batch 10; also checks the host's teams against the requested groups). Classes:
  enemy / teammate / their summons, projectiles per side, power_box, power_cube, showdown bomb; tiles by the
  `classes.py` seg rule; hypercharge ready, area effects per side (no radius /
  kind). Missing: gadget ready (always 0), faded_bush / power_box
  tiles, box size. Noisy path only: entity velocity = differenced noisy positions (0 on first sight / after a drop,
  matched to the last decision by class within 1 tile), tiles under the HUD controls UNKNOWN (`hud_mask`, an
  approximate screen-to-tile scale). The bot-side adapter (perception + tracking -> same arrays) and the
  sim-vs-labels test are not written.
- Adapter contract (what the bot-side adapter must do so the real game matches `ObsV5`): (1) drop detections
  outside the sim window |dx| < 4500, -3500 < dy < 2900 world units (a 16:9 frame likely shows more sideways than the
  sim; not measured); (2) seg `ui` -> UNKNOWN, `faded_bush` -> `bush`, `power_box` tiles -> `floor`; (3) area effects
  as one class per side (own + mates = teammate), position only; (4) slot order is free (the policy max-pools entities); (5) HP fractions
  and ammo from the bar pixels, not from the detector.
- Host = `~/nb68/bulletlabels/native/loader/nbhost_rl` (rl round; `nbhost_maps` is left as it was). Build:
  `cd ~/nb68/bulletlabels/native/loader && sed 's/nbhost_spawnfix2/nbhost_rl/g' build.sh | sh`. Originals of every
  file changed: `*.orig_rl` (env.c, encode.c, vendor/nbpc/instance.py, fair.py). encode.c + instance.py as a patch
  against bulletlabels 52184b2: `host/bulletlabels_rl.patch` (`git apply` in `~/nb68/bulletlabels`). Changes: `div["team_groups"]`
  (group 0-7 per player, 0 = alone); state2 enemy rows now include summons, feature 14 = summon, 15 = on my team
  (were: last hit by me, is Piper); self 18 = hypercharge fraction (was gas step); grid channel 0 = tiles.csv
  row + 1 (was wall 0/1); `Player.reserved` = hypercharge, its max, team; fair.py never bush-hides teammates;
  item feature 4 = 1 bomb or enemy area effect (data 17xxxxxx, already in the items dump), -1 my team's area effect
  (was: bomb only; area effects were 0); instance.py `APK_BY_IDX` = per-instance game dir.
  Anything of nirkt's that reads those old fields is wrong on this host. Test-only actions: fire 100 + k = raw
  input kind k, fire 200 + i = skill slot i (`~/nb68/rl/hyp_kind.py`).
- Benchmark on `nbhost_rl`: n=4 19.8k decisions/s, 211 battles, 0 restarts.
- Slider rewards (`env_v5.py` `slider_terms`, `Env`): every player gets random weights 0-2 per battle; transitions
  carry reward + `s_<slider>` terms, episode rows `sliders` + summed terms; `env.cond(ids)` = (B, 26) policy
  conditioning. `python -m env_v5 --check` = self-check (passes). Live, random agents, mean per episode: bush_camp
  0.36, collect 1.32, protect -2.66 (before its mate-damage penalty was limited to mates within 1800), retreat 0.01, kite 0.24, peek 0.23, focus 0.18, bush_control 0.06; push was
  2.72 and is now scaled / 5; third_party was 0.0 and its rule was loosened afterwards (both changes unit-tested,
  not re-run live). Scales are untuned: tune with the plan's slider test (weight 2.0 vs 0.0 must move its metric).
  Cost: n=4 drops 19.8k -> 13.6k decisions/s (Python per env step).
- Run outputs: `~/nb68/rl/` (WSL).

## Done (stage 0, part 1)
- `maenv.py`, `fair.py`, `diversity.py` copied from `reference/nirkt/nbrl-env/py/nbpc` into
  `~/nb68/bulletlabels/vendor/nbpc/` (same protocol structs; ours adds `team_mask`, `map_location`, `hyper_fill`).
- Benchmark, random actions, all 10 slots as agents, `ticks=2`, no rendering:
  n=4: 21.9k decisions/s; n=12: 34.7k decisions/s, 735 battles, **0 restarts**, ~1.3 GB RAM per instance (16 GB at 12).
  nirkt's whole teacher run was 24 M decisions = ~12 min of sim at this rate, so the policy (GPU), not the sim, sets
  the training time.
- Setup check (`~/nb68/rl/chk_div.py`): brawler / skin / location / theme applied in 12 of 12 battles. 4 of 12 report
  `map_vs_csv` false (Survival642, 6102, 682): live tiles differ from maps.csv; not looked into (obs uses live tiles).

- Training python = `~/yolo_env/bin/python` (torch 2.5.1 with GPU support: MLP 436-512-512-64, batch 120 =
  0.12 ms; numpy 2.5.2). Run it by full path after `source env.sh` (which activates the numpy-only `~/nb68/venv`):
  it picks up our vendor nbpc and benchmarks the same (n=4: 21.4k decisions/s, 0 restarts). Do not install into
  `yolo_env` (owner's env).

## Pitfalls
- MAEnv retries a dead boot forever without a message. If a run prints nothing, boot one `Instance` with
  `log_dir=.` and read `game_*.log` (seen: `NB_PIX_CAP` mismatch -> "too small").
- The old `team_mask` (player 0 + mates) never took effect in MAEnv: teams were only applied on the client-mode
  build path, not the sim-only reset MAEnv uses. Fixed in `nbhost_rl` (`apply_teams` on both paths).
- Gadgets cannot be triggered in this libg (client build): the activation 0xb27c58 is an empty stub and a player
  with a gadget card has no skill slot for it (slots 2 / 3 empty). Hypercharge activation not found: input kinds
  3-21 (zero and non-zero x / y, no skill data) and skill slots 2-5 all leave a full meter (player +0x98) full.
  So the policy's fire head is none / attack / super for now. The v5 libg 69 harness (`tools/libg_qemu`) is the
  server build and may have both, but runs ~93 ticks/s.
- A heredoc inside the `wsl ... bash -s <<'EOF'` call halves backslashes (`\\n` arrives as a newline): build C
  strings with `chr(92)` when patching from Python.
- `train/train_loop.py` of nbrl-env puts its own `py/` first on sys.path (its nbpc, not ours).
- Boot is ~60-90 s per start.

## Stage 1 (BC + PPO sanity, Piper, one map)
- `model.py` `Policy`: entity set encoder + seg-grid CNN + self / cond vector -> GRU (256) -> move 17 / fire 3 /
  pointer aim 360 / distance 8 + value (one network; critic sees the same fair obs). `python -m model` = self-check.
  No per-brawler adapter yet (stage 2).
- `env_v5.Env.record`: battles with cfg `native_ctl` put the native AI's action into the transition terms (`_act`,
  nirkt's collect_demos4 rules: move = displacement bucket, fire = first unblocked controller record).
- `train_ppo.py`: updates 1..`--bc-updates` (40) = BC on all 10 native players (cross-entropy + value), then PPO
  (k = 1-5 agents vs native, KL anchor to the BC snapshot). GRU trained on chunks of `--seq` 8 steps from the stored
  hidden state. Slider rewards off (`--slider-scale 0`), ObsV5 noise on. Eval inside the run every 30 min (seeds
  1..200, player 0 vs 9 native) -> `evals.jsonl`. Gate: rank <= 3.0 at `--eval-episodes 500`.
  `--smoke` = 2 instances, 3 tiny updates (plumbing check). `--resume ckpt_last.pt` continues a run.
- Run: `~/yolo_env/bin/python -m train_ppo --run s1 --hours 3` (after `source env.sh`; start it with `nohup ... &`
  inside `bash -s` so it survives the wsl call). Outputs `~/nb68/rl/runs/<run>/`: `progress.jsonl`, `evals.jsonl`,
  `episodes.jsonl`, `replays/`, `bc.pt`, `ckpt_u*.pt`, `ckpt_last.pt`, `train.log`.

## Dashboard
`viz.html` (copied into every run folder as `index.html`; reads the run's files every 5 s): eval / training rank,
win rate, return, damage, kills, hit rate, survival, storm, action histograms, optimiser health, eval table, and a
battle replay (one battle of instance 0 per update: all players on the map + the focus agent's 25 x 25 policy view
with the noisy detections and its move / aim). Serve: `.claude/launch.json` `rl-dashboard` (preview_start), or
`cd ~/nb68/rl/runs && python3 -m http.server 8765` in WSL, then http://localhost:8765/<run>/.
Replay limits: the map is the battle-start tiles (destroyed walls stay), the storm shows only in the policy view.

## Stage 1 results (2026-11-12)
Eval = player 0 vs 9 native, 200 episodes, ci95 ~ +-0.3.
- `s1` (defaults, 3 h, 535 updates, 17.5 M agent steps): rank 6.17 after BC (u40) -> 3.65 (u124) -> 3.18 (u311)
  -> best 3.13 (u411, win 25 %, top3 69 %) -> 3.19 (u514). Flat from u311 on: stalled just above the gate. Healthy
  (0 restarts, expl_var 0.76, storm deaths 1 %); kl_bc 0.31 with beta pinned at the 0.02 `--kl-floor`.
- `s1b` (`--bc-updates 120`, else identical; stopped at u223 / 59 min): rank 6.08 after BC (u120, no better than
  s1's 6.17: BC had already plateaued) -> 3.95 (u213). Lower entropy (1.1 vs 1.7). Behind s1 at equal wall time
  (s1: 3.65 at 43 min). Conclusion: more BC is not the lever.
- Launching from Windows: `wsl -d Ubuntu-24.04` (default distro is a different Ubuntu without ~/nb68). A
  `bash -s` heredoc launch from Git Bash silently started nothing; working way = a script file, `tr -d '\r'`, then
  `setsid nohup ... < /dev/null &` inside it.

## Stage 1.5: self-play league (2026-11-12, still Piper / one map)
Why: s1 only met the native AI and was KL-anchored to its play, so it could not get past "native AI plus a bit".
- `train_ppo.py` (originals: `*.orig_s1`): `--sp 0.8` of PPO battles are self-play, all 10 players on policies, 0-7
  of them played by a league snapshot (`runs/<run>/league/u*.pt`, every `--league-every` 20 updates, 16 kept with
  random eviction, one drawn per update; reloaded on `--resume`); 20 % stay k agents vs native. Only learner rows
  train. `progress.jsonl` `rank_sp` = learner rank in self-play battles (5.5 = equal to the league mix).
- Critic gets `priv` (true offset / hp / alive of all 10 players) through a zero-initialised branch (`model.py` `vp`):
  s1 checkpoints load (strict=False, fresh Adam).
- Evals play a frozen copy: `rank` vs 9 native (sanity only) and `b_rank` vs 9 copies of `--base` (default: the
  resumed checkpoint): 5.5 = equal, lower = stronger than base. This is the strength measure to watch.
- `--kl-floor 0` on a resumed run = no BC anchor (beta is 0 past u100).
- Run `s2` = `--resume runs/s2/ckpt_last.pt --base runs/s1/ckpt_last.pt --kl-floor 0 --eval-episodes 200 --eval-min 60 --hours 6`
  (pid in `runs/s2/pid`; stop: `kill $(cat runs/s2/pid)`, it saves `ckpt_last.pt`; continue with `--resume`).
- Stage 1 gate on s1 `ckpt_last.pt` (500 episodes, frozen weights): rank 3.19 +-0.17, win 17 %, top3 67 %: not
  passed (<= 3.0). Same weights vs 9 copies of themselves: b_rank 5.3 +-0.39 (5.5 expected), so evalb is sane.
- Evals take every instance until done: 500 + 200 episodes = 13 min with ~no training (2 updates). Hence 200 + 200
  hourly. Training speed between evals: ~2.8k learner decisions/s, ~15 s per update (0 restarts).
- s2 after 210 updates (79 min, u749): b_rank 5.33 -> 4.19 +-0.41 (win 10 % -> 20 %, top3 33 % -> 55 %): clearly
  stronger than s1. Rank vs native 3.24 -> 3.67 +-0.35 (kills 0.98 -> 0.73, hit rate 0.23 -> 0.19): slightly worse.
  Entropy rose 1.7 -> 2.1 over the same time. Watch the later rows of `runs/s2/evals.jsonl`: if native rank keeps
  getting worse, raise the native share (`--sp 0.6`) or bring back a small anchor before blaming the league.
- u959 (148 min): b_rank 3.90 +-0.45 (win 34 %), native rank 4.14 +-0.37 (win 10 %): still stronger head to head,
  worse vs native twice in a row = forgetting the native AI. Restarted at u~960 with `--sp 0.6` (40 % native
  battles), no anchor, 3.5 h; the weights before the change are `runs/s2/ckpt_sp80_end.pt`.
- u1148 (70 min after the change): native rank 3.99 -> 3.16 +-0.25 (back at s1's level), b_rank 3.49 -> 3.58
  +-0.43 (win 38 %): `--sp 0.6` fixed the forgetting without costing head-to-head strength. Keep 0.6.
- s2 ended at u1525 (50.0 M agent steps, 0 errors). Last eval u1333: native rank 3.12 +-0.26 (gate 3.0 still not
  passed), b_rank 3.57 +-0.44, win 41 % vs s1 copies. b_rank flat for the last 2 evals (3.49 / 3.58 / 3.57): the
  league has stopped pushing. u1333-u1525 (the last hour) was never evaluated: `ckpt_last.pt` is untested;
  `ckpt_u1333.pt` is the last measured one.

## Timeline (set 2026-11-12; dates assume the v5 dataset run + refill end on 13 Nov)
Vision comes first: the noise measurement and the sim-vs-labels test need the trained vision models and change what
the policy sees. Vision training and RL training share one GPU, so they run one after the other. The dataset /
vision half of this timeline is in `out/v5/HANDOFF.md` ("Timeline").

| When | Step | Time |
| --- | --- | --- |
| 14-15 Nov, while the GPU trains vision | Host work (rebuild + booted game, CPU only): area effects in the state dump, teammate respawn, generated maps; short live test of the 2026-11-12 team-reward and HUD changes | 1-2 days of work |
| 16 Nov | `measure_noise.py`: trained vision models on the val split -> replace the guessed `NOISE` ranges in `obs_v5.py` | half a day |
| 16 Nov | Sim-vs-labels test (also replaces the estimated HUD mask, settles the view window) | half a day |
| 17 Nov | Resume `s3` from `runs/s3/ckpt_last.pt` on the fixed observation; confirm it recovers its old eval numbers (owner's go first) | 6 h run |
| 17 Nov on | Stage 2 generalist: all brawlers, all maps, teams, sliders on | 2-3 weeks (plan) |
| with stage 2 | Bot-side adapter (vision output -> ObsV5 arrays), then the first live Trio Showdown test | 3-4 days (plan) |

Checkpoints of s1-s3 are also kept read-only in `~/nb68/rl/backup_2026-11-12`. Code is on GitHub: `nirkt/nbrl-env`
branch `rl-v5` (this folder is the git repo).

## Next
0. **Gas as its own grid channel: NOT DOING unless the owner says otherwise (2026-11-12, judged a small gain).** The
   idea, kept for reference: the v5 seg model will give two answers per
   pixel: what stands there (19 classes; a wall / bush in the gas keeps its class) and a separate "in the gas yes/no"
   map (`out/v5/HANDOFF.md`, Models). The observation has to match: tile class in one grid channel, gas in another,
   so "bush + gas" (cover that hurts) differs from "bush" (safe). Today: gas = host grid channel 3, and it replaces
   `floor` only (`obs_v5.py` `from_state2`; a wall / bush in the gas keeps its class), the same rule as the v5 seg labels.
1. **Run `s3` is paused (stopped 2026-11-12 16:28 at u1626 on the owner's request, to free the machine for another
   job; continue with `--resume runs/s3/ckpt_last.pt`, owner's go first).** Read `runs/s3/evals.jsonl`:
   `b_rank` is now vs 9 copies of `s2/ckpt_u1333.pt` (5.5 = equal; s2 ended at ~3.5 vs s1, so do not compare the
   two numbers). If `b_rank` drops well under 5.5: set `--base` to the best s3 checkpoint and continue. If it stays
   flat: add exploiters (a second learner trained only to beat this one; not built).
   Command: `--run s3 --resume runs/s2/ckpt_last.pt --base runs/s2/ckpt_u1333.pt --sp 0.6 --kl-floor 0
   --eval-episodes 200 --eval-min 60 --eval-n 12 --hours 6` (restart: `--resume runs/s3/ckpt_last.pt`), with s2's 16 league snapshots copied into `runs/s3/league/`.
   New in `train_ppo.py` (original: `train_ppo.py.orig_s2`):
   - PFSP: 3 league snapshots loaded at once (`--opp-slots`), a self-play battle is pinned to one; one slot is
     redrawn every 2 updates (`--opp-hold`). Draw weight = ((10 - r) / 9) ^ 2, r = running mean rank of the
     snapshot's players against the learner (`league/pfsp.json`, `pfsp` in `progress.jsonl`: [r, rows]; lower r =
     it beats the learner more = drawn more). Only battles whose snapshot stayed loaded to the end are scored.
     A full pool evicts the snapshot with the worst r (was: random).
   - `--eval-n` = instances that play the evals. Keep it at `--n` (12): with 4, the first 37 min of s3 gave 43
     updates at 425-1158 learner decisions/s (s2: 2.2k) and the eval was not done after 40 min. The code default
     is now all instances.
2. Stage 2 (generalist): per-brawler adapter in `model.py`, `make_battle_fn` rosters / maps / teams, sliders on.
   Team placement is in `Env` (team rank = best member rank; a battle cut short by end "agents" ranks a living native
   mate by the truncation estimate). Open host-side work (not done; each needs a host rebuild + a live test when the
   machine is free): teammate respawn in duo / trio (absent: Solo underneath; needs RE + an episode change, owner's
   call: HISTORY 49); trio
   is 3 teams + 1 solo (10 players, real mode 12); enemy rows share 9 slots between players and summons (overflow
   is silent); the sim-vs-labels
   observation test is not written.
3. Speed: up to 6 small forward passes per poll now (learner / 3 league slots / eval / base); batch them or CUDA graphs if
   decisions/s is the limit.
4. "Better than human" cannot be shown in the sim: it needs the bot-side adapter (perception -> ObsV5 arrays, not
   written) and live games. No gadgets / hypercharge in this libg (Pitfalls).
Not tested yet: team battles at length (friendly fire, native AI targeting a teammate), summons in a live battle
(unit-tested only), n > 12, native-AI mix, crash rate with a
real policy (random agents die early, ~1100 ticks per episode).
