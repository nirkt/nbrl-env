# nbrl-env

A reinforcement-learning environment for Null's Brawl, Solo Showdown: Piper (player 0) against 9 native AI players.
The game runs offline on this PC. The game's ARM64 code (from the APK) runs under a patched `qemu-aarch64` (`qemu/`),
its graphics are replayed on the local GPU (`glhost/`), and the game host (`loader/`) runs the battle and talks to
Python through shared memory. Nothing uses the network.

The repo has what you need to train a policy from scratch: the environment, evaluation, behavioural cloning and PPO
scripts, and a template for your own code. It includes one trained model, the PPO state teacher, as a reference.

## Setup

```bash
./setup.sh                                   # packages, game files, emulator, game host, renderer
```

Setup needs Ubuntu or Debian (it uses `apt` and `sudo`) and an internet connection for the qemu source. Rerun with
`./setup.sh --skip-apt` once the packages are installed.

Three large files are not in git (GitHub rejects files over 100 MB):

| file | size | where it goes | how it is obtained |
|---|---|---|---|
| `apk/nb_68.279_release_7c5c5152.apk` | 1.17 GB | `apk/` | put it there, or give its URL: `NB_APK_URL=https://... ./setup.sh` |
| `qemu/src/qemu-8.2.2.tar.xz` | 129 MB | `qemu/src/` | downloaded from qemu.org by `setup.sh` |
| `apk_extract/` | about 1.6 GB | generated | extracted from the APK by `setup.sh` |

The APK is checked by its sha256 (`3c249ea89050220eb32fb7eff0a94f360eb88da19244fa96f51e596bc3dcee43`). A download
with the wrong checksum is rejected.

## Hardware this was built and trained on

- Ryzen 7 5800X (8 cores, 16 threads), 31 GB RAM, RX 9070 XT (16 GB VRAM).
- Linux with user namespaces (every game process runs under `unshare -rn`).
- The renderer uses a pbuffer, so no X display is needed; it does need Mesa with EGL and GLESv2.
- Python 3 with `numpy`, `torch` (CUDA, or ROCm: `import torch_rocm_fix` before using CUDA), `pillow`, and `gymnasium` (optional).
- Disk: about 3 GB for the APK and the extracted game files.

## Layout

| part | where |
|---|---|
| game | `apk/` (the APK), `apk_extract/` (made by `loader/setup_apk.sh`) |
| game host, renderer, emulator | `loader/` (`loader/build.sh`), `glhost/` (`nbgl`), `qemu/bin/qemu-aarch64-nb` |
| environment | `py/nbpc/` (`MAEnv`) |
| eval and training scripts | `py/` |
| your code | `train/` (a template and an example policy) |
| the teacher | `models/ppo_teacher_u365.pt`, `models/ppo_teacher_config.json` |
| map | `data/map_showdown.npy` (the map the training uses) |
| tests | `tests/` (`run_unit_tests.sh`, `test_e2e_smoke.py`) |
| run outputs | `runs/` (created by the scripts) |

## The environment

```python
from nbpc.maenv import MAEnv        # from py/
def battle_fn(rng, i):               # called for every new battle on instance i; return None to leave it idle
    return {"seed": 123, "ctl": [0], "end": "player0", "tag": "mine"}
env = MAEnv(4, battle_fn, ticks=4, fair=True)     # 4 game instances, 4 ticks per decision
while True:
    obs, ids, transitions = env.poll(min_ready=1, max_wait=0.003)   # agents whose instance finished a step
    actions = my_policy(obs["vec"], obs["grid"])                    # (len(ids), 4) int array
    env.send(ids, actions)
    # transitions: [(agent_id, reward, done, terms, rank), ...] for the steps that just finished
    # env.episodes: one row per finished game of a controlled player (seed, player, rank, ticks, return, ...)
```

- `ctl`: the players whose actions you send. `end`: `"player0"` stops a game when player 0 is out, `"done"` plays until
  the battle is over, `"agents"` stops when all controlled players are out.
- `ticks`: game ticks per decision. The game runs 20 ticks per second: `ticks=4` is 5 decisions per second (the
  teacher's setting), `ticks=1` is 20.
- Agent ids are `(battle << 4) | player`; key per-agent memory by that id.
- `fair=True` gives each player only what it can see. Use it for any policy that should play fairly.

### Observations

- `obs["vec"]`: `(B, 436)` float32, egocentric. Your own state (22 numbers: HP, ammo, super, position, ...), then 9
  enemies (16 each), 16 projectiles (12 each), 6 boxes (5 each), 8 items (6 each). Positions are relative to you and
  scaled by 3000; velocities per tick. Layout: the docstring of `py/state2.py` and `py/nbpc/fair.py`.
- `obs["grid"]`: `(B, 6, 25, 25)` uint8: walls, water, bush, storm or off-map, off-map, enemy count around you.
- `obs["pixels"]` (with `pixels=True`): the frames the real client draws, `(B, H, W, 3*frame_stack)`. Set the size with
  `obs_size=(W, H)`; the phone size is `(256, 115)`.
- `env.last_state[row]`: the raw state dump, not fair (every player). For analysis and for teachers.

### Actions (v3)

| column | range | meaning |
|---|---|---|
| move | 0..16 | 0 = stop; 1..16 = joystick direction `(k-1) * 22.5` degrees (x right, y down) |
| fire | 0, 1, 2 | 0 = none; 1 = attack at `aim`; 2 = super (Piper's leap and bombs) at `aim` |
| aim | 0..359 | degrees |
| super distance | 0..7 | leap distance bin: `(d + 0.5) / 8 * 2600` units |

An attack is applied at the start of the next step, and its projectile is born one tick later.

### Rewards

`MAEnv` logs every reward term per step: damage dealt and taken, HP, boxes, cubes, kills, storm damage, placement, win.
The weights are `maenv.REWARD_W`; the PPO script takes overrides with `--reward '{"kill": 2}'`.

## Evaluating a policy

```bash
python3 py/eval_many.py --out runs/eval --pol mine=state:models/ppo_teacher_u365.pt --episodes 500 --n 12 --ticks 4
python3 py/eval_suite.py --policy state:models/ppo_teacher_u365.pt --protocol 1v9 --out runs/eval/teacher
```

Held-out protocol: player 0 is the policy, the other 9 are native AI, seeds 1..500, sampled actions, at most 4000 ticks.
Report the mean rank with its 95 % confidence interval, and the win rate.

Reference numbers on this protocol (500 seeds, pctrain round): native AI 5.56 ± 0.25; PPO teacher 2.86 ± 0.19, 41 % wins.

To evaluate your own policy, make it a `state:` policy (see `py/eval_suite.py:load_policy`) or add a branch in
`py/eval_many.py`. The policy returns actions from `pol.act(obs["vec"], obs["grid"])[0]`.

## Training: the pipeline

The teacher came from three steps, which you can run again:

1. **Demonstrations** (about 9 minutes for 1000 battles, native AI as the teacher):
   ```bash
   cd py && python3 collect_demos4.py --battles 1000 --seed0 1000000 --out ../data/demos4_native
   ```
   Writes shards of (fair observation, action) for every living player, and `battles.jsonl`.
2. **Behavioural cloning** (about 30 minutes on the GPU):
   ```bash
   cd py && python3 train_bc4.py --data ../data/demos4_native --run bc4_native --pointer-lead 0 --epochs 8
   ```
   Writes `runs/bc4_native/bc_best.pt`, the starting point for PPO.
3. **PPO** (the teacher took 3 hours and 24 M agent decisions):
   ```bash
   cd py && python3 train_ppo4.py --init ../runs/bc4_native/bc_best.pt --run ppo_mine --hours 8 --n 12 --delay-lead 1.0
   ```
   Writes checkpoints in `runs/ppo_mine/` and held-out evaluations in `runs/ppo_mine/evals.jsonl`. Stop with
   `kill -TERM <pid>` (the checkpoint is written on exit); `--resume <ckpt>` continues.

PPO trains against native AI: 1 to 5 of the 10 players are policy-controlled per battle. Use training seeds from
10,000,000 up, so training never sees the evaluation games. Compare every model on the same held-out seeds.

## Writing your own training code

Put new code in `train/`:

- `train/example_policy.py`: a random policy. Any object with `act(vec, grid) -> (actions, ...)` works with `train/` and
  with `eval_many.py`.
- `train/train_loop.py`: a loop that plays games through `MAEnv`, with a marked place for your update:
  `python3 train/train_loop.py --episodes 20 --n 4`.

The teacher's network is `py/models4.py` (`StatePolicy`, pointer aim with intercept candidates), trained by
`py/train_ppo4.py`. Copy it for a new architecture, and keep the observation and action formats above.

## The teacher model

- `models/ppo_teacher_u365.pt`: PPO from behavioural cloning, after 365 updates (24 M agent decisions).
- 500 seeds, 1 vs 9 native AI (pctrain round): mean rank 2.86 ± 0.19, 41 % wins.
- 20 games here: mean rank 2.30 ± 1.05, 65 % wins. A small sample.
- Config: `models/ppo_teacher_config.json`. Trained at `ticks=4`.

## Known limits

- **One map:** training and evaluation use Survival612 (`data/map_showdown.npy`). The other maps crash early in the battle
  in this build: some of their data points to online content the APK does not contain.
- **No game window:** the repo has no viewer. Frames are available as `obs["pixels"]`.
