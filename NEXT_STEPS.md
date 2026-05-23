# Next Steps — Online SAC Training with Live HIL Loop

## Current State
- HIL Tool ↔ Aimsun ↔ live_rollout.py pipeline is wired and functional
- `HilConnector` talks to HIL Tool exactly like test.cpp (START2/MSGEND protocol)
- False crash detection fixed (negative gap sentinel)
- Dynamic lane count from HIL Tool (`get_total_lanes()`)
- Dynamic lane sync after reset
- `--no-teleport` CLI flag for continuous observation mode
- `--keep-dashboard-alive` flag in both `live_rollout.py` and `train_live.py`
- `train_live.py` wired with SB3 SAC, per-episode reward term logger, `--warmup-steps` flag
- Reward display in `live_rollout.py` corrected to actual keys

---

## Step 1 — Verify the Stack Runs Cleanly ✅ (done)
```bash
cd src/sac
python simulations/live_rollout.py
```

---

## Step 2 — Fix Reward Display in live_rollout.py ✅ (done)
Keys updated: `collision`, `ttc`, `jerk`, `speed_limit`, `lane_start`, `lc_progress`, `lane_success`, `time_penalty`.

---

## Step 3 — Run Online Training

```bash
cd src/sac

# Cold start
python simulations/train_live.py --timesteps 100000

# Warm-start from sim pretrain
python simulations/train_live.py --resume checkpoints/sim_baseline --timesteps 50000

# Keep HIL dashboard alive after training ends
python simulations/train_live.py --timesteps 100000 --keep-dashboard-alive

# Don't teleport on reset (observe car as-is)
python simulations/train_live.py --timesteps 100000 --no-teleport
```

Key flags:
- `--timesteps N`      — total environment steps
- `--warmup-steps N`   — random exploration before first SAC update (default 1000)
- `--checkpoint-dir`   — where to save weights (default `checkpoints/`)
- `--resume path`      — load existing checkpoint zip
- `--no-teleport`      — keep car's current position on reset
- `--keep-dashboard-alive` — idle keepalive after training ends

Checkpoints saved every 5,000 steps as `live_sac_NNNNN_steps.zip`.
Final model saved as `live_final.zip`.

---

## Step 4 — Pretrain in Simulation (optional)
Run `train_sim.py` first to get a `sim_baseline.zip`, then fine-tune live.
Highway-env simulation trains much faster (no wall-clock latency).

```bash
python simulations/train_sim.py --timesteps 200000
python simulations/train_live.py --resume checkpoints/sim_baseline
```

---

## Step 5 — Checkpointing & Logging
- SB3 checkpoint callback fires every 5,000 steps automatically
- Per-episode reward breakdown printed to console by `EpisodeLogger`
- TensorBoard logs written to `runs/live/`
- Optional: `tensorboard --logdir runs/` to visualize

---

## Step 6 — Real-Time Safety Check
The HIL loop runs at ~10 Hz (100 ms/step). SB3 SAC update on CPU takes ~2–5 ms —
safe to update every step. If updates become slow:
- Reduce batch size (default 256)
- Move to GPU: add `device="cuda"` to `SAC(...)` constructor

---

## File Map
```
src/sac/
├── simulations/
│   ├── train_live.py         ← online training vs HIL (SB3 SAC + EpisodeLogger)
│   ├── train_sim.py          ← offline sim pretrain
│   └── live_rollout.py       ← smoke test / scripted policy
├── cda_live/
│   └── cda_hil_connector.py  ← HilConnector — complete
├── lanechange_env.py         ← environment — complete
└── wrappers.py               ← ObsWrapper — use this in training
```

---

## Architecture
```
[Aimsun] ──→ [HIL Tool :7999] ──→ [train_live.py :7998]
                  ↑                       |
                  └───── START2 ──────────┘
                    (ego pos/speed/lane)
```
Run order: **Aimsun → HIL Tool → train_live.py**
