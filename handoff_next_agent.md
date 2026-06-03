# Handoff — Next Agent

## Current State

Pipeline is running: **Phase 2 → Phase 3** (unattended, full day).

Phase 2 resumes from `checkpoints/p15_v5/sim_baseline` → saves to `checkpoints/p2_v5`.
Phase 3 resumes from `checkpoints/p2_v5/sim_baseline` → saves to `checkpoints/p3_v3`.

Run was started with:
```
cd src/sac
python simulations/run_pipeline.py
```

---

## Curriculum Structure

| Phase code | Name | Description |
|---|---|---|
| 1 | Phase 1A | Single stopped obstacle in ego lane, adjacent open. LC mechanics. |
| 11 | Phase 1B | Two staggered stopped obstacles. Repeated justified LCs. |
| 15 | Phase 1.5 | 75% follow / 25% escape. Follow vs LC judgment. `lc_progress=0`. |
| 2 | Phase 2 | 10 IDM vehicles, 2–4 lanes via RandomLanesWrapper. |
| 3 | Phase 3 | Full reward suite, 10 IDM vehicles, 2–4 lanes. |

Checkpoint trail:
```
p1a_v4 (90k steps, no sim_baseline — crashed) →
p1b_v1 (sim_baseline ✓) →
p15_v5 (sim_baseline ✓) →
p2_v5 (in progress) →
p3_v3 (in progress)
```

---

## Key Config Values

- `road_length = 2000 m` (increased from 1000 — was causing road-end crashes)
- `initial_spacing = 5` (increased from default 2 — spawn crashes were unavoidable at 2)
- `fwd_ttc_gate = 3.74`, `lc_ttc_gate = 4.0` (all phases except 0)
- `SPEED_LIMIT = 30 m/s`
- `CONSTANT_REW_SCALING = 10.0` (all reward weights divided by this in make_env)
- Phase 2 timesteps: 200k, Phase 3 timesteps: 400k

---

## Eval Commands

```powershell
cd 'c:\Users\janus\Desktop\BerkeleyFileMasterDirectory\PATH\Lane Change!\PATH-SAC-Lane_Change\src\sac'

# Phase 2 final
python simulations/eval_sim.py --model checkpoints/p2_v5/sim_baseline --phase 2 --render --episodes 5

# Phase 3 final
python simulations/eval_sim.py --model checkpoints/p3_v3/sim_baseline --phase 3 --render --episodes 5

# Mid-run check (replace XXXXX with step count from checkpoint dir)
python simulations/eval_sim.py --model checkpoints/p3_v3/sim_sac_XXXXX_steps --phase 3 --render --episodes 3
```

---

## What Healthy Looks Like

**Phase 2:**
- `ep_len_mean` → 300–600 (surviving most of the episode)
- `ep_rew_mean` → 20–50 (positive, occasional lane_success)
- `fwd_interventions/step` < 0.1

**Phase 3:**
- `ep_len_mean` → 400–600
- `ep_rew_mean` → rising trend over 400k steps
- Reward breakdown: `lane_success` dominant, `jerk`/`lat_accel` terms small and shrinking
- `critic_loss` should trend downward — if it stays above 20 the training is unstable

---

## Red Flags

- `ep_len_mean` < 150 at end of Phase 2 → spawn crashes still happening or IDM vehicles too aggressive. Check `initial_spacing`.
- `critic_loss` > 30 at Phase 3 midpoint → instability. Consider `--max-grad-norm 0.5` (already in pipeline).
- `ep_rew_mean` deeply negative throughout Phase 3 → Phase 2 foundation wasn't solid enough. May need to retrain Phase 2 with 300k steps.
- `oscillation_penalty` dominant in Phase 3 reward breakdown → `lc_progress` weight (20) causing premature LCs.

---

## Open Issues / Planned Work

1. **Gap reward redesign**: `_r_fn_gap` currently rewards big gaps → agent learned to stop and collect. Needs 3-zone reward:
   - Gap above upper bound AND below speed limit → penalty
   - Gap in comfortable band → reward
   - Gap below lower bound → penalty
   This is planned but NOT yet implemented. Will require Phase 1.5 retrain.

2. **Target lane pre-assignment**: `find_target_lane()` pre-assigns which lane to go to. Agent learns WHEN to LC but not WHICH lane. Matters in Phase 2/3 where lane quality varies.

3. **Phase 2 timesteps**: 200k may be too short. Previous run (p2_v4) showed `ep_len=137` and high critic loss at Phase 3. If p2_v5/p3_v3 results are similar, extend Phase 2 to 300k.

---

## Key Files

```
src/sac/
  lanechange_env.py      — env, reward terms, obstacle placement, shield
  wrappers.py            — ObsWrapper (6-feature decoder state appended)
  simulations/
    train_sim.py         — PHASE_CONFIGS, make_env(), training loop
    eval_sim.py          — evaluation harness (mirrors training env per phase)
    run_pipeline.py      — chained pipeline runner with skip logic
    brake_viz.py         — manual braking/shield validation tool
  checkpoints/           — all checkpoint zips
```

---

## Previous Run Results (for comparison)

Phase 2/3 previous run (p2_v4 → p3_v2, initial_spacing=2):
- `ep_len_mean = 137` at Phase 3 end
- `ep_rew_mean = -1.29`
- `critic_loss = 32.6`, `actor_loss = 30.1`
- Verdict: crashing frequently, likely spawn crashes from initial_spacing=2
