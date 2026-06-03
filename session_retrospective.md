# Session Retrospective — Curriculum Design (up to Phase 2)

## What We Built

Replaced a broken 2-phase pipeline with a structured 5-phase curriculum based on `phase_1_curriculum_notes.md`.

Old pipeline: Phase 1 (two moving obstacles) → Phase 15 (one obstacle at 13 m/s) → Phase 2 → Phase 3

New pipeline: **1A → 1B → 1.5 → 2 → 3**

---

## Phase 1A — Single Clean Lane Change

**Goal:** Teach the agent the mechanics of one lane change with no ambiguity.

**What worked:**
- One stopped obstacle in ego lane, adjacent lane open. The "correct" behavior is always obvious — see obstacle, LC, hold lane.
- `require_obstacle_for_lc=True` with obstacle at 50–100 m: obstacle is always within 80 m when the LC completes, so the gate fires naturally without blocking exploration.
- `continuous_targets=True` + `osc=5`: episode runs full 600 steps after the LC. Oscillation penalty discourages ping-ponging back.
- `--target-entropy -0.5`: kept policy exploratory long enough to discover the LC reward.

**What did not work:**

*Model collapse (v1):*
- Obstacle at 50–200 m. With ego at 25–30 m/s, starts with obstacle at 50 m were physically impossible (braking distance ~78 m > gap). `ent_coef` collapsed to 0.05 before agent ever LCed. Fixed by raising minimum obstacle distance.

*Immediate LC + early termination (v2/v3):*
- Obstacle moved to 150–300 m. Agent LCed at step 1, episode terminated at step ~50.
- Root cause: `continuous_targets=False` + `duration_after_lane_change=40` terminates after 40 steps in target lane. Agent learned "LC immediately, collect reward, done."
- Fixed by using close obstacles (50–100 m), `continuous_targets=True`.

*Road-end crash (v4):*
- `ep_len=456`, `ep_rew=-46.8`. Agent successfully LCed but crashed into road end ~30 s later.
- `road_length=1000 m` not enough for 40 s at 30 m/s (30 × 40 = 1200 m > 1000 m).
- Fixed: `road_length` → 2000 m.
- Proceeded with 90k-step checkpoint after computer crash rather than retraining.

**Final behavior:** Agent approaches stopped obstacle, LCs cleanly, holds new lane. Full-stop behavior on occasion — acceptable for this phase.

---

## Phase 1B — Repeated Justified Lane Changes

**Goal:** Teach the agent to execute multiple LCs, one per obstacle, without ping-ponging.

**Design:** Two stopped obstacles, one per lane, staggered. `continuous_targets=True`. `osc=10`.

**What worked:**
- Agent transferred LC mechanics from 1A cleanly.
- Two-obstacle layout forced two sequential LCs per episode.
- Oscillation penalty effectively suppressed unnecessary reversals.

**What did not work:**
- Full-stop behavior carried over from 1A (agent stops behind first obstacle before LCing). Acceptable as foundation — Phase 1.5 introduces ACC rewards to address this.

**Final behavior:** Agent LCs around first obstacle, moves to adjacent lane, LCs around second obstacle. Full-stop instead of ACC, but LC decisions are correct.

---

## Phase 1.5 — Follow vs Lane Change Judgment

**Goal:** Teach the agent when a LC is warranted vs when to follow.

**Design:** 75% follow scenarios (both lanes have lead vehicle at similar distance/speed), 25% escape scenarios (ego lane blocked, adjacent open). `lc_progress=0` (no lateral progress reward — LC is not always correct here).

**What worked:**
- The 75/25 mix successfully taught both behaviors. Rendering confirmed genuine following in follow scenarios (agent slowed behind lead vehicle or maintained speed if gap comfortable) and clean LCs in escape scenarios.
- `lc_progress=0` was critical — with it enabled, the agent swerved unnecessarily in follow scenarios.
- `osc=20` (strong oscillation penalty) kept the policy stable.
- ACC rewards (`closing_speed`, `braking_reward`, `gap`) provided meaningful signal in follow scenarios.

**What did not work:**
- `_r_fn_gap` as designed rewards large gaps → agent learns to stop (maximizing gap) rather than following at speed. Agent was stopping on open road to collect gap reward. **This is a known issue, fix is planned but not yet implemented.** The 3-zone gap reward (penalize gap too large without speed limit, reward comfortable band, penalize too close) is the intended replacement.

**Final behavior:** Genuinely follows in follow scenarios. LCs in escape scenarios. Some stopping behavior due to gap reward issue, but overall clean.

---

## Eval Script Fix

`eval_sim.py` was not mirroring training config per phase:
- Used `RandomLanesWrapper` for all phases including 1A/1B/1.5 (should be fixed 2 lanes)
- Did not set `p1a_obstacle`, `p1_obstacles`, `p15_obstacle` config keys (obstacles never spawned during eval)
- Did not set `require_obstacle_for_lc`, `continuous_targets`

Fixed: `eval_sim.py`'s `make_env()` now mirrors `train_sim.py`'s phase-specific config blocks exactly.

---

## Infrastructure Fixes

| Fix | Why |
|---|---|
| `road_length` 1000 → 2000 m | Road-end crashes at end of 40 s episodes |
| `initial_spacing` 2 → 5 | Spawn crashes in Phase 2/3 (vehicles 10 m apart) |
| `--target-entropy -0.5` for Phase 1A | Default -2.0 collapsed `ent_coef` before agent explored LCs |
| `phase_label[11]` added to train_sim.py | `KeyError: 11` on startup |
| `p1a_v4/sim_baseline.zip` manually created | Computer crash left no baseline; copied 90k checkpoint |

---

## Phase 2 — First Run (p2_v4, initial_spacing=2)

**Result:** Not usable.
- `ep_len_mean = 137` at Phase 3 end (~9 s per episode)
- `ep_rew_mean = -1.29`
- `critic_loss = 32.6`, `actor_loss = 30.1`
- Likely cause: spawn crashes from `initial_spacing=2` with 10 IDM vehicles

**Phase 2 rerun (p2_v5, initial_spacing=5):** In progress.

---

## Key Design Decisions That Held Up

- **No Phase 0.** Dropped because LinearVehicles at ~14 m/s + ego at ~25 m/s + 21.6 m gap = physically impossible to avoid crash. ACC taught implicitly in Phase 1.5 via closing_speed and braking_reward.
- **`require_obstacle_for_lc` belongs in 1B and 1.5, not 1A.** In 1A the adjacent lane is always open and any LC is justified. The gate adds friction without benefit.
- **`lc_progress=0` in Phase 1.5.** Generic lateral-progress reward leaks into follow scenarios and teaches unnecessary swerving. Only re-enable when LC is always the correct behavior.
- **Stopped obstacles for 1A and 1B.** Moving obstacles (0–12 m/s) add unnecessary variance. TTC changes with relative speed, making the LC timing ambiguous. Stopped obstacles isolate the LC mechanic cleanly.

---

## Planned Next Steps

1. **Implement 3-zone gap reward** in `_r_fn_gap`. Current version causes stopping behavior.
2. **Evaluate p2_v5/p3_v3** when pipeline completes. Key check: `ep_len_mean` > 300.
3. **Extend Phase 2 to 300k steps** if p2_v5 still shows high critic loss.
4. **Target lane selection** — currently pre-assigned. Agent learns when to LC, not which lane. Relevant for Phase 2/3 with multiple valid target lanes.
