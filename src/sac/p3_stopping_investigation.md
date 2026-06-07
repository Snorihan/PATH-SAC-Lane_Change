# Phase 3 Stopping Behavior — Investigation Log

## Problem Statement

The Phase 3 SAC agent stops (parks) instead of driving. The behavior evolved across runs:

- **Early runs (p3_v2, p3_v3)**: Agent stops in lane and waits indefinitely — "stops and farms"  
- **Later runs (p3_v8, p3_v9)**: Agent completes one lane change correctly, then immediately stops and stays stopped for the rest of the episode

The behavior is unsafe and represents a degenerate policy: the agent found that stopping eliminates TTC risk at near-zero cost.

---

## Root Cause Analysis

### Why stopping is rational under the original reward structure

Phase 2 trained with `speed_limit_weight = 0` — the agent learned that stopping is **free** (no speed penalty, no time penalty). In Phase 3:

| Signal | Effective weight/step (after ÷10 scaling) | Notes |
|---|---|---|
| `ttc_weight = 80` | −8.0/step when TTC < 3.0s | Large penalty, eliminated by stopping |
| `closing_speed_weight = 15` | −1.5/step | Also eliminated by stopping |
| `braking_reward_weight = 20` | +2.0/step when braking | Goes to 0 when fully stopped |
| `speed_limit_weight = 1` | +0.1/step at full speed, **0 when stopped** | Nearly invisible differential |
| `time_penalty = 0.05` | −0.005/step | Invisible |

The agent's rational calculation: **stop → TTC penalty disappears (save 8/step) → cost is only 0.1/step forgone speed reward**. Net gain = ~8/step. Stopping wins.

### Why it stops specifically after a lane change (later runs)

Once `continuous_targets = False` (original Phase 3), the agent knows the episode ends 50 steps after `got_lane_success`. After the LC, there is nothing left to optimize. Stopping is the safest strategy for the remaining 50 steps.

With `continuous_targets = True` (p3_v8+), the episode continues with a new target, but the agent has **no pressure to pursue it** — no penalty for sitting in the wrong lane. Stopping is still optimal.

---

## Chronological Attempts

### Attempt 1 — Strong speed penalty + remove braking reward (p3_v4)

**Changes:**
- `_r_fn_matching_speed_limit`: changed to return `−1` at stopped, `+1` at speed limit (sign-flip)
- Phase 3 `speed_limit_weight`: 1 → 8 (effective ±0.8/step)
- Phase 3 `braking_reward_weight`: 20 → 0

**Result:** Complete failure. `ep_len = 66`, `ep_rew = −196 ≈ collision penalty`, `ent_coef = 0.998` never moved.

**Why it failed:** Removing `braking_reward` took away the signal that taught safe behavior near vehicles. The p2_v5 policy relied on braking when vehicles were close. With braking now penalized by speed and no braking reward, the agent stopped braking → crashed immediately into IDM vehicles.

---

### Attempt 2 — Moderate version (p3_v5)

**Changes:**
- `braking_reward_weight`: 20 → 10
- `speed_limit_weight`: 8 → 4

**Result:** Same crash pattern. Killed at 25% (101k steps). `ep_len = 72.8`, `ep_rew = −179`, `ent_coef = 1.07`.

**Why it failed:** Still too different from Phase 2's reward landscape. The p2 policy's braking behavior was penalized rather than reinforced.

---

### Lesson from Attempts 1 & 2

Any Phase 3 speed penalty strong enough to overcome the "stop = safe" incentive also breaks the p2 policy's braking behavior. **The stopping problem cannot be solved in Phase 3 alone if Phase 2 never taught the agent that speed has value.**

---

### Attempt 3 — Fix Phase 2: add `speed_limit_weight = 5` (p2_v6 → p3_v6)

**Reasoning:** Phase 2 trained with `speed_limit_weight = 0`. If the agent enters Phase 3 already valuing speed, a moderate Phase 3 speed penalty becomes safe to use.

**Result:** Phase 2 (p2_v6) failed. Agent rarely lane-changed and drove slowly.

**Why it failed:** `speed_limit_weight = 5` (effective 0.5/step) over a 300-step episode = 150 cumulative reward. `lane_success = 200` → effective 20. **Speed reward (150) overwhelmed lane change incentive (20).** Agent learned to just drive fast in one lane.

---

### Attempt 4 — Fix Phase 2: reduce to `speed_limit_weight = 1` (p2_v7)

**Reasoning:** Smaller speed weight so lane_success still dominates.

- At weight=1 (effective 0.1/step): over 300 steps = 30 cumulative vs lane_success 20 + lc_progress (~60). LC is more rewarding than staying in lane.

**Phase 2 result:** Agent lane-changes actively, prefers ACC behavior (follows lead vehicle before committing to LC). Acceptable Phase 2 behavior. ✓

**Phase 3 result (p3_v7, original single-LC config):** Agent still stops after LC. Stopping problem persists.

---

### Attempt 5 — Multi-LC episodes: `continuous_targets = True`, `max_lane_changes = 3` (p3_v8)

**Reasoning:** After a successful LC, immediately assign the next adjacent lane as the new target. The agent can never be "done" until 3 LCs are completed. Stopping between LCs means never reaching the termination condition (3 LCs), so the episode runs to 600-step truncation — but during those 600 steps the agent has to survive.

**Implementation:**
- New `successful_lc_count` counter in `_reset_episode_state`
- `_check_lane_change_termination` increments counter and terminates at N
- Phase 3 `make_env`: `continuous_targets = True`, `max_lane_changes = 3`
- Default `max_lane_changes = 0` (unlimited) preserves Phases 1/11/15 behavior

**Result:** Agent completes 1 LC correctly, then **full stops**. Never pursues LC 2 or LC 3.

**Why it failed:** After LC 1 completes and a new target is assigned, there is no penalty for sitting in the wrong lane without starting the next LC. `lc_progress` only rewards lateral movement (positive signal when moving), but nothing penalizes NOT moving. The agent rationally parks.

---

### Attempt 6 — Add `lane_keeping_penalty_when_requested = 3.0` (p3_v9 — current)

**Reasoning:** `_r_fn_lane_keeping_when_requested` returns 1.0 when the agent is (a) NOT in the target lane AND (b) has not started a lane change. With `sign = −1`, this is a per-step penalty for exactly the "parked in wrong lane" behavior.

At weight=3.0 (effective 0.3/step), over 100 steps waiting for a gap: −30 accumulated. Net from LC after 100-step wait = lane_success(50) + lc_progress(~60) − lane_keeping(−30) = +80. Still positive, so the agent is not forced into reckless LCs.

**Result (as of latest eval):** Agent still stops after the first LC. The `lane_keeping_penalty` may not be strong enough, or p3_v9 has not fully converged yet (training resumed from 670k steps, 330k remaining).

---

## Code Changes Currently in Place

### `lanechange_env.py`

| Change | Location | Effect |
|---|---|---|
| `duration_after_lane_change` default: 40 → 50 | line 493 | 50 post-LC steps before termination |
| `steps_after_success` counter | line 496–497, 204 | Countdown starts when `got_lane_success` fires, not at mid-maneuver `lane_index` snap |
| `successful_lc_count` counter | line 203, 510+ | Tracks number of completed LCs per episode |
| `max_lane_changes` gate in termination | line 511–514 | Terminates after N LCs; 0 = unlimited |

### `train_sim.py`

| Change | Value | Notes |
|---|---|---|
| Phase 2 `speed_limit_weight` | 1.0 (was 0) | Mild speed incentive without overwhelming lane_success |
| Phase 3 `continuous_targets` | True | New target assigned after each LC |
| Phase 3 `max_lane_changes` | 3 | Terminate after 3 LCs |
| Phase 3 `lane_keeping_penalty_when_requested` | 3.0 (was 0) | Penalty for not pursuing next target |

### `eval_sim.py`

Phase 3 block added to match `train_sim.py` exactly (previously eval used `continuous_targets = False`, causing mismatch between training and evaluation environments).

---

## What Did NOT Work (Summary)

| Approach | Why it broke |
|---|---|
| Strong speed penalty in Phase 3 | Contradicted Phase 2's braking behavior → crashes |
| Remove braking reward in Phase 3 | Removed the only safety signal for traffic interaction → crashes |
| Speed reward in Phase 2 at weight=5 | Overwhelmed lane_success → agent never lane-changed |
| Single-LC episodes (original) | Agent stops after 1 LC, no reason to keep driving |
| `continuous_targets` alone (no penalty) | Agent stops after 1 LC, no incentive to pursue next target |

---

## What Is Confirmed Working

- Phase 2 (p2_v7): actively lane-changes, mild ACC preference, safe behavior ✓
- Phase 3 lane change execution: the agent CAN execute a clean LC correctly ✓
- `steps_after_success` timing fix: post-LC buffer now counts from success, not mid-maneuver ✓
- `max_lane_changes` infrastructure: correctly terminates after N LCs ✓

---

## Open Issue

**The agent stops after the first lane change.** After `got_lane_success`, a new target is assigned, but the agent has no strong incentive to pursue it. Current `lane_keeping_penalty = 3.0` (0.3/step effective) may not be sufficient.

---

## Next Steps to Try

1. **Increase `lane_keeping_penalty_when_requested`** — try 8–10 (effective 0.8–1.0/step). At 1.0/step, waiting 50 steps = −50, which equals one lane_success. Strong enough to force action.

2. **Increase Phase 3 `speed_limit_weight` to 3–4** — now that p2_v7 provides a foundation with some speed awareness, a moderate Phase 3 speed penalty is safer than it was with p2_v5. At 3.0 (effective 0.3/step), stopping costs 0.3/step without breaking braking behavior.

3. **Combine both** — `lane_keeping_penalty = 8` + `speed_limit_weight = 3` with `braking_reward = 20` unchanged. The braking reward protects safe behavior; the other two terms attack the stopping incentive from different angles.

4. **Check p3_v9 training logs** (TensorBoard `runs/sim`) — if `ep_len_mean` is trending up and `ent_coef` is dropping, the current config may just need more time. If both are flat/worsening, the penalty is insufficient.
