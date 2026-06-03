# Phase 1 Lane-Change Curriculum Notes

## Purpose

The goal of Phase 1 and Phase 1.5 is to build a stable foundation for safe lane-change behavior before adding dense traffic.

The agent should eventually learn:

- when to follow the vehicle ahead
- when to brake
- when a lane change is safer or more useful
- how to complete a lane change without oscillating
- how to avoid unnecessary lane changes

The main design principle is:

```text
Phase 1 teaches how to lane change.
Phase 1.5 teaches when to lane change.
```

Do not make the earliest lane-change phases too realistic. Their job is to isolate the behavior we want the agent to learn.

---

## Phase 1A: Single Justified Lane Change

### Goal

Teach the basic mechanics of a lane change in the cleanest possible setting.

The agent should learn:

- detect that the current lane has a real obstacle
- choose the adjacent open lane
- initiate a lane change
- complete the lane change
- stabilize in the new lane
- avoid drifting or oscillating after completion

### Scenario Layout

```text
lane 0: ego ---------------- slow/stopped obstacle

lane 1: empty
```

The correct behavior is intentionally obvious:

```text
obstacle ahead -> adjacent lane safe -> lane change -> hold lane
```

### Episode Structure

Phase 1A should not terminate immediately after the lane change succeeds.

The episode should continue for the full rollout horizon, for example 600 timesteps, so the agent must prove it can hold the new lane cleanly after completing the maneuver.

The desired behavior is:

```text
one justified lane change -> complete merge -> drive straight for the rest of the episode
```

This matters because a policy can look good if the episode ends at the instant of lane-change success, while still having unstable behavior afterward. The hold period exposes:

- drifting after success
- oscillating intent after success
- unnecessary second lane changes
- poor lateral stabilization
- reward hacking through repeated lane-change attempts

Lane-change success should be paid once. After that, the reward should mostly encourage staying safe, staying centered, and not making another unnecessary lane change.

### Reward Shape

Phase 1A can safely use lane-change progress because lane changing is always the desired behavior in this scenario.

Recommended reward priorities:

- high `lane_success`
- nonzero `lc_progress`
- small or moderate `ttc`
- small oscillation penalty
- penalty for unnecessary second lane changes after success
- lane-centering or low lateral-motion incentive after success
- low or zero braking reward
- low or zero gap reward

The phase should not strongly reward car-following yet. That belongs in Phase 1.5.

### Why This Phase Exists

This phase prevents the agent from having to learn lane-change judgment and lane-change execution at the same time.

It is a controlled classroom for the physical skill:

```text
Can the policy actually move into another lane cleanly?
```

---

## Phase 1B: Sequential Justified Lane Changes

### Goal

Extend Phase 1A by teaching repeated lane-change execution.

The agent should learn:

- complete one lane change
- encounter a second obstacle later
- decide to lane change again
- avoid ping-pong behavior
- commit to a lane change once started

### Scenario Layout

```text
lane 0: ego ---------------- obstacle 1

lane 1: ----------------------------- obstacle 2
```

The expected behavior is:

```text
start in lane 0
change to lane 1 to avoid obstacle 1
later change back to lane 0 to avoid obstacle 2
```

### Reward Shape

Phase 1B can use a similar reward shape to Phase 1A:

- high `lane_success`
- nonzero `lc_progress`
- small or moderate `ttc`
- small or moderate oscillation penalty
- low or zero braking reward
- low or zero gap reward

The oscillation penalty can be slightly stronger than Phase 1A because this scenario creates more opportunity for unnecessary direction reversals.

### Why This Phase Exists

Phase 1A teaches one clean lane change.

Phase 1B teaches that lane changes are reusable maneuvers, but they should still be tied to an obstacle or safety reason.

This phase is richer than Phase 1A, but it still avoids the harder question:

```text
Should I lane change, or should I follow?
```

That question belongs in Phase 1.5.

---

## Phase 1.5: Follow vs Lane Change Judgment

### Goal

Teach the agent to choose between safe following and safe lane changing.

This is the first phase where lane changing should not always be rewarded.

The agent should learn:

- follow when both lanes are effectively blocked or equivalent
- brake when the front vehicle is too close and the ego is gaining
- lane change only when the adjacent lane gives a safer option
- avoid pointless lane changes
- avoid oscillating between lanes

### Scenario Mix

Use two scenario types.

#### 75% Follow Scenario

Two lanes have lead vehicles at similar speed and similar distance.

```text
lane 0: ego ---------------- lead vehicle

lane 1: -------------------- lead vehicle
```

The adjacent lane does not offer a meaningful advantage.

The desired behavior is:

```text
slow down -> maintain safe following distance -> do not lane change unnecessarily
```

#### 25% Pass/Escape Scenario

This is closer to Phase 1.

```text
lane 0: ego ---------------- slower obstacle

lane 1: open or safer gap
```

The desired behavior is:

```text
recognize unsafe/inefficient following -> lane change safely -> stabilize
```

### Reward Shape

Phase 1.5 should use a shared safety objective across both scenario types.

Recommended reward priorities:

- strong collision penalty
- TTC penalty
- close-and-gaining penalty
- braking reward when front gap is tight
- modest gap/following reward
- blocked merge penalty
- oscillation penalty

For the first implementation, set `lc_progress` to `0`.

This is intentional. In Phase 1.5, lane changing is not always correct, so generic lane-progress reward can teach the wrong behavior.

Lane-change success should only be rewarded when the lane change is justified by the scene.

Good justification signals:

- front vehicle exists
- front gap is below desired headway
- ego is faster than the front vehicle
- TTC is low enough to indicate growing risk
- target lane has safe lead and lag TTC

This avoids hard-coding the action while still preventing reward leakage.

The reward should say:

```text
This state is risky. Resolve it safely.
```

It should not say:

```text
Always lane change.
```

### Close-And-Gaining Penalty

Add a penalty when the ego is too close to the leading car and still gaining on it.

Conceptually:

```text
if front_vehicle_exists:
    closing = ego_speed - front_speed
    desired_gap = max(ego_speed * headway_time, minimum_gap)

    if front_gap < desired_gap and closing > 0:
        penalty = gap_risk * closing_risk
```

This teaches the agent that staying fast behind a slower lead vehicle is bad.

It does not force the agent to lane change. The policy can resolve the penalty by braking, following, or changing lanes safely when the adjacent lane is useful.

### Main Risk

The biggest risk in Phase 1.5 is reward mismatch between the two scenario types.

In the 75% follow scenario, lane changing should usually not help.

In the 25% pass scenario, lane changing may be the best answer.

To reduce mismatch, make both scenarios share the same deeper objective:

```text
Maintain safe headway and TTC.
Resolve unsafe closing behavior.
Only lane change when it improves safety or progress.
```

---

## Recommended Curriculum

```text
Phase 1A:
    one obstacle
    adjacent lane open
    teach one clean justified lane change
    continue to full horizon after success
    verify the agent holds the new lane without drifting or changing back

Phase 1B:
    staggered obstacles
    teach repeated justified lane changes

Phase 1.5:
    75% follow scenario
    25% pass/escape scenario
    teach when lane changing is actually useful
```

---

## Current Opinion

Set `lc_progress` to `0` in Phase 1.5 for the first pass.

This is safer because Phase 1.5 contains scenarios where lane changing is not the correct behavior. Generic lateral-progress reward could accidentally teach the policy to drift or lane-change even when following is safer.

Bring `lc_progress` back later only after it is gated by a justified-lane-change condition.

Phase 1 can keep `lc_progress` because every Phase 1 episode is intentionally designed to require a lane change.
