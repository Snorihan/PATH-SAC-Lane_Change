"""
test_rewards.py

Usage:
  python test_rewards.py

What it does:
  - Instantiates your custom env (lane-changing-v0)
  - Runs a short rollout with simple actions
  - Evaluates your reward component functions each step
  - Prints + aggregates results for sanity checking

Assumptions:
  - You registered env id 'lane-changing-v0' pointing to your LaneChangingEnv
  - Your env exposes the methods exactly as pasted (names match)
  - Env uses ContinuousAction (or you can change action generation below)
"""

import time
import numpy as np
import gymnasium as gym

from highway_env.vehicle.controller import ControlledVehicle
# Ensure your env module is imported so the register() call runs.
# If you register in lanechange_env.py, import it here.
import lanechange_env  # noqa: F401


def finite(x) -> bool:
    return x is not None and np.isfinite(x)


def pretty(x):
    if x is None:
        return "None"
    if isinstance(x, float):
        if not np.isfinite(x):
            return "inf" if x > 0 else "-inf"
        return f"{x: .3f}"
    return str(x)


def main():
    # --- Create env with configs that increase chance of neighbors existing ---
    # You may need to tweak these depending on your highway_env version.
    config = {
        # "action": {"type": "ContinuousAction"},
        # Try to keep traffic present so front/rear vehicles exist
        "vehicles_count": 25,
        "controlled_vehicles": 1,
        "lanes_count": 4,
        # For speed limit component
        "speed_limit": 30.0,  # whatever unit your env uses for speed; just be consistent
        # Keep sim stable
        "policy_frequency": 15,
        "duration": 300,
    }

    # --- Monkey-patch the controller gains to be less aggressive ---
    # The default PD controller is too aggressive and causes the car to oscillate and spin.
    # We reduce the proportional gains for lateral and heading error.
    ControlledVehicle.K_LAT = 0.2
    ControlledVehicle.K_HEADING = 0.2

    env = gym.make("lane-changing-v0", config=config, render_mode="human")
    base = env.unwrapped

    # --- Reset once ---
    obs, info = env.reset(seed=0)
    base = env.unwrapped
    from_n, to_n, lane_id = base.vehicle.lane_index
    lanes = int(base.config.get("lanes_count", 4))
    target_id = lane_id + 1 if lane_id + 1 < lanes else lane_id - 1
    base.target_lane_index = (from_n, to_n, int(target_id))

    print("START:", base.vehicle.lane_index, "TARGET:", base.target_lane_index)


    # Some of your reward functions rely on state variables existing:
    # start_time, prev_acceleration, prev_lat, lane_changing, target_lane_index
    # Make sure they're initialized (do this once here for the test harness).
    base.start_time = 0.0
    base.prev_acceleration = 0.0
    # prev_lat should be initialized to current lat so v_lat doesn't spike at t=0
    lane = base.road.network.get_lane(base.vehicle.lane_index)
    _, lat0 = lane.local_coordinates(base.vehicle.position)
    base.prev_lat = float(lat0)

    # Ensure you have a target lane index for the success term
    if not hasattr(base, "target_lane_index"):
        base.target_lane_index = base.find_target_lane(base.vehicle.lane_index)

    # Ensure the lane_changing flag exists
    if not hasattr(base, "lane_changing"):
        base.lane_changing = False

    # --- Rollout parameters ---
    n_steps = 300
    dt = 1.0 / float(base.config.get("policy_frequency", 15))

    # --- Aggregation ---
    hist = {
        "jerk": [],
        "rel_dist": [],
        "rel_speed": [],
        "speed_limit": [],
        "lane_success": [],
        "lane_changing": [],
        # context:
        "gap_front": [],
        "gap_rear": [],
        "ego_speed": [],
        "front_speed": [],
        "rear_speed": [],
    }

    # --- Simple action generator ---
    # accel in [-1, 1], intent in [-1, 1]
    # We'll do a mild oscillation to create varied behavior.
    def sample_action(k):
        front_v, _, gapF, _ = base._vehicle_in_front_rear(base.vehicle.lane_index, max_range=200.0)

        accel = 0.05
        intent = 0.0

        # If there is a lead car close and slower, request lane change
        if front_v is not None:
            closing = float(base.vehicle.speed - front_v.speed)
            if gapF < 25.0 and closing > 1.0:
                intent = 1.0
        if k % 10 == 0:
            x, y = base.vehicle.position
            print("pos", (x, y), "heading", base.vehicle.heading)

        return np.array([accel, intent], dtype=np.float32)

    # --- Run rollout ---
    sim_time = 0.0
    for k in range(n_steps):
        action = sample_action(k)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render() # Ensure PyGame updates

        # --- Context: neighbors + speeds ---
        curr_lane_index = base.vehicle.lane_index
        front_v, rear_v, gap_front, gap_rear = base._vehicle_in_front_rear(
            curr_lane_index, max_range=200.0
        )

        ego_speed = float(base.vehicle.speed)
        front_speed = float(front_v.speed) if front_v is not None else None
        rear_speed = float(rear_v.speed) if rear_v is not None else None

        # --- Retrieve reward components from info ---
        # Note: These are the weighted reward terms, except raw_jerk
        jerk = info.get("raw_jerk", 0.0)
        
        # Log the weighted terms directly from the env
        rel_dist = info.get("r_dist", 0.0)
        rel_speed = info.get("r_speed", 0.0)
        speed_limit_term = info.get("r_limit", 0.0)
        lane_success = info.get("r_success", 0.0)
        lane_changing_term = info.get("r_changing", 0.0)

        # --- Basic sanity checks (should not crash and should be finite) ---
        # jerk can be finite; if dt ever goes 0 your jerk will blow up.
        if not np.isfinite(jerk):
            print(f"[WARN] jerk is non-finite at step {k}: {jerk}")

        for name, val in [
            ("rel_dist", rel_dist),
            ("rel_speed", rel_speed),
            ("speed_limit", speed_limit_term),
            ("lane_success", lane_success),
            ("lane_changing", lane_changing_term),
        ]:
            if not np.isfinite(float(val)):
                print(f"[WARN] {name} is non-finite at step {k}: {val}")

        # --- Store ---
        hist["jerk"].append(float(jerk))
        hist["rel_dist"].append(float(rel_dist))
        hist["rel_speed"].append(float(rel_speed))
        hist["speed_limit"].append(float(speed_limit_term))
        hist["lane_success"].append(float(lane_success))
        hist["lane_changing"].append(float(lane_changing_term))

        hist["gap_front"].append(float(gap_front) if np.isfinite(gap_front) else np.inf)
        hist["gap_rear"].append(float(gap_rear) if np.isfinite(gap_rear) else -np.inf)
        hist["ego_speed"].append(ego_speed)
        hist["front_speed"].append(front_speed if front_speed is not None else np.nan)
        hist["rear_speed"].append(rear_speed if rear_speed is not None else np.nan)

        # --- Print a compact line every so often ---
        if k % 20 == 0:
            print(
                f"step {k:03d} | lane={curr_lane_index} "
                f"| lane_changing={base.lane_changing} "
                f"| gapF={pretty(float(gap_front))} gapR={pretty(float(gap_rear))} "
                f"| vE={pretty(ego_speed)} vF={pretty(front_speed)} vR={pretty(rear_speed)} "
                f"| jerk={pretty(float(jerk))} "
                f"| r_dist={pretty(float(rel_dist))} r_spd={pretty(float(rel_speed))} "
                f"| r_lim={pretty(float(speed_limit_term))} "
                f"| r_succ={pretty(float(lane_success))} r_lc={pretty(float(lane_changing_term))}"
            )

        if k % 5 == 0:
            print("action_in_vehicle:", base.vehicle.action, "heading:", base.vehicle.heading)

        if terminated or truncated:
            print(f"[END] terminated={terminated} truncated={truncated} at step {k}")
            break

    # --- Summary stats ---
    def summarize(name):
        arr = np.array(hist[name], dtype=np.float64)
        return {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
        }

    print("\n=== Summary ===")
    for key in ["jerk", "rel_dist", "rel_speed", "speed_limit", "lane_success", "lane_changing"]:
        s = summarize(key)
        print(f"{key:>12}: min={s['min']:+.3f} max={s['max']:+.3f} mean={s['mean']:+.3f}")

    env.close()


if __name__ == "__main__":
    main()
