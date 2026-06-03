"""
brake_viz.py — render the Phase 0 environment with a scripted "brake when close" policy.

Use this to visually confirm the action scaling fix works before training.
Watch the ego (green) slow down and follow the lead vehicles instead of crashing.

Run from src/sac/:
    python simulations/brake_viz.py
    python simulations/brake_viz.py --accel 0.3   # cruising action instead of braking
    python simulations/brake_viz.py --random       # random actions (should crash quickly)
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
import lanechange_env  # noqa: F401
from wrappers import ObsWrapper
from train_sim import PHASE_CONFIGS, CONSTANT_REW_SCALING, _PHASE_VEHICLES


def make_render_env():
    env = gym.make("lane-changing-v0", render_mode="human", config={
        "lanes_count":        2,
        "lane_width":         4.0,
        "road_length":        1000.0,
        "duration":           40,
        "policy_frequency":   15,
        "speed_limit":        30.0,
        "continuous_targets": False,
        "vehicles_count":     10,
        "lc_ttc_gate":        float("inf"),
        "fwd_ttc_gate":       3.74,
        "other_vehicles_type": "highway_env.vehicle.behavior.LinearVehicle",
        "initial_spacing":     2,
        "rewards": {k: v / CONSTANT_REW_SCALING for k, v in PHASE_CONFIGS[0].items()},
    })
    return ObsWrapper(env)


def scripted_action(env_unwrapped, fixed_accel=None, random=False):
    """Simple reactive policy: brake proportionally to closing speed."""
    if random:
        return np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

    if fixed_accel is not None:
        return np.array([fixed_accel, 0.0], dtype=np.float32)

    sv = env_unwrapped._get_surrounding_vehicles()
    ego_speed = float(env_unwrapped.vehicle.speed)

    if sv.front_v is not None:
        closing = ego_speed - float(sv.front_v.speed)
        gap = sv.front_gap
        # Proportional brake: stronger as gap shrinks or closing speed grows
        if closing > 0 and gap < 60.0:
            urgency = np.clip(closing / max(gap, 1.0), 0.0, 1.0)
            accel_norm = -urgency
        else:
            accel_norm = 0.2  # gentle cruise
    else:
        accel_norm = 0.5  # open road: accelerate

    return np.array([accel_norm, 0.0], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--accel",    type=float, default=None,
                        help="Fixed accel_norm in [-1,1]. Omit for scripted brake policy.")
    parser.add_argument("--random",   action="store_true",
                        help="Random actions (should crash to confirm shield baseline)")
    args = parser.parse_args()

    env = make_render_env()
    base = env.unwrapped

    for ep in range(args.episodes):
        obs, _ = env.reset()
        print(f"\n── Episode {ep + 1} ──")
        print(f"{'step':>4}  {'speed':>6}  {'front_gap':>9}  {'accel_cmd':>9}  "
              f"{'fwd_int':>7}  {'lc_int':>6}  {'reward':>7}  {'crash':>5}")

        total_r = 0.0
        for step in range(600):
            action = scripted_action(base, fixed_accel=args.accel, random=args.random)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward

            sv      = base._get_surrounding_vehicles()
            shield  = info.get("shield", {})
            crashed = base.vehicle.crashed

            print(
                f"{step:4d}  "
                f"{base.vehicle.speed:6.2f}  "
                f"{sv.front_gap if sv.front_v else float('inf'):9.1f}  "
                f"{float(action[0]):+9.3f}  "
                f"{shield.get('fwd_interventions', 0):7d}  "
                f"{shield.get('lc_interventions', 0):6d}  "
                f"{reward:+7.3f}  "
                f"{'YES' if crashed else '':>5}"
            )

            if terminated or truncated:
                reason = "CRASH" if crashed else "TRUNCATED"
                print(f"→ {reason} at step {step}  total_reward={total_r:+.2f}")
                break

    env.close()


if __name__ == "__main__":
    main()
