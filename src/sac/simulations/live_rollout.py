"""
live_rollout.py — smoke-test the full live pipeline against a running Aimsun instance.

What it does:
  1. Connects to Aimsun via CdaLiveConnector (real DLL, real UDP).
  2. Resets the episode and runs N steps with a trivial scripted policy.
  3. Prints reward terms every step so you can eyeball them.

Usage (from src/sac/):
  python simulations/live_rollout.py

Edit the CONFIG block below before running.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # src/sac on path

import lanechange_env  # noqa: F401 — registers lane-changing-v0
from cda_live.cda_live_connector import CdaLiveConnector
from lanechange_env import LaneChangingEnv

# ── CONFIG — edit these before running ───────────────────────────────────────

DLL_PATH    = r"C:\Users\janus\Desktop\BerkeleyFileMasterDirectory\PATH\Vehicle_Test_Interface\out\build\Release\Debug\CDA_Interface_Wrapper.dll"
REMOTE_IP   = "127.0.0.1"
REMOTE_PORT = 5555
LOCAL_PORT  = 5556
EGO_ID      = 1
TIMEOUT_S   = 10.0

N_STEPS     = 200
POLICY_HZ   = 15
DURATION_S  = 40
LANES       = 4
SPEED_LIMIT = 30.0   # m/s — set to None to disable speed-limit reward term

# ─────────────────────────────────────────────────────────────────────────────


def sample_action(env: LaneChangingEnv) -> np.ndarray:
    """
    Trivial scripted policy: gentle constant throttle, no lane change.
    Swap this out with a trained SAC model when ready.
    """
    front_v, _, gap_front, _ = env._vehicle_in_front_rear(
        env.vehicle.lane_index, max_range=200.0
    )
    accel  = 0.05
    intent = 0.0
    if front_v is not None and gap_front < 25.0:
        closing = float(env.vehicle.speed - front_v.speed)
        if closing > 1.0:
            intent = 1.0   # request lane change
    return np.array([accel, intent], dtype=np.float32)


def pretty(x) -> str:
    if x is None:
        return "None"
    try:
        f = float(x)
        return "inf" if f == float("inf") else "-inf" if f == float("-inf") else f"{f:+.3f}"
    except Exception:
        return str(x)


def main():
    print(f"[live_rollout] Connecting to Aimsun at {REMOTE_IP}:{REMOTE_PORT} ...")
    connector = CdaLiveConnector({
        "dll_path":        DLL_PATH,
        "remote_ip":       REMOTE_IP,
        "remote_port":     REMOTE_PORT,
        "local_port":      LOCAL_PORT,
        "ego_id":          EGO_ID,
        "timeout_s":       TIMEOUT_S,
        "poll_interval_s": 0.001,
    })

    env = LaneChangingEnv(config={
        "backend":          "aimsun_live",
        "live_connector":   connector,
        "lanes_count":      LANES,
        "lane_width":       4.0,
        "road_length":      1000.0,
        "policy_frequency": POLICY_HZ,
        "duration":         DURATION_S,
        "speed_limit":      SPEED_LIMIT,
    })

    try:
        print("[live_rollout] Waiting for first Aimsun frame (reset) ...")
        obs, info = env.reset()
        print(f"[live_rollout] Reset OK  lane={env.vehicle.lane_index}  "
              f"speed={env.vehicle.speed:.1f} m/s  "
              f"target={env.target_lane_index}")
        print(f"              cda keys: {list(info.get('cda', {}).keys())}")

        total_reward = 0.0

        for step in range(N_STEPS):
            action = sample_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            rt = info.get("reward_terms", {})
            lcs = info.get("lane_change_state", {})
            snap_ego = info.get("snapshot", {}).get("ego", {})

            print(
                f"step {step:03d} | "
                f"lane={env.vehicle.lane_index[2]}  "
                f"spd={pretty(env.vehicle.speed)}  "
                f"pos={pretty(snap_ego.get('pos_m'))}  "
                f"lc={lcs.get('lane_changing', '?')}  "
                f"| r={pretty(reward)}  "
                f"(col={pretty(rt.get('collision'))}  "
                f"jrk={pretty(rt.get('jerk'))}  "
                f"spd={pretty(rt.get('speed'))}  "
                f"dst={pretty(rt.get('dist'))}  "
                f"lim={pretty(rt.get('speed_limit'))}  "
                f"suc={pretty(rt.get('lane_success'))}  "
                f"chg={pretty(rt.get('lane_changing'))})"
            )

            if terminated:
                print(f"[live_rollout] TERMINATED at step {step}")
                break
            if truncated:
                print(f"[live_rollout] TRUNCATED at step {step} "
                      f"(elapsed_steps={env.elapsed_steps})")
                break

        print(f"\n[live_rollout] Done.  total_reward={total_reward:.2f}  "
              f"steps={env.elapsed_steps}")

    finally:
        env.close()
        print("[live_rollout] Connector closed.")


if __name__ == "__main__":
    main()
