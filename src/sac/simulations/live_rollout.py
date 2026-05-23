"""
live_rollout.py — smoke-test the full live pipeline against a running Aimsun instance.

What it does:
  1. Connects to the HIL Tool via HilConnector (START2/MSGEND framed UDP).
  2. Resets the episode and runs N steps with a trivial scripted policy.
  3. Prints reward terms every step so you can eyeball them.

Usage (from src/sac/):
  python simulations/live_rollout.py

To test without a real Aimsun, use simulated_test_rollout.py instead.

Edit the CONFIG block below before running.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # src/sac on path

import lanechange_env  # noqa: F401 — registers lane-changing-v0
from cda_live.cda_hil_connector import HilConnector
from lanechange_env import LaneChangingEnv

# ── CONFIG — edit these before running ───────────────────────────────────────

HIL_IP     = "127.0.0.1"
HIL_PORT   = 7999   # HIL Tool listens here (same port Aimsun used to use)
LOCAL_PORT = 7998   # Python listens here; HIL Tool sends back to this port
EGO_ID     = 10     # Test vehicle ID — must match HIL Tool vehicle config (vehicle_id[0] = 10)
TIMEOUT_S   = 10.0

INITIAL_POS_M     = 100.0   # starting position on section (metres)
INITIAL_SPEED_MPS = 15.0    # starting speed (m/s)

N_STEPS     = 400
POLICY_HZ   = 15
DURATION_S  = 40
LANES       = 1
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
    parser = argparse.ArgumentParser(description="Live HIL rollout")
    parser.add_argument(
        "--no-teleport", action="store_true",
        help="On reset, keep the car's current position/speed instead of teleporting back to INITIAL_POS_M.",
    )
    parser.add_argument(
        "--keep-dashboard-alive", action="store_true",
        help="After the run ends, keep sending keepalive packets so the HIL dashboard stays up. Press Ctrl+C to exit.",
    )
    args = parser.parse_args()

    print(f"[live_rollout] Connecting to HIL Tool at {HIL_IP}:{HIL_PORT} ...")
    print(f"[live_rollout] teleport_on_reset={'False' if args.no_teleport else 'True'}")
    connector = HilConnector({
        "hil_ip":              HIL_IP,
        "hil_port":            HIL_PORT,
        "local_port":          LOCAL_PORT,
        "ego_id":              EGO_ID,
        "initial_pos_m":       INITIAL_POS_M,
        "initial_speed_mps":   INITIAL_SPEED_MPS,
        "timeout_s":           TIMEOUT_S,
        "poll_interval_s":     0.001,
        "teleport_on_reset":   not args.no_teleport,
    })

    lanes = connector.get_total_lanes()
    print(f"[live_rollout] lanes={lanes} (from HIL Tool)")

    env = LaneChangingEnv(config={
        "backend":          "aimsun_live",
        "live_connector":   connector,
        "lanes_count":      lanes,
        "lane_width":       4.0,
        "road_length":      1000.0,
        "policy_frequency": POLICY_HZ,
        "duration":         DURATION_S,
        "speed_limit":      SPEED_LIMIT,
    })

    try:
        print("[live_rollout] Waiting for first HIL Tool frame (reset) ...")
        obs, info = env.reset()
        print(f"[live_rollout] Reset OK  lane={env.vehicle.lane_index}  "
              f"speed={env.vehicle.speed:.1f} m/s  "
              f"target={env.target_lane_index}")
        print(f"              cda keys: {list(info.get('cda', {}).keys())}")

        total_reward = 0.0

        # Sanity-check: warn when the HIL Tool stops sending new frames.
        # The HIL Tool timestamp freezing means Aimsun paused or the test
        # vehicle was de-registered.  pos_m / speed_mps are locally computed
        # by the connector so they always advance; checking them here would
        # miss the real problem.
        # At 15 Hz policy vs 10 Hz HIL Tool, 1-2 stale frames per step are normal.
        FREEZE_WARN_STEPS = 5

        for step in range(N_STEPS):
            action = sample_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            rt   = info.get("reward_terms", {})
            lcs  = info.get("lane_change_state", {})
            snap = info.get("snapshot", {})
            snap_ego = snap.get("ego", {})

            stale = snap.get("aimsun_stale_steps", 0)
            if stale == FREEZE_WARN_STEPS:
                print(
                    f"[WARN] HIL Tool timestamp frozen for {stale} consecutive steps "
                    f"(time_s={snap.get('time_s')}).  "
                    f"Ego will not receive updated surrounding state.  "
                    f"Check: (1) Aimsun simulation is running and not paused, "
                    f"(2) EGO_ID={EGO_ID} is registered in HIL Tool config, "
                    f"(3) HIL Tool is sending to local_port={LOCAL_PORT}."
                )

            print(
                f"step {step:03d} | "
                f"lane={env.vehicle.lane_index[2]}  "
                f"spd={pretty(env.vehicle.speed)}  "
                f"pos={pretty(snap_ego.get('pos_m'))}  "
                f"lc={lcs.get('lane_changing', '?')}  "
                f"| r={pretty(reward)}  "
                f"(col={pretty(rt.get('collision'))}  "
                f"ttc={pretty(rt.get('ttc'))}  "
                f"jrk={pretty(rt.get('jerk'))}  "
                f"lim={pretty(rt.get('speed_limit'))}  "
                f"lst={pretty(rt.get('lane_start'))}  "
                f"prg={pretty(rt.get('lc_progress'))}  "
                f"suc={pretty(rt.get('lane_success'))}  "
                f"tpen={pretty(rt.get('time_penalty'))})"
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
        if args.keep_dashboard_alive:
            connector.idle()   # blocks until Ctrl+C, then closes
        else:
            env.close()
        print("[live_rollout] Connector closed.")


if __name__ == "__main__":
    main()
