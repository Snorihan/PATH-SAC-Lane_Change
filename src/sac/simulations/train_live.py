"""
train_live.py — train SAC against a live Aimsun instance via HIL Tool.

Rewards and observations come from real Aimsun traffic — not highway-env physics.
Optionally warm-starts from a sim-pretrained checkpoint (train_sim.py output).

Run order: Aimsun → HIL Tool → this script.

Usage (from src/sac/):
    python simulations/train_live.py
    python simulations/train_live.py --resume checkpoints/sim_baseline
    python simulations/train_live.py --timesteps 100000 --checkpoint-dir checkpoints/live_run1
    python simulations/train_live.py --keep-dashboard-alive
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import lanechange_env  # noqa: F401 — registers lane-changing-v0
from lanechange_env import LaneChangingEnv
from cda_live.cda_hil_connector import HilConnector
from wrappers import ObsWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# ── CONFIG — edit these before running ───────────────────────────────────────

HIL_IP     = "127.0.0.1"
HIL_PORT   = 7999
LOCAL_PORT = 7998
EGO_ID     = 10
TIMEOUT_S  = 10.0

INITIAL_POS_M     = 100.0
INITIAL_SPEED_MPS = 15.0

POLICY_HZ   = 15
DURATION_S  = 40
SPEED_LIMIT = 30.0  # m/s

# ─────────────────────────────────────────────────────────────────────────────


class EpisodeLogger(BaseCallback):
    """Logs per-episode reward breakdown from info['reward_terms']."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_steps  = 0
        self._ep_terms  = {}
        self._episode   = 0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        info   = self.locals["infos"][0]
        done   = self.locals["dones"][0]

        self._ep_reward += float(reward)
        self._ep_steps  += 1

        for k, v in (info.get("reward_terms") or {}).items():
            self._ep_terms[k] = self._ep_terms.get(k, 0.0) + float(v)

        if done:
            self._episode += 1
            terms_str = "  ".join(
                f"{k}={v:+.2f}" for k, v in sorted(self._ep_terms.items())
            )
            print(
                f"[ep {self._episode:04d}] "
                f"steps={self._ep_steps}  "
                f"reward={self._ep_reward:+.2f}  |  {terms_str}"
            )
            self._ep_reward = 0.0
            self._ep_steps  = 0
            self._ep_terms  = {}
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",           type=int, default=100_000)
    parser.add_argument("--checkpoint-dir",       type=str, default="checkpoints")
    parser.add_argument("--resume",               type=str, default=None,
                        help="Checkpoint zip to load (sim_baseline or prior live run)")
    parser.add_argument("--warmup-steps",         type=int, default=1_000,
                        help="Random exploration steps before first SAC update")
    parser.add_argument("--no-teleport",          action="store_true")
    parser.add_argument("--keep-dashboard-alive", action="store_true",
                        help="After training ends, keep sending keepalive packets "
                             "so the HIL dashboard stays up. Press Ctrl+C to exit.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"[train_live] Connecting to HIL Tool at {HIL_IP}:{HIL_PORT} ...")
    connector = HilConnector({
        "hil_ip":            HIL_IP,
        "hil_port":          HIL_PORT,
        "local_port":        LOCAL_PORT,
        "ego_id":            EGO_ID,
        "initial_pos_m":     INITIAL_POS_M,
        "initial_speed_mps": INITIAL_SPEED_MPS,
        "timeout_s":         TIMEOUT_S,
        "poll_interval_s":   0.001,
        "teleport_on_reset": not args.no_teleport,
    })

    lanes = connector.get_total_lanes()
    print(f"[train_live] lanes={lanes} (from HIL Tool)")

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
    env = ObsWrapper(env)

    if args.resume:
        print(f"[train_live] Loading weights from {args.resume}")
        model = SAC.load(args.resume, env=env)
    else:
        print("[train_live] Training from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            learning_starts=args.warmup_steps,
            tensorboard_log="runs/live",
        )

    callbacks = [
        EpisodeLogger(),
        CheckpointCallback(
            save_freq=5_000,
            save_path=args.checkpoint_dir,
            name_prefix="live_sac",
        ),
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
        )
    finally:
        final_path = os.path.join(args.checkpoint_dir, "live_final")
        model.save(final_path)
        print(f"[train_live] Saved to {final_path}.zip")

        if args.keep_dashboard_alive:
            connector.idle()  # blocks until Ctrl+C, then closes
        else:
            env.close()
        print("[train_live] Connector closed.")


if __name__ == "__main__":
    main()
