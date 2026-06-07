"""
train_sim.py — train SAC in pure highway-env simulation (no HIL required).

Usage (from src/sac/):
    python simulations/train_sim.py
    python simulations/train_sim.py --phase 1   # lane-change task only (fast to learn)
    python simulations/train_sim.py --phase 2   # full reward (refine after phase 1)
    python simulations/train_sim.py --resume checkpoints/sim_baseline --phase 2
    python simulations/train_sim.py --timesteps 500000 --checkpoint-dir checkpoints/run2

Phase 1 — learn to change lanes:   collision + lc_progress + lane_success (500)
Phase 2 — add comfort/efficiency:  full reward suite
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch as th
import gymnasium as gym
import lanechange_env  # noqa: F401 — registers lane-changing-v0
from wrappers import ObsWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self._bar = tqdm(total=total_timesteps, unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        self._bar.update(1)
        return True

    def _on_training_end(self) -> None:
        self._bar.close()

LANE_CHOICES = (2, 3, 4)
SPEED_LIMIT  = 30.0  # m/s (~70 mph)
CONSTANT_REW_SCALING = 10.0

# Reward weights per training phase. All divided by CONSTANT_REW_SCALING in make_env().
#
# Phase 0  — braking only, no traffic, lane changes blocked via lc_ttc_gate=inf.
#             Teaches longitudinal safety before any lateral behavior.
# Phase 1  — empty road lane change. No traffic. Teaches clean merge + hold.
# Phase 15 — single slow lead vehicle in ego lane, adjacent lane open. Combines
#             braking and lane changing around one obstacle (replaces old 15).
# Phase 2  — 10 IDM vehicles, phase-1 rewards. Gap selection + wait-for-gap.
# Phase 3  — full reward suite, 10 vehicles. Comfort + efficiency refinement.
PHASE_CONFIGS = {
    0: {
        # Longitudinal safety only. LC blocked via lc_ttc_gate=inf in make_env().
        "collision_penalty":               -500.0,
        "ttc_weight":                        15.0,
        "closing_speed_weight":             5.0,
        "braking_reward_weight":             20.0,
        # gap_weight=0: when stopped, comfortable_gap collapses to 10m (ego_speed=0),
        # and LinearVehicles drive away → front_gap/10 = 1.0 every step for free.
        # Zeroing gap prevents that hack; braking_reward teaches gap management instead.
        "gap_weight":                         0.0,
        "jerk_weight":                        2.0,
        # zero everything lateral
        "lane_success":                       0.0,
        "lane_progress_weight":               0.0,
        "lane_start_bonus":                   0.0,
        "lane_keeping_penalty_when_requested": 0.0,
        "wrong_lane_penalty":                 0.0,
        "blocked_merge_weight":               0.0,
        "lat_accel_weight":                   0.0,
        # speed_limit_weight=6: 0.6/step at full speed — enough to make stopping
        # unprofitable without drowning TTC/closing_speed safety signals.
        "speed_limit_weight":                 6.0,
        # time_penalty capped at ~0.8 raw before crashing becomes preferable to stopping.
        # 0.5 → 0.05/step → -30 total: mild discouragement, safe.
        "time_penalty":                       0.5,
        "oscillation_penalty_weight":         0.0,
    },
    # Phase 1A: single stopped obstacle in ego lane, adjacent lane open.
    # Teaches clean LC mechanics in the simplest possible setting.
    1: {
        "collision_penalty":               -500.0,
        "lane_success":                     200.0,
        "lane_progress_weight":              20.0,
        "ttc_weight":                         40.0,
        "gap_weight":                         0.0,
        "closing_speed_weight":               0.0,
        "braking_reward_weight":              0.0,
        "blocked_merge_weight":               0.0,
        "jerk_weight":                        0.0,
        "lat_accel_weight":                   0.0,
        "speed_limit_weight":                 0.0,
        "time_penalty":                       0.0,
        "lane_start_bonus":                   0.0,
        "lane_keeping_penalty_when_requested": 0.0,
        "wrong_lane_penalty":                 0.0,
        "oscillation_penalty_weight":          5.0,  # small: dampen swerving without blocking exploration
    },
    # Phase 1B: two staggered stopped obstacles, one per lane.
    # Teaches repeated justified lane changes and commitment.
    11: {
        "collision_penalty":               -500.0,
        "lane_success":                     200.0,
        "lane_progress_weight":              20.0,
        "ttc_weight":                         40.0,
        "gap_weight":                         0.0,
        "closing_speed_weight":               0.0,
        "braking_reward_weight":              0.0,
        "blocked_merge_weight":               0.0,
        "jerk_weight":                        0.0,
        "lat_accel_weight":                   0.0,
        "speed_limit_weight":                 0.0,
        "time_penalty":                       0.0,
        "lane_start_bonus":                   0.0,
        "lane_keeping_penalty_when_requested": 0.0,
        "wrong_lane_penalty":                 0.0,
        "oscillation_penalty_weight":         10.0,  # moderate: more opportunity to oscillate here
    },
    # Phase 1.5: 75% follow (both lanes blocked) / 25% escape (ego lane only).
    # Teaches when to lane change vs when to follow. lc_progress=0 intentionally.
    15: {
        "collision_penalty":              -2000.0,
        "lane_success":                     200.0,
        "lane_progress_weight":               0.0,  # LC is not always correct here
        "ttc_weight":                         40.0,
        "closing_speed_weight":              10.0,
        "braking_reward_weight":             15.0,
        "gap_weight":                         5.0,
        "blocked_merge_weight":             -10.0,
        "jerk_weight":                        0.0,
        "lat_accel_weight":                   0.0,
        "speed_limit_weight":                 0.0,
        "time_penalty":                       0.0,
        "lane_start_bonus":                   0.0,
        "lane_keeping_penalty_when_requested": 0.0,
        "wrong_lane_penalty":                 0.0,
        "oscillation_penalty_weight":         20.0,  # strong: unnecessary swerving must be costly
    },
    # Phase 2: 10 IDM vehicles, phase-1 rewards. Gap selection, wait-for-gap.
    2: {
        "collision_penalty":              -2000.0,
        "lane_success":                     200.0,
        "lane_progress_weight":              20.0,
        "ttc_weight":                         40.0,
        "closing_speed_weight":              10.0,
        "braking_reward_weight":             15.0,
        "gap_weight":                         5.0,
        "blocked_merge_weight":             -10.0,
        "jerk_weight":                        0.0,
        "lat_accel_weight":                   0.0,
        "speed_limit_weight":                 1.0,  # mild: doesn't overwhelm lane_success incentive (~0.1/step effective)
        "time_penalty":                       0.0,
        "lane_start_bonus":                   0.0,
        "lane_keeping_penalty_when_requested": 0.0,
        "wrong_lane_penalty":                 0.0,
        "oscillation_penalty_weight":         10.0,
    },
    # Phase 3: full reward suite. Comfort + efficiency on top of safety.
    3: {
        "collision_penalty":              -2000.0,
        "lane_success":                     500.0,
        "lane_progress_weight":              20.0,
        "ttc_weight":                        80.0,
        "gap_weight":                         0.0,
        "closing_speed_weight":              15.0,
        "braking_reward_weight":             20.0,
        "blocked_merge_weight":             -10.0,  # penalize unsafe merge attempts
        "jerk_weight":                        2.0,
        "lat_accel_weight":                   1.0,
        "speed_limit_weight":                 1.0,
        "time_penalty":                       0.05,
        "lane_start_bonus":                   0.0,
        "lane_keeping_penalty_when_requested":  3.0,  # 0.3/step effective: pressure to pursue next target, not park
        "wrong_lane_penalty":                   1.0,
        "oscillation_penalty_weight":          20.0,
    },
}


class RandomLanesWrapper(gym.Wrapper):
    """Randomizes lane count on every reset so the agent trains on 2–4 lane roads."""

    def __init__(self, env, lane_choices=LANE_CHOICES):
        super().__init__(env)
        self.lane_choices = lane_choices

    def reset(self, **kwargs):
        lanes = int(np.random.choice(self.lane_choices))
        self.env.unwrapped.configure({"lanes_count": lanes})
        return self.env.reset(**kwargs)


_PHASE_VEHICLES = {0: 10, 1: 0, 11: 0, 15: 0, 2: 10, 3: 10}

def make_env(phase: int = 1):
    # Phase 0: block LC entirely via an infinite TTC gate so the agent can only learn braking.
    lc_gate = float("inf") if phase == 0 else 4.0

    config = {
        "lane_width":         4.0,
        "road_length":        2000.0,
        "duration":           40,
        "policy_frequency":   15,
        "speed_limit":        SPEED_LIMIT,
        "continuous_targets": False,
        "vehicles_count":     _PHASE_VEHICLES[phase],
        "initial_spacing":    5,
        "lc_ttc_gate":        lc_gate,
        "fwd_ttc_gate":       3.74,
        "rewards":            {k: v / CONSTANT_REW_SCALING for k, v in PHASE_CONFIGS[phase].items()},
    }

    # Phase 0: fix 2 lanes so 4 vehicles create genuine congestion (~2 per lane).
    # Use LinearVehicle (no lane changes, constant speed) so the ego is never side-swiped
    # by an IDM lane change — Phase 0 should be pure longitudinal car-following.
    # All other phases randomize lanes via RandomLanesWrapper and use default IDM.
    if phase == 0:
        config["lanes_count"] = 2
        config["other_vehicles_type"] = "highway_env.vehicle.behavior.LinearVehicle"
        config["initial_spacing"] = 2

    if phase == 1:
        config["lanes_count"]             = 2
        config["p1a_obstacle"]            = True
        config["vehicles_count"]          = 0
        config["continuous_targets"]      = True   # full 600-step episodes; osc penalty prevents ping-pong
        config["require_obstacle_for_lc"] = True   # obstacle at 50-100m always within 80m at LC completion

    if phase == 11:
        config["lanes_count"]             = 2
        config["p1_obstacles"]            = True
        config["vehicles_count"]          = 0
        config["continuous_targets"]      = True
        config["require_obstacle_for_lc"] = True

    if phase == 15:
        config["lanes_count"]             = 2
        config["p15_obstacle"]            = True
        config["vehicles_count"]          = 0
        config["continuous_targets"]      = True
        config["require_obstacle_for_lc"] = True

    if phase == 3:
        config["continuous_targets"] = True
        config["max_lane_changes"]   = 3   # terminate after 3 successful LCs; agent must keep driving between them

    env = gym.make("lane-changing-v0", config=config)
    if phase not in (0, 1, 11, 15):
        env = RandomLanesWrapper(env)
    return ObsWrapper(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",      type=int, default=200_000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path to existing checkpoint zip to continue training")
    parser.add_argument("--phase",          type=int, default=1, choices=[0, 1, 11, 15, 2, 3],
                        help="0=braking only, 1=1A single stopped obstacle, "
                             "11=1B two staggered obstacles, 15=1.5 follow-vs-LC, "
                             "2=10 IDM vehicles, 3=full reward suite")
    parser.add_argument("--n-envs",         type=int, default=1,
                        help="Number of parallel envs (uses SubprocVecEnv). Try 2-4.")
    parser.add_argument("--target-entropy", type=float, default=None,
                        help="Override SAC target entropy (default: keep checkpoint value). "
                             "SAC default is -dim(action_space)=-2.0. More negative = more deterministic.")
    parser.add_argument("--init-ent-coef",  type=float, default=None,
                        help="Reset ent_coef to this value on load (useful when checkpoint has "
                             "an inflated ent_coef from a prior sweep). Typical: 1.0.")
    parser.add_argument("--learning-rate",  type=float, default=None,
                        help="Override SAC learning rate (default: 3e-4). Use 1e-4 for Phase 2 "
                             "to reduce distributional shift when loading a Phase 1 checkpoint.")
    parser.add_argument("--max-grad-norm",  type=float, default=None,
                        help="Clip gradient norm for actor and critic (default: SB3 default=10). "
                             "Use 0.5-1.0 when critic_loss is blowing up (>5000).")
    args = parser.parse_args()

    phase_label = {
        0:  "braking only (LC blocked)",
        1:  "1A single stopped obstacle",
        11: "1B two staggered obstacles",
        15: "1.5 follow vs LC judgment",
        2:  "10 IDM vehicles",
        3:  "full reward suite",
    }
    print(f"[train_sim] Phase {args.phase}: {phase_label[args.phase]}  n_envs={args.n_envs}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.n_envs > 1:
        env = make_vec_env(
            make_env,
            n_envs=args.n_envs,
            env_kwargs={"phase": args.phase},
            vec_env_cls=SubprocVecEnv,
        )
    else:
        env = make_env(phase=args.phase)

    if args.resume:
        print(f"[train_sim] Resuming from {args.resume}")
        custom_objects = {}
        if args.learning_rate is not None:
            custom_objects["learning_rate"] = args.learning_rate
            custom_objects["lr_schedule"] = lambda _: args.learning_rate
            print(f"[train_sim] learning_rate overridden to {args.learning_rate}")
        model = SAC.load(args.resume, env=env, custom_objects=custom_objects or None)
        if args.target_entropy is not None:
            model.target_entropy = args.target_entropy
            print(f"[train_sim] target_entropy overridden to {args.target_entropy}")
        if args.init_ent_coef is not None:
            model.log_ent_coef = th.log(
                th.ones(1, device=model.device) * args.init_ent_coef
            ).requires_grad_(True)
            model.ent_coef_optimizer = th.optim.Adam(
                [model.log_ent_coef], lr=model.lr_schedule(1)
            )
            print(f"[train_sim] ent_coef reset to {args.init_ent_coef}")
        if args.max_grad_norm is not None:
            model.max_grad_norm = args.max_grad_norm
            print(f"[train_sim] max_grad_norm overridden to {args.max_grad_norm}")
    else:
        te = args.target_entropy if args.target_entropy is not None else "auto"
        lr = args.learning_rate if args.learning_rate is not None else 3e-4
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="runs/sim",
            target_entropy=te,
            learning_rate=lr,
        )
        if args.max_grad_norm is not None:
            model.max_grad_norm = args.max_grad_norm
            print(f"[train_sim] max_grad_norm set to {args.max_grad_norm}")

    callbacks = [
        TqdmCallback(args.timesteps),
        CheckpointCallback(
            save_freq=10_000,
            save_path=args.checkpoint_dir,
            name_prefix="sim_sac",
        ),
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
        )
    except KeyboardInterrupt:
        print("\n[train_sim] Interrupted — saving current weights ...")
    finally:
        final_path = os.path.join(args.checkpoint_dir, "sim_baseline")
        model.save(final_path)
        print(f"[train_sim] Saved to {final_path}.zip")
        env.close()


if __name__ == "__main__":
    main()
