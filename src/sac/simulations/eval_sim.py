"""
eval_sim.py — evaluate a trained SAC checkpoint in pure highway-env simulation.

No Aimsun or HIL Tool required.  Loads a checkpoint zip and runs N episodes,
printing per-episode reward breakdown and a summary at the end.

Usage (from src/sac/):
    python simulations/eval_sim.py --model checkpoints/p2_clean/sim_baseline --render
    python simulations/eval_sim.py --model checkpoints/p2_clean/sim_baseline --render --phase 2 --episodes 10
    python simulations/eval_sim.py --model checkpoints/p2_clean/sim_baseline --render --stochastic
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
import lanechange_env  # noqa: F401
from wrappers import ObsWrapper
from stable_baselines3 import SAC
from train_sim import (
    PHASE_CONFIGS, CONSTANT_REW_SCALING, SPEED_LIMIT, RandomLanesWrapper,
    _PHASE_VEHICLES,
)


def make_env(phase: int = 3, lanes: int = None, render: bool = True):
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

    # Mirror train_sim.py phase-specific config exactly
    if phase == 0:
        config["lanes_count"]             = 2
        config["other_vehicles_type"]     = "highway_env.vehicle.behavior.LinearVehicle"
        config["initial_spacing"]         = 2
    if phase == 1:
        config["lanes_count"]             = 2
        config["p1a_obstacle"]            = True
        config["vehicles_count"]          = 0
        config["continuous_targets"]      = True
        config["require_obstacle_for_lc"] = True
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

    # --lanes flag overrides the phase default
    if lanes is not None:
        config["lanes_count"] = lanes

    env = gym.make("lane-changing-v0",
                   render_mode="human" if render else None,
                   config=config)
    # RandomLanesWrapper only for phases that trained with it (2, 3),
    # and only when no explicit --lanes override was given
    if lanes is None and phase not in (0, 1, 11, 15):
        env = RandomLanesWrapper(env)
    return ObsWrapper(env)


def pretty(x) -> str:
    if x is None:
        return "None"
    try:
        f = float(x)
        return "inf" if f == float("inf") else "-inf" if f == float("-inf") else f"{f:+.3f}"
    except Exception:
        return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="checkpoints/p2b/sim_baseline",
                        help="Path to checkpoint zip (omit .zip)")
    parser.add_argument("--phase",      type=int, default=3, choices=[0, 1, 11, 15, 2, 3])
    parser.add_argument("--episodes",   type=int, default=5)
    parser.add_argument("--lanes",      type=int, default=None,
                        help="Fix lane count (default: random 2-4 as in training)")
    parser.add_argument("--render",     action="store_true",
                        help="Open a pygame window to watch the agent drive")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample stochastically instead of taking the mean action")
    args = parser.parse_args()

    deterministic = not args.stochastic
    env = make_env(phase=args.phase, lanes=args.lanes, render=args.render)

    print(f"[eval_sim] Loading {args.model}.zip  phase={args.phase}  deterministic={deterministic}")
    model = SAC.load(args.model, env=env)
    print(f"[eval_sim] Running {args.episodes} episodes\n")

    successes  = 0
    collisions = 0
    ep_rewards = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        base = env.unwrapped
        print(f"── Episode {ep + 1}  start={base.vehicle.lane_index[2]}  "
              f"target={base.target_lane_index[2]} ──")

        total_reward   = 0.0
        term_totals    = {}
        lane_changed   = False
        crashed        = False
        ep_len         = 0
        last_shield    = {}

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_len       += 1

            rt  = info.get("reward_terms", {})
            lcs = info.get("lane_change_state", {})
            last_shield = info.get("shield", {})
            for k, v in rt.items():
                term_totals[k] = term_totals.get(k, 0.0) + v

            if lcs.get("lane_changing"):
                lane_changed = True
            if rt.get("collision", 0) < 0:
                crashed = True

            if terminated or truncated:
                reason  = "CRASH" if crashed else ("TRUNCATED" if truncated else "DONE")
                success = lane_changed and not crashed
                if success:
                    successes += 1
                if crashed:
                    collisions += 1
                ep_rewards.append(total_reward)

                print(f"  {reason}  len={ep_len}  reward={total_reward:+.2f}")
                print("  reward breakdown (sorted by magnitude):")
                for k, v in sorted(term_totals.items(), key=lambda x: abs(x[1]), reverse=True):
                    if k == "raw_jerk":
                        continue
                    bar = "█" * min(int(abs(v) * 0.3), 30)
                    print(f"    {k:42s} {v:+8.2f}  {bar}")
                if last_shield:
                    fwd = last_shield.get("fwd_interventions", 0)
                    lci = last_shield.get("lc_interventions", 0)
                    rate = last_shield.get("intervention_rate", 0.0)
                    min_fwd = last_shield.get("min_fwd_ttc", float("inf"))
                    min_lc  = last_shield.get("min_lc_ttc",  float("inf"))
                    print(f"  shield: fwd={fwd}  lc={lci}  rate={rate:.2f}/step  "
                          f"min_fwd_ttc={pretty(min_fwd)}  min_lc_ttc={pretty(min_lc)}")
                print()
                break

    print("══ Summary ══")
    print(f"  Episodes  : {args.episodes}")
    print(f"  Successes : {successes}/{args.episodes}  ({100*successes/args.episodes:.0f}%)")
    print(f"  Collisions: {collisions}/{args.episodes}")
    print(f"  Reward    : mean={np.mean(ep_rewards):+.1f}  "
          f"min={np.min(ep_rewards):+.1f}  max={np.max(ep_rewards):+.1f}")

    env.close()


if __name__ == "__main__":
    main()
