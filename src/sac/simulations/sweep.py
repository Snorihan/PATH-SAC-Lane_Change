"""
sweep.py — Optuna hyperparameter sweep for Phase 1.5 SAC training.

Searches over env/reward parameters and target_entropy. Prunes trials that
diverge early. Objective: ep_len_mean at 150k, with ent_coef <= 0.4 required.

Results are persisted to SQLite so the study survives interruption and can
be resumed or inspected with optuna-dashboard.

Usage (from src/sac/):
    python simulations/sweep.py
    python simulations/sweep.py --trials 12
    python simulations/sweep.py --trials 5 --resume checkpoints/phase15/sim_sac_224035_steps
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
import lanechange_env  # noqa: F401 — registers lane-changing-v0
from wrappers import ObsWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from train_sim import RandomLanesWrapper, PHASE_CONFIGS, SPEED_LIMIT

TIMESTEPS    = 150_000
PHASE        = 15
RESUME_PATH  = "checkpoints/sim_baseline"

# ── Pruning rules evaluated in order ──────────────────────────────────────────
# (step, metric, direction, threshold)
# direction "lt" → prune if metric < threshold (too low)
# direction "gt" → prune if metric > threshold (too high / diverging)
PRUNE_RULES = [
    (30_000,  "ep_len",      "lt", 60.0),
    (50_000,  "critic_loss", "gt", 2500.0),
    (80_000,  "ep_len",      "lt", 120.0),
    (80_000,  "ent_coef",    "gt", 0.70),
    (100_000, "critic_loss", "gt", 2000.0),
]


# ── Callback ──────────────────────────────────────────────────────────────────

class SweepCallback(BaseCallback):
    """Reports ep_len_mean to Optuna at pruning checkpoints and kills bad trials."""

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial
        self._rule_idx    = 0
        self.last_ep_len      = float("nan")
        self.last_critic_loss = float("nan")
        self.last_ent_coef    = float("nan")

    def _on_step(self) -> bool:
        self._refresh_metrics()

        if self._rule_idx >= len(PRUNE_RULES):
            return True

        rule_step, metric, direction, threshold = PRUNE_RULES[self._rule_idx]
        if self.num_timesteps < rule_step:
            return True

        self._rule_idx += 1

        value = {
            "ep_len":      self.last_ep_len,
            "critic_loss": self.last_critic_loss,
            "ent_coef":    self.last_ent_coef,
        }.get(metric, float("nan"))

        if not np.isnan(self.last_ep_len):
            self.trial.report(self.last_ep_len, step=self.num_timesteps)

        if not np.isnan(value):
            bad = (value < threshold) if direction == "lt" else (value > threshold)
            if bad:
                raise TrialPruned(
                    f"step={self.num_timesteps}: {metric}={value:.3f} {direction} {threshold}"
                )

        if self.trial.should_prune():
            raise TrialPruned()

        return True

    def _refresh_metrics(self):
        buf = getattr(self.model, "ep_info_buffer", [])
        if buf:
            lens = [ep["l"] for ep in buf if "l" in ep]
            if lens:
                self.last_ep_len = float(np.mean(lens))

        try:
            kv = self.model.logger.name_to_value
            v = kv.get("train/critic_loss")
            if v is not None:
                self.last_critic_loss = float(v)
            v = kv.get("train/ent_coef")
            if v is not None:
                self.last_ent_coef = float(v)
        except AttributeError:
            pass


# ── Env factory ───────────────────────────────────────────────────────────────

def make_env(params: dict) -> ObsWrapper:
    rewards = dict(PHASE_CONFIGS[PHASE])
    rewards["collision_penalty"] = params["collision_penalty"]
    rewards["ttc_weight"]        = params["ttc_weight"]

    env = gym.make("lane-changing-v0", config={
        "lane_width":         4.0,
        "road_length":        1000.0,
        "duration":           40,
        "policy_frequency":   15,
        "speed_limit":        SPEED_LIMIT,
        "continuous_targets": True,
        "vehicles_count":     10,
        "lc_ttc_gate":        params["lc_ttc_gate"],
        "fwd_ttc_gate":       params["fwd_ttc_gate"],
        "rewards":            rewards,
    })
    env = RandomLanesWrapper(env)
    return ObsWrapper(env)


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, resume: str, out_dir: str) -> float:
    params = {
        # <1s is too tight for a real lane change; >4s blocks the agent near any traffic.
        # trial 18 best hit 3.74 (ceiling) — consider raising upper bound to 6.0 next run.
        "lc_ttc_gate":       trial.suggest_float("lc_ttc_gate",       1.0,   4.0),
        "fwd_ttc_gate":      trial.suggest_float("fwd_ttc_gate",      1.0,   4.0),

        # -500 destabilised the critic; -100 was too weak. bracketed the known-good range.
        "collision_penalty": trial.suggest_float("collision_penalty", -500.0, -50.0),

        # Phase 1.5 default is 40; Phase 2 uses 80. search between conservative floor and Phase 2 value.
        "ttc_weight":        trial.suggest_float("ttc_weight",         20.0,  80.0),

        # SAC default is -dim(action_space) = -2.0. more negative → more deterministic policy.
        # range chosen to bracket the default and push toward ent_coef <= 0.4 requirement.
        "target_entropy":    trial.suggest_float("target_entropy",    -4.0,  -1.0),
    }

    env = make_env(params)
    model = SAC.load(resume, env=env)
    model.target_entropy = params["target_entropy"]

    trial_dir = os.path.join(out_dir, f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)

    cb = SweepCallback(trial)
    pruned = False

    try:
        model.learn(total_timesteps=TIMESTEPS, callback=cb, reset_num_timesteps=False)
    except TrialPruned:
        pruned = True
    except Exception as e:
        pruned = True
        print(f"[sweep] trial {trial.number} crashed: {e}")
    finally:
        try:
            model.save(os.path.join(trial_dir, "model"))
        except Exception:
            pass
        with open(os.path.join(trial_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2)
        with open(os.path.join(trial_dir, "final_metrics.json"), "w") as f:
            json.dump({
                "ep_len":      cb.last_ep_len,
                "ent_coef":    cb.last_ent_coef,
                "critic_loss": cb.last_critic_loss,
                "pruned":      pruned,
            }, f, indent=2)
        env.close()

    if pruned:
        raise TrialPruned()

    ep_len   = cb.last_ep_len
    ent_coef = cb.last_ent_coef

    if np.isnan(ep_len):
        return 0.0

    # Hard requirement: ent_coef <= 0.4 by end of run
    if not np.isnan(ent_coef) and ent_coef > 0.4:
        print(f"[sweep] trial {trial.number} failed ent_coef constraint: {ent_coef:.3f} > 0.4")
        return 0.0

    return float(ep_len)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=10,
                        help="Max number of trials to run")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Hard stop after N hours (study saved, current trial finishes)")
    parser.add_argument("--resume",  type=str, default=RESUME_PATH,
                        help="Checkpoint to resume each trial from")
    parser.add_argument("--out-dir", type=str, default="checkpoints/sweep",
                        help="Directory for trial checkpoints and results")
    parser.add_argument("--study",   type=str, default="phase15_sweep",
                        help="Optuna study name (persisted to SQLite)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # MedianPruner: after 3 startup trials, prune if below median at each checkpoint
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=40_000, interval_steps=10_000)
    study = optuna.create_study(
        study_name=args.study,
        direction="maximize",
        pruner=pruner,
        storage=f"sqlite:///{args.out_dir}/sweep.db",
        load_if_exists=True,
    )

    print(f"[sweep] Study: {args.study}  trials={args.trials}  resume={args.resume}")
    print(f"[sweep] Results → {args.out_dir}/sweep.db")
    print(f"[sweep] Pruning rules:")
    for step, metric, direction, thresh in PRUNE_RULES:
        print(f"         {step:>7,} steps  {metric} {direction} {thresh}")
    print()

    timeout_secs = args.timeout * 3600 if args.timeout else None
    study.optimize(
        lambda trial: objective(trial, args.resume, args.out_dir),
        n_trials=args.trials,
        timeout=timeout_secs,
        catch=(Exception,),
    )

    # ── Report ────────────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.value is not None and t.value > 0]
    if not completed:
        print("[sweep] No trials completed successfully.")
        return

    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"  trial #     : {best.number}")
    print(f"  ep_len_mean : {best.value:.1f}")
    print(f"  params:")
    for k, v in best.params.items():
        print(f"    {k:22s} = {v:.4f}")

    best_params_path = os.path.join(args.out_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\n  Saved to {best_params_path}")
    print(f"  Best model  : {args.out_dir}/trial_{best.number:03d}/model.zip")


if __name__ == "__main__":
    main()
