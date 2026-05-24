"""
run_pipeline.py — full P1 → P1.5 → P2 training pipeline.

Trains from scratch with continuous_targets=False throughout.
All phases share the same MDP structure — no critic distributional shift between phases.

Run from src/sac/:
    python simulations/run_pipeline.py

Resume-safe: if a phase checkpoint already exists, that phase is skipped automatically.
Ctrl+C during any phase saves the current checkpoint and aborts.
"""

import subprocess
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────

P1_CHECKPOINT_DIR  = "checkpoints/p1_v2"
P1_TIMESTEPS       = 150_000

P15_CHECKPOINT_DIR = "checkpoints/p15_v3"
P15_TIMESTEPS      = 200_000

P2_CHECKPOINT_DIR  = "checkpoints/p2_v2"
P2_TIMESTEPS       = 400_000

# ── Commands ──────────────────────────────────────────────────────────────────

P1_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "1",
    "--timesteps",      str(P1_TIMESTEPS),
    "--checkpoint-dir", P1_CHECKPOINT_DIR,
]

P15_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "15",
    "--resume",         f"{P1_CHECKPOINT_DIR}/sim_baseline",
    "--timesteps",      str(P15_TIMESTEPS),
    "--checkpoint-dir", P15_CHECKPOINT_DIR,
    "--learning-rate",  "1e-4",
    "--target-entropy", "-1.0",
    "--init-ent-coef",  "1.0",
]

P2_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "2",
    "--resume",         f"{P15_CHECKPOINT_DIR}/sim_baseline",
    "--timesteps",      str(P2_TIMESTEPS),
    "--checkpoint-dir", P2_CHECKPOINT_DIR,
    "--learning-rate",  "1e-4",
    "--max-grad-norm",  "0.5",
    "--target-entropy", "-1.0",
]

# ── Runner ────────────────────────────────────────────────────────────────────

def run_phase(cmd, label):
    print(f"\n{'=' * 64}")
    print(f"  {label}")
    print(f"{'=' * 64}")
    print(f"  {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"\n[pipeline] Interrupted during: {label}")
        print("[pipeline] Checkpoint saved by train_sim. Pipeline aborted.")
        sys.exit(1)
    if result.returncode not in (0, 1):
        print(f"\n[pipeline] {label} exited with code {result.returncode} — aborting.")
        sys.exit(result.returncode)


def check_exists(path, label):
    if not os.path.exists(path):
        print(f"\n[pipeline] ERROR: {path} not found after {label}.")
        print("[pipeline] Training may have been interrupted before saving. Aborting.")
        sys.exit(1)


def run_or_skip(cmd, label, output_zip):
    if os.path.exists(output_zip):
        print(f"\n[pipeline] {label}: checkpoint found at {output_zip} — skipping.")
        return
    run_phase(cmd, label)
    check_exists(output_zip, label)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_or_skip(
        P1_CMD,
        f"Phase 1   — no traffic, single target  ({P1_TIMESTEPS:,} steps)",
        f"{P1_CHECKPOINT_DIR}/sim_baseline.zip",
    )

    run_or_skip(
        P15_CMD,
        f"Phase 1.5 — traffic bridge             ({P15_TIMESTEPS:,} steps)",
        f"{P15_CHECKPOINT_DIR}/sim_baseline.zip",
    )

    run_phase(
        P2_CMD,
        f"Phase 2   — full reward suite          ({P2_TIMESTEPS:,} steps)",
    )

    print("\n[pipeline] All phases complete.")
    print("[pipeline] Eval with:")
    print(f"  python simulations/eval_sim.py --model {P2_CHECKPOINT_DIR}/sim_baseline --render --phase 2 --episodes 10")
