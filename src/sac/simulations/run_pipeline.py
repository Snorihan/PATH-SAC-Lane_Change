"""
run_pipeline.py — sequential P1.5 → P2f training pipeline.

Run from src/sac/:
    python simulations/run_pipeline.py

Ctrl+C during either phase saves the current checkpoint and aborts —
it will NOT auto-advance to the next phase if interrupted.
"""

import subprocess
import sys
import os

# ── Config ────────────────────────────────────────────────────────────────────

P1_CHECKPOINT    = "checkpoints/sim_baseline"

P15_CHECKPOINT_DIR = "checkpoints/p15_v2"
P15_TIMESTEPS      = 200_000

P2F_CHECKPOINT_DIR = "checkpoints/p2f"
P2F_TIMESTEPS      = 400_000

# ── Commands ──────────────────────────────────────────────────────────────────

P15_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "15",
    "--resume",         P1_CHECKPOINT,
    "--timesteps",      str(P15_TIMESTEPS),
    "--checkpoint-dir", P15_CHECKPOINT_DIR,
    "--learning-rate",  "1e-4",
    "--target-entropy", "-1.0",
]

P2F_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "2",
    "--resume",         f"{P15_CHECKPOINT_DIR}/sim_baseline",
    "--timesteps",      str(P2F_TIMESTEPS),
    "--checkpoint-dir", P2F_CHECKPOINT_DIR,
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
        print(f"\n[pipeline] {label} exited with unexpected code {result.returncode} — aborting.")
        sys.exit(result.returncode)


def check_exists(path, label):
    if not os.path.exists(path):
        print(f"\n[pipeline] ERROR: {path} not found after {label}.")
        print("[pipeline] Training may have been interrupted before saving. Aborting.")
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Verify p1_scaled exists before starting
    if not os.path.exists(f"{P1_CHECKPOINT}.zip"):
        print(f"[pipeline] ERROR: base checkpoint {P1_CHECKPOINT}.zip not found.")
        sys.exit(1)

    run_phase(P15_CMD, f"Phase 1.5 — traffic bridge  ({P15_TIMESTEPS:,} steps)")

    p15_out = f"{P15_CHECKPOINT_DIR}/sim_baseline.zip"
    check_exists(p15_out, "Phase 1.5")

    run_phase(P2F_CMD, f"Phase 2f  — full reward suite ({P2F_TIMESTEPS:,} steps)")

    print("\n[pipeline] Both phases complete.")
    print(f"[pipeline] Eval with:")
    print(f"  python simulations/eval_sim.py --model {P2F_CHECKPOINT_DIR}/sim_baseline --render --phase 2 --episodes 10")
