"""
run_pipeline.py — 1A → 1B → 1.5 → 2 → 3 training pipeline.

Phase 1  (1A) — single stopped obstacle, adjacent lane open. Teaches clean LC mechanics.
Phase 11 (1B) — two staggered stopped obstacles. Teaches repeated justified LCs.
Phase 15 (1.5)— 75% follow / 25% escape. Teaches when to LC vs when to follow.
Phase 2       — 10 IDM vehicles. Gap selection in real traffic.
Phase 3       — full reward suite. Comfort + efficiency refinement.

Run from src/sac/:
    python simulations/run_pipeline.py

Resume-safe: if a phase checkpoint already exists, that phase is skipped automatically.
Ctrl+C during any phase saves the current checkpoint and aborts.
"""

import subprocess
import sys
import os

# ── Checkpoint dirs ───────────────────────────────────────────────────────────

P1A_CHECKPOINT_DIR = "checkpoints/p1a_v4"
P1A_TIMESTEPS      = 100_000

P1B_CHECKPOINT_DIR = "checkpoints/p1b_v1"
P1B_TIMESTEPS      = 150_000

P15_CHECKPOINT_DIR = "checkpoints/p15_v6"
P15_TIMESTEPS      = 150_000

P2_CHECKPOINT_DIR  = "checkpoints/p2_v7"
P2_TIMESTEPS       = 200_000

P3_CHECKPOINT_DIR  = "checkpoints/p3_v9"
P3_TIMESTEPS       = 400_000

# ── Commands ──────────────────────────────────────────────────────────────────

P1A_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "1",
    "--timesteps",      str(P1A_TIMESTEPS),
    "--checkpoint-dir", P1A_CHECKPOINT_DIR,
    "--init-ent-coef",  "1.0",
    "--target-entropy", "-0.5",
]

P1B_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "11",
    "--resume",         f"{P1A_CHECKPOINT_DIR}/sim_baseline",
    "--timesteps",      str(P1B_TIMESTEPS),
    "--checkpoint-dir", P1B_CHECKPOINT_DIR,
    "--learning-rate",  "1e-4",
    "--target-entropy", "-1.0",
    "--init-ent-coef",  "1.0",
]

P15_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "15",
    "--resume",         f"{P1B_CHECKPOINT_DIR}/sim_baseline",
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
    "--target-entropy", "-1.0",
    "--init-ent-coef",  "1.0",
]

P3_CMD = [
    sys.executable, "simulations/train_sim.py",
    "--phase",          "3",
    "--resume",         f"{P2_CHECKPOINT_DIR}/sim_baseline",
    "--timesteps",      str(P3_TIMESTEPS),
    "--checkpoint-dir", P3_CHECKPOINT_DIR,
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
        P1A_CMD,
        f"Phase 1A  — single stopped obstacle, clean LC       ({P1A_TIMESTEPS:,} steps)",
        f"{P1A_CHECKPOINT_DIR}/sim_baseline.zip",
    )

    run_or_skip(
        P1B_CMD,
        f"Phase 1B  — two staggered obstacles, repeated LC    ({P1B_TIMESTEPS:,} steps)",
        f"{P1B_CHECKPOINT_DIR}/sim_baseline.zip",
    )

    run_or_skip(
        P15_CMD,
        f"Phase 1.5 — follow vs LC judgment (75/25 mix)       ({P15_TIMESTEPS:,} steps)",
        f"{P15_CHECKPOINT_DIR}/sim_baseline.zip",
    )

    run_or_skip(
        P2_CMD,
        f"Phase 2   — 10 IDM vehicles                         ({P2_TIMESTEPS:,} steps)",
        f"{P2_CHECKPOINT_DIR}/sim_baseline.zip",
    )

    run_phase(
        P3_CMD,
        f"Phase 3   — full reward suite                        ({P3_TIMESTEPS:,} steps)",
    )

    print("\n[pipeline] All phases complete.")
    print("[pipeline] Eval with:")
    print(f"  python simulations/eval_sim.py --model {P3_CHECKPOINT_DIR}/sim_baseline --render --phase 3 --episodes 10")
