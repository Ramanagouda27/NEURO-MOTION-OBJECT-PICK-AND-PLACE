"""
run_task1.py  –  One-click runner for Task 1: Object Pick and Place
Physical AI Hackathon 2026

Runs the full pipeline in sequence:
  Step 1  Verify environment / install deps
  Step 2  Run scripted simulation demo (with viewer)
  Step 3  Collect training demonstrations (headless)
  Step 4  Train ACT policy on demonstrations
  Step 5  Evaluate trained policy

Usage:
    python run_task1.py                     # full pipeline
    python run_task1.py --step sim          # only scripted sim
    python run_task1.py --step collect      # only collect demos
    python run_task1.py --step train        # only train
    python run_task1.py --step eval         # only evaluate
    python run_task1.py --quick             # quick smoke-test (fast)
"""

import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str], desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with code {result.returncode}")
        sys.exit(result.returncode)


def step_sim(quick: bool):
    """Run the scripted pick-and-place simulation."""
    cmd = [sys.executable, "pick_place_sim.py", "--num_trials", "1"]
    if quick:
        cmd += ["--headless"]
    run(cmd, "Step 2 – Scripted Pick-and-Place Simulation")


def step_collect(quick: bool):
    """Collect demonstration episodes."""
    n = "5" if quick else "50"
    run(
        [sys.executable, "collect_demos.py",
         "--num_demos", n,
         "--output_dir", "demo_data",
         "--headless"],
        f"Step 3 – Collecting {n} Demonstrations"
    )


def step_train(quick: bool):
    """Train the ACT policy."""
    epochs = "10" if quick else "200"
    run(
        [sys.executable, "train_act_policy.py",
         "--mode", "standalone",
         "--data_dir", "demo_data",
         "--checkpoint_dir", "checkpoints",
         "--epochs", epochs,
         "--batch_size", "32" if quick else "64"],
        f"Step 4 – Training ACT Policy ({epochs} epochs)"
    )


def step_eval():
    """Evaluate the trained policy."""
    run(
        [sys.executable, "train_act_policy.py",
         "--mode", "eval",
         "--checkpoint", "checkpoints/best.pt",
         "--eval_episodes", "5"],
        "Step 5 – Evaluating Trained Policy"
    )


def step_smoke_test():
    """Quick environment smoke-test (no training)."""
    run(
        [sys.executable, "lerobot_env.py"],
        "Smoke-test: LeRobot Gym Environment"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Task 1 Pick-and-Place – Full Pipeline Runner"
    )
    parser.add_argument(
        "--step",
        choices=["sim", "collect", "train", "eval", "smoke"],
        default=None,
        help="Run only a specific step (default: full pipeline)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer demos, fewer epochs (for testing)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Physical AI Hackathon 2026")
    print("  Task 1 – Object Pick and Place (LeRobot SO101 / MuJoCo)")
    print("=" * 60)

    if args.step == "smoke":
        step_smoke_test()
    elif args.step == "sim":
        step_sim(args.quick)
    elif args.step == "collect":
        step_collect(args.quick)
    elif args.step == "train":
        step_train(args.quick)
    elif args.step == "eval":
        step_eval()
    else:
        # Full pipeline
        step_smoke_test()
        step_sim(quick=True)
        step_collect(args.quick)
        step_train(args.quick)
        step_eval()

    print("\n" + "=" * 60)
    print("  All steps completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
