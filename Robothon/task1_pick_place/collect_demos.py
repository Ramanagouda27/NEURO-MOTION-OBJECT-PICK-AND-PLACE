"""
Demonstration Data Collection for Task 1 – Pick and Place
Physical AI Hackathon 2026

Uses the scripted pick-and-place controller to automatically collect
high-quality demonstration episodes and saves them in the LeRobot
dataset format (HuggingFace datasets + LeRobotDataset schema).

The saved dataset can be loaded directly by LeRobot for training
ACT, Diffusion Policy, or TDMPC2.

Usage:
    # Collect 50 demos, save to ./demo_data/
    python collect_demos.py --num_demos 50 --output_dir demo_data

    # Headless (faster)
    python collect_demos.py --num_demos 100 --headless

    # Push to HuggingFace Hub (requires huggingface-cli login)
    python collect_demos.py --num_demos 50 --push_to_hub your-name/so101-pick-place
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import mujoco

from lerobot_env import SO101PickPlaceEnv, ACTION_DIM, OBS_DIM
from pick_place_sim import (
    SO101Controller,
    pick_and_place,
    reset_episode,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    MOVE_STEPS,
    SETTLE_STEPS,
    CTRL_FREQ,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scripted action generator
# ─────────────────────────────────────────────────────────────────────────────

def scripted_action_generator(env: SO101PickPlaceEnv):
    """
    Generator that yields (action, phase_label) pairs by running the
    scripted policy inside the environment's simulation.

    Yields one action per env.step() call.
    """
    model = env.model
    data  = env.data
    ctrl  = SO101Controller(model, data)

    obj_pos = ctrl.get_object_pos()
    tgt_pos = ctrl.get_target_pos()

    HOVER_H = 0.12
    GRASP_H = 0.02

    phases = [
        # (target_ee_pos,                                gripper_w,     phase_name,   n_steps)
        (obj_pos + [0, 0, HOVER_H],                      GRIPPER_OPEN,  "approach",   MOVE_STEPS),
        (obj_pos + [0, 0, GRASP_H],                      GRIPPER_OPEN,  "lower",      int(MOVE_STEPS*0.6)),
        (obj_pos + [0, 0, GRASP_H],                      GRIPPER_CLOSED,"grasp",      SETTLE_STEPS),
        (obj_pos + [0, 0, HOVER_H + 0.05],               GRIPPER_CLOSED,"lift",       MOVE_STEPS),
        (tgt_pos  + [0, 0, HOVER_H],                     GRIPPER_CLOSED,"transport",  MOVE_STEPS),
        (tgt_pos  + [0, 0, GRASP_H],                     GRIPPER_CLOSED,"place",      int(MOVE_STEPS*0.6)),
        (tgt_pos  + [0, 0, GRASP_H],                     GRIPPER_OPEN,  "release",    SETTLE_STEPS),
        (tgt_pos  + [0, 0, HOVER_H],                     GRIPPER_OPEN,  "retreat",    int(MOVE_STEPS*0.5)),
    ]

    for target_pos, gw, phase_name, n_steps in phases:
        q_start  = ctrl.get_joint_angles()
        q_target = ctrl.ik(np.array(target_pos), q_init=q_start)

        for i in range(n_steps):
            alpha   = (i + 1) / n_steps
            alpha_s = alpha * alpha * (3 - 2 * alpha)
            q_cmd   = q_start + alpha_s * (q_target - q_start)

            # Build normalised action
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            for k in range(5):
                lo = model.actuator_ctrlrange[k, 0]
                hi = model.actuator_ctrlrange[k, 1]
                action[k] = float(
                    np.clip(2.0 * (q_cmd[k] - lo) / (hi - lo) - 1.0, -1.0, 1.0)
                )
            # Gripper normalised
            action[5] = float(np.clip(2.0 * gw / 0.035 - 1.0, -1.0, 1.0))

            yield action, phase_name

            # Apply to simulation for kinematics consistency
            for k in range(5):
                lo = model.actuator_ctrlrange[k, 0]
                hi = model.actuator_ctrlrange[k, 1]
                data.ctrl[k] = lo + (action[k] + 1.0) / 2.0 * (hi - lo)
            data.ctrl[5] = gw
            data.ctrl[6] = gw
            mujoco.mj_step(model, data)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset writer (LeRobot-compatible schema)
# ─────────────────────────────────────────────────────────────────────────────

class LeRobotDemoWriter:
    """
    Writes demonstration episodes in LeRobot's flat file format:

    output_dir/
    ├── meta_data.json
    ├── data/
    │   ├── episode_000000.npz
    │   ├── episode_000001.npz
    │   └── …
    └── stats.json   (computed at finalise())

    Each .npz contains arrays:
        observation.state   (T, OBS_DIM)
        action              (T, ACTION_DIM)
        reward              (T,)
        done                (T,)   bool
        phase               (T,)   str (phase label)
    """

    def __init__(self, output_dir: str, fps: int = 50):
        self.output_dir = Path(output_dir)
        self.fps        = fps
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        self._episodes: list[dict] = []
        self._episode_idx = 0

    def write_episode(self, transitions: list[dict]) -> str:
        """Save one episode. Returns filepath."""
        obs_list    = np.array([t["obs"]    for t in transitions], dtype=np.float32)
        act_list    = np.array([t["action"] for t in transitions], dtype=np.float32)
        rew_list    = np.array([t["reward"] for t in transitions], dtype=np.float32)
        done_list   = np.array([t["done"]   for t in transitions], dtype=bool)
        phase_list  = np.array([t["phase"]  for t in transitions])

        fname = self.output_dir / "data" / f"episode_{self._episode_idx:06d}.npz"
        np.savez_compressed(
            fname,
            **{
                "observation.state": obs_list,
                "action":            act_list,
                "reward":            rew_list,
                "done":              done_list,
                "phase":             phase_list,
            }
        )

        self._episodes.append({
            "episode_index": self._episode_idx,
            "file":          str(fname),
            "num_frames":    len(transitions),
            "success":       bool(done_list[-1]),
        })
        self._episode_idx += 1
        return str(fname)

    def finalise(self, env: SO101PickPlaceEnv):
        """Write meta_data.json and stats.json."""
        # Metadata
        meta = {
            "robot_type":          "so101",
            "task":                "pick_place",
            "fps":                 self.fps,
            "num_episodes":        self._episode_idx,
            "total_frames":        sum(e["num_frames"] for e in self._episodes),
            "obs_dim":             OBS_DIM,
            "action_dim":          ACTION_DIM,
            "success_rate":        float(
                sum(e["success"] for e in self._episodes) / max(1, self._episode_idx)
            ),
            "observation_space":   {
                "joint_pos":    [0, 5],
                "joint_vel":    [5, 10],
                "gripper_pos":  [10, 11],
                "ee_pos":       [11, 14],
                "ee_quat":      [14, 18],
                "obj_pos":      [18, 21],
                "rel_pos":      [21, 24],
            },
            "action_space": {
                "joint_cmds":   [0, 5],
                "gripper_cmd":  [5, 6],
                "note":         "normalised to [-1, 1]",
            },
            "episodes": self._episodes,
        }
        with open(self.output_dir / "meta_data.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Compute running statistics (mean/std for normalisation)
        all_obs  = []
        all_acts = []
        for ep in self._episodes:
            d = np.load(ep["file"])
            all_obs.append(d["observation.state"])
            all_acts.append(d["action"])

        if all_obs:
            obs_arr  = np.concatenate(all_obs,  axis=0)
            act_arr  = np.concatenate(all_acts, axis=0)
            stats = {
                "observation.state": {
                    "mean": obs_arr.mean(axis=0).tolist(),
                    "std":  obs_arr.std(axis=0).clip(1e-3).tolist(),
                    "min":  obs_arr.min(axis=0).tolist(),
                    "max":  obs_arr.max(axis=0).tolist(),
                },
                "action": {
                    "mean": act_arr.mean(axis=0).tolist(),
                    "std":  act_arr.std(axis=0).clip(1e-3).tolist(),
                    "min":  act_arr.min(axis=0).tolist(),
                    "max":  act_arr.max(axis=0).tolist(),
                },
            }
            with open(self.output_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)

        print(f"\nDataset saved to: {self.output_dir}")
        print(f"  Episodes : {self._episode_idx}")
        print(f"  Frames   : {meta['total_frames']}")
        print(f"  Success  : {meta['success_rate']*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Push to HuggingFace Hub (optional)
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hub(output_dir: str, repo_id: str):
    """Upload dataset to HuggingFace Hub using huggingface_hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"\nDataset pushed to: https://huggingface.co/datasets/{repo_id}")
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"Push failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect(args):
    env    = SO101PickPlaceEnv(
        render_mode=None if args.headless else "human",
        max_episode_steps=600,
        randomise_object=not args.no_random,
    )
    writer = LeRobotDemoWriter(args.output_dir, fps=50)
    rng    = np.random.default_rng(seed=0)

    n_success = 0
    t_start   = time.time()

    for ep_idx in range(args.num_demos):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        transitions = []

        gen = scripted_action_generator(env)

        done = False
        while not done:
            try:
                action, phase = next(gen)
            except StopIteration:
                break

            obs_next, reward, terminated, truncated, info = env.step(action)

            transitions.append({
                "obs":    obs,
                "action": action,
                "reward": reward,
                "done":   terminated or truncated,
                "phase":  phase,
            })

            obs  = obs_next
            done = terminated or truncated

        if not transitions:
            continue

        success = info.get("success", False)
        if success:
            n_success += 1

        ep_file = writer.write_episode(transitions)
        elapsed = time.time() - t_start

        print(
            f"  ep {ep_idx+1:4d}/{args.num_demos}  "
            f"frames={len(transitions):4d}  "
            f"err={info['placement_error_m']*100:5.1f}cm  "
            f"{'✓' if success else '✗'}  "
            f"({n_success}/{ep_idx+1} ok)  "
            f"{elapsed:.0f}s"
        )

        if args.only_success and not success:
            # Overwrite (don't count failed demos)
            writer._episode_idx -= 1
            writer._episodes.pop()
            import shutil; shutil.rmtree(ep_file, ignore_errors=True)
            os.remove(ep_file)

    writer.finalise(env)
    env.close()

    if args.push_to_hub:
        push_to_hub(args.output_dir, args.push_to_hub)


def main():
    parser = argparse.ArgumentParser(
        description="Collect scripted demos for SO101 pick-and-place"
    )
    parser.add_argument("--num_demos",    type=int, default=50,
                        help="Number of episodes to collect")
    parser.add_argument("--output_dir",   type=str, default="demo_data",
                        help="Directory to save the dataset")
    parser.add_argument("--headless",     action="store_true",
                        help="Run without GUI")
    parser.add_argument("--no_random",    action="store_true",
                        help="Disable object randomisation")
    parser.add_argument("--only_success", action="store_true",
                        help="Only save successful episodes")
    parser.add_argument("--push_to_hub",  type=str, default=None,
                        metavar="HF_REPO_ID",
                        help="Push dataset to HuggingFace Hub repo")
    args = parser.parse_args()

    print("=" * 60)
    print("  Task 1 Demo Collection – SO101 Pick and Place")
    print(f"  Collecting {args.num_demos} episodes → {args.output_dir}")
    print("=" * 60)

    collect(args)


if __name__ == "__main__":
    main()
