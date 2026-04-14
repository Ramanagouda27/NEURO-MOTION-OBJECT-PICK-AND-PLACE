"""
LeRobot-Compatible Gymnasium Environment — Task 1: Pick and Place
Physical AI Hackathon 2026

Uses the REAL SO101 joint names from:
  /home/hacker/workspace/src/so101_description/urdf/so101.urdf

Joints (6 DOF):
  shoulder_pan  | shoulder_lift | elbow_flex
  wrist_flex    | wrist_roll    | gripper  (single revolute jaw)

Compatible with LeRobot ACT / Diffusion Policy / TDMPC2.
"""

import os
from typing import Optional

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_scene.xml")

# ── Real SO101 joint names (from official URDF) ──────────────────────────────
ARM_JOINT_NAMES    = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                      "wrist_flex",   "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"   # single revolute jaw

N_ARM_JOINTS = 5
N_GRIPPER    = 1
ACTION_DIM   = N_ARM_JOINTS + N_GRIPPER   # 6

# Gripper angles (rad) — from URDF limits
GRIPPER_OPEN   = 1.4    # jaw open
GRIPPER_CLOSED = 0.05   # jaw clamped

# Observation layout (24-D):
#  [0:5]   arm joint positions
#  [5:10]  arm joint velocities
#  [10]    gripper jaw angle
#  [11:14] ee position (XYZ)
#  [14:18] ee quaternion
#  [18:21] object position (XYZ)
#  [21:24] relative position (obj - ee)
OBS_DIM = 5 + 5 + 1 + 3 + 4 + 3 + 3   # = 24


# ─────────────────────────────────────────────────────────────────────────────

class SO101PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for SO101 Object Pick and Place.

    Parameters
    ----------
    render_mode        : "human" | "rgb_array" | None
    max_episode_steps  : int  (default 300)
    randomise_object   : bool (default True)
    success_threshold  : float  metres (default 0.05)
    scene_xml          : path to MJCF scene file
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 300,
        randomise_object: bool = True,
        success_threshold: float = 0.05,
        scene_xml: str = SCENE_XML,
    ):
        super().__init__()

        self.render_mode       = render_mode
        self.max_episode_steps = max_episode_steps
        self.randomise_object  = randomise_object
        self.success_threshold = success_threshold
        self._step_count       = 0

        # ── Load MuJoCo model ──────────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)

        # ── Cache MuJoCo ids ───────────────────────────────────────────────
        self._ee_site_id  = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._obj_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_site")
        self._tgt_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Real SO101 arm joint ids
        self._arm_jnt_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ARM_JOINT_NAMES
        ]
        self._gripper_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT_NAME)

        # Actuator indices (match order in XML)
        # 0=shoulder_pan, 1=shoulder_lift, 2=elbow_flex,
        # 3=wrist_flex, 4=wrist_roll, 5=gripper
        self._gripper_act = 5

        self._obj_fj = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")

        # ── Gymnasium spaces ───────────────────────────────────────────────
        # Action: [5 arm joints + 1 gripper] normalised to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # ── Renderer ──────────────────────────────────────────────────────
        self._renderer: Optional[mujoco.Renderer] = None
        self._viewer   = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._rng = np.random.default_rng(seed=None)

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        if self.randomise_object:
            adr = self.model.jnt_qposadr[self._obj_fj]
            self.data.qpos[adr + 0] = self._rng.uniform(0.13, 0.24)
            self.data.qpos[adr + 1] = self._rng.uniform(-0.10, 0.10)
            self.data.qpos[adr + 2] = 0.432
            self.data.qpos[adr + 3] = 1.0
            self.data.qpos[adr + 4:adr + 7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self._step_count += 1

        # ── Apply action ────────────────────────────────────────────────────
        # Arm joints: de-normalise [-1,1] → actual joint range
        for i, jid in enumerate(self._arm_jnt_ids):
            lo  = self.model.actuator_ctrlrange[i, 0]
            hi  = self.model.actuator_ctrlrange[i, 1]
            cmd = float(np.clip(action[i], -1.0, 1.0))
            self.data.ctrl[i] = lo + (cmd + 1.0) / 2.0 * (hi - lo)

        # Gripper: de-normalise [-1,1] → [GRIPPER_CLOSED, GRIPPER_OPEN]
        g  = float(np.clip(action[5], -1.0, 1.0))
        gw = GRIPPER_CLOSED + (g + 1.0) / 2.0 * (GRIPPER_OPEN - GRIPPER_CLOSED)
        self.data.ctrl[self._gripper_act] = gw

        # Simulate 5 sub-steps for stability
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        reward     = self._compute_reward()
        info       = self._get_info()
        terminated = bool(info["success"])
        truncated  = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer:
            self._renderer.close()
        if self._viewer:
            self._viewer.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the 24-D observation vector."""
        # Arm joint positions & velocities
        q  = np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                        for j in self._arm_jnt_ids], dtype=np.float32)
        dq = np.array([self.data.qvel[self.model.jnt_dofadr[j]]
                        for j in self._arm_jnt_ids], dtype=np.float32)

        # Gripper jaw angle
        gw = np.array([self.data.qpos[
            self.model.jnt_qposadr[self._gripper_jnt_id]
        ]], dtype=np.float32)

        # EE pose
        ee_pos  = self.data.site_xpos[self._ee_site_id].astype(np.float32)
        ee_mat  = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        ee_quat = np.zeros(4, dtype=np.float32)
        mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())

        # Object position
        obj_pos = self.data.site_xpos[self._obj_site_id].astype(np.float32)
        rel_pos = (obj_pos - ee_pos).astype(np.float32)

        return np.concatenate([q, dq, gw, ee_pos, ee_quat, obj_pos, rel_pos])

    def _compute_reward(self) -> float:
        ee_pos  = self.data.site_xpos[self._ee_site_id]
        obj_pos = self.data.site_xpos[self._obj_site_id]
        tgt_pos = self.data.site_xpos[self._tgt_site_id]

        d_reach     = np.linalg.norm(ee_pos - obj_pos)
        d_transport = np.linalg.norm(obj_pos[:2] - tgt_pos[:2])

        reach_rew     = 1.0 - np.tanh(5.0 * d_reach)
        transport_rew = 1.0 - np.tanh(5.0 * d_transport)

        # Grasp bonus: gripper closed and near object
        gw        = self.data.qpos[self.model.jnt_qposadr[self._gripper_jnt_id]]
        grasp_rew = 0.5 if (gw < 0.2 and d_reach < 0.05) else 0.0

        success_rew = 10.0 if d_transport < self.success_threshold else 0.0

        return float(reach_rew + grasp_rew + transport_rew + success_rew)

    def _get_info(self) -> dict:
        obj_pos = self.data.site_xpos[self._obj_site_id]
        tgt_pos = self.data.site_xpos[self._tgt_site_id]
        d       = float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2]))
        return {
            "placement_error_m": d,
            "success":           d < self.success_threshold,
            "object_pos":        obj_pos.tolist(),
            "target_pos":        tgt_pos.tolist(),
        }

    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    # ─────────────────────────────────────────────────────────────────────────
    # LeRobot helpers
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def unwrapped(self):
        return self

    def get_obs_dict(self) -> dict:
        return {"observation.state": self._get_obs().tolist()}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Smoke-testing SO101PickPlaceEnv …")
    env = SO101PickPlaceEnv(render_mode=None, max_episode_steps=50)
    obs, info = env.reset(seed=0)
    print(f"  obs shape : {obs.shape}  (expected ({OBS_DIM},))")
    assert obs.shape == (OBS_DIM,), f"Wrong obs shape: {obs.shape}"

    total_r = 0.0
    for step in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    env.close()
    print(f"  steps: {step+1}  total_reward: {total_r:.2f}")
    print(f"  info : {info}")
    print("Smoke-test PASSED ✓")
