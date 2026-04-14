#!/usr/bin/env python3
"""
Gymnasium Environment Wrapper for SO101 Pick and Place
======================================================
Compatible with LeRobot's evaluation and training pipeline.

Usage:
    import gymnasium as gym
    from envs.so101_pick_place_env import SO101PickPlaceEnv
    
    env = SO101PickPlaceEnv(render_mode="human")
    obs, info = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import os
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import mujoco

SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "assets", "so101_scene.xml")


class SO101PickPlaceEnv(gym.Env):
    """
    SO101 Pick and Place Environment for Task 1.
    
    Observation Space:
        - Joint positions (7): 5 arm joints + 2 gripper joints
        - Gripper position (3): XYZ in world frame
        - Object position (3): XYZ in world frame
        - Target position (3): XYZ in world frame
        - Object-gripper distance (1)
        - Object-target distance (1)
        Total: 18 dims
    
    Action Space:
        - Joint velocity targets (5): for 5 arm DOF
        - Gripper command (1): 0=close, 1=open
        Total: 6 dims
    
    Reward:
        - Shaped reward based on progress through phases:
          1. Approach: -distance(gripper, object)
          2. Grasp: bonus for contact
          3. Lift: bonus for object height
          4. Transport: -distance(object, target)
          5. Place: bonus for accuracy
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None, max_steps=1000, scene_xml=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Load MuJoCo
        xml_path = scene_xml or SCENE_XML
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
        )
        
        # Renderer
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        
        # State
        self.initial_obj_pos = None
        self.grasped = False
        self.lifted = False
        
    def _get_obs(self):
        """Construct observation vector."""
        joint_pos = self.data.sensordata[:7]
        gripper_pos = self.data.sensordata[7:10]
        object_pos = self.data.sensordata[10:13]
        target_pos = self.data.sensordata[13:16]
        
        grip_obj_dist = np.array([np.linalg.norm(gripper_pos - object_pos)])
        obj_tgt_dist = np.array([np.linalg.norm(object_pos[:2] - target_pos[:2])])
        
        return np.concatenate([
            joint_pos, gripper_pos, object_pos, target_pos,
            grip_obj_dist, obj_tgt_dist
        ]).astype(np.float32)
    
    def _get_info(self):
        """Additional info dict."""
        gripper_pos = self.data.sensordata[7:10]
        object_pos = self.data.sensordata[10:13]
        target_pos = self.data.sensordata[13:16]
        
        return {
            "gripper_pos": gripper_pos.copy(),
            "object_pos": object_pos.copy(),
            "target_pos": target_pos.copy(),
            "grip_obj_distance": np.linalg.norm(gripper_pos - object_pos),
            "obj_target_distance": np.linalg.norm(object_pos[:2] - target_pos[:2]),
            "grasped": self.grasped,
            "lifted": self.lifted,
            "step": self.current_step,
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Optionally randomize object position
        if options and options.get("randomize", False):
            obj_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")
            # Small random offset
            offset = self.np_random.uniform(-0.03, 0.03, size=2)
            # Note: freejoint objects have 7 DOF in qpos (pos + quat)
            joint_id = self.model.body(obj_body_id).jntadr[0]
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] += offset[0]
            self.data.qpos[qpos_addr + 1] += offset[1]
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        self.initial_obj_pos = self.data.sensordata[10:13].copy()
        self.grasped = False
        self.lifted = False
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one step."""
        self.current_step += 1
        
        # Parse action
        joint_deltas = action[:5] * 0.05  # Scale to reasonable joint velocities
        gripper_cmd = action[5]
        
        # Apply joint position changes
        for i in range(5):
            current = self.data.sensordata[i]
            target = current + joint_deltas[i]
            # Clip to joint limits
            jnt_id = self.model.joint(
                ["shoulder_pan", "shoulder_lift", "elbow", 
                 "wrist_pitch", "wrist_roll"][i]).id
            target = np.clip(target, 
                           self.model.jnt_range[jnt_id, 0],
                           self.model.jnt_range[jnt_id, 1])
            self.data.ctrl[i] = target
        
        # Gripper control
        gripper_width = gripper_cmd * 0.025  # 0=closed, 1=fully open
        self.data.ctrl[5] = gripper_width
        self.data.ctrl[6] = gripper_width
        
        # Step simulation
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Compute reward
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Success condition
        object_pos = self.data.sensordata[10:13]
        target_pos = self.data.sensordata[13:16]
        xy_error = np.linalg.norm(object_pos[:2] - target_pos[:2])
        
        if xy_error < 0.015 and abs(object_pos[2] - target_pos[2]) < 0.05:
            # Object placed near target and on the table
            if not self.grasped:  # Only if we've released it
                reward += 100.0  # Big bonus
                terminated = True
        
        info = self._get_info()
        info["success"] = terminated and xy_error < 0.015
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self):
        """Shaped reward function."""
        gripper_pos = self.data.sensordata[7:10]
        object_pos = self.data.sensordata[10:13]
        target_pos = self.data.sensordata[13:16]
        
        grip_obj_dist = np.linalg.norm(gripper_pos - object_pos)
        obj_tgt_dist = np.linalg.norm(object_pos[:2] - target_pos[:2])
        
        reward = 0.0
        
        # Phase 1: Approach object
        reward -= 0.5 * grip_obj_dist
        
        # Phase 2: Grasp bonus
        if grip_obj_dist < 0.05:
            reward += 0.5
            
            # Check if object lifted
            if object_pos[2] > self.initial_obj_pos[2] + 0.03:
                self.lifted = True
                self.grasped = True
                reward += 2.0
        
        # Phase 3: Transport to target (if lifted)
        if self.lifted:
            reward -= obj_tgt_dist
            
            # Phase 4: Placement accuracy bonus
            if obj_tgt_dist < 0.05:
                reward += 5.0
            if obj_tgt_dist < 0.015:
                reward += 10.0
        
        # Penalty for dropping object
        if self.grasped and object_pos[2] < self.initial_obj_pos[2] - 0.05:
            reward -= 5.0
        
        return reward
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data, camera="front_cam")
            return self._renderer.render().copy()
        elif self.render_mode == "human":
            # For human rendering, use mujoco.viewer externally
            pass
        return None
    
    def close(self):
        """Clean up."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register with gymnasium
def register_env():
    """Register the environment with Gymnasium."""
    try:
        gym.register(
            id="SO101PickPlace-v0",
            entry_point="envs.so101_pick_place_env:SO101PickPlaceEnv",
            max_episode_steps=1000,
        )
        print("Registered SO101PickPlace-v0 with Gymnasium")
    except Exception as e:
        print(f"Could not register environment: {e}")


if __name__ == "__main__":
    # Quick test
    print("Testing SO101 Pick Place Environment...")
    
    env = SO101PickPlaceEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"\nRan {step+1} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final distance to target: {info['obj_target_distance']*1000:.1f}mm")
    
    # Render a frame
    frame = env.render()
    if frame is not None:
        print(f"Rendered frame: {frame.shape}")
    
    env.close()
    print("\nEnvironment test complete!")
