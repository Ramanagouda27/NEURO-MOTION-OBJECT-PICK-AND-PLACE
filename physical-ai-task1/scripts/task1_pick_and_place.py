#!/usr/bin/env python3
"""
Physical AI Hackathon 2026 - Task 1: Object Pick and Place
===========================================================
SO101 robot arm simulation in MuJoCo.

Steps:
  1. DETECT  - Locate the object using simulated sensor data
  2. GRASP   - Move to object and close gripper
  3. MOVE    - Transport object to target location
  4. PLACE   - Release object at target

Usage:
  python task1_pick_and_place.py --render          # With GUI
  python task1_pick_and_place.py --record -o out.mp4  # Headless + video
  python task1_pick_and_place.py                   # Headless, no video
"""

import argparse
import os
import sys
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: MuJoCo not installed. Run: pip install mujoco")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────

SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                          "..", "assets", "so101_scene.xml")

# Joint indices in the actuator array
SHOULDER_PAN = 0
SHOULDER_LIFT = 1
ELBOW = 2
WRIST_PITCH = 3
WRIST_ROLL = 4
GRIPPER_LEFT = 5
GRIPPER_RIGHT = 6

# Gripper states
GRIPPER_OPEN = 0.02      # Open position
GRIPPER_CLOSED = 0.001   # Closed position (gripping)

# Control gains
KP = 8.0   # Proportional gain for joint control
KD = 0.5   # Derivative gain

# Tolerance
POSITION_TOLERANCE = 0.015  # 15mm position accuracy
JOINT_TOLERANCE = 0.03      # Joint angle tolerance (rad)


class SO101Controller:
    """Controller for the SO101 robot arm in MuJoCo simulation."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.n_joints = 5  # 5 DOF arm (excluding gripper)
        self.dt = model.opt.timestep
        
    def get_joint_positions(self):
        """Read current joint angles."""
        return np.array([
            self.data.sensordata[i] for i in range(7)
        ])
    
    def get_gripper_pos(self):
        """Get gripper center position in world frame."""
        return self.data.sensordata[7:10].copy()
    
    def get_object_pos(self):
        """Get object position (simulated detection)."""
        return self.data.sensordata[10:13].copy()
    
    def get_target_pos(self):
        """Get target zone position."""
        return self.data.sensordata[13:16].copy()
    
    def set_gripper(self, width):
        """Set gripper opening width."""
        self.data.ctrl[GRIPPER_LEFT] = width
        self.data.ctrl[GRIPPER_RIGHT] = width
    
    def inverse_kinematics_numerical(self, target_pos, max_iter=100, tol=0.005):
        """
        Simple numerical IK using Jacobian transpose method.
        Returns target joint angles to reach the desired end-effector position.
        """
        # Save current state
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        
        # Joint indices for the arm (in qpos)
        joint_ids = [
            self.model.joint(f"shoulder_pan").qposadr[0],
            self.model.joint(f"shoulder_lift").qposadr[0],
            self.model.joint(f"elbow").qposadr[0],
            self.model.joint(f"wrist_pitch").qposadr[0],
            self.model.joint(f"wrist_roll").qposadr[0],
        ]
        
        q = np.array([self.data.qpos[j] for j in joint_ids])
        
        for iteration in range(max_iter):
            # Forward kinematics - get current gripper position
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.get_gripper_pos()
            
            # Compute error
            error = target_pos - current_pos
            dist = np.linalg.norm(error)
            
            if dist < tol:
                break
            
            # Compute Jacobian numerically
            jac = np.zeros((3, len(joint_ids)))
            eps = 1e-4
            
            for i, jid in enumerate(joint_ids):
                # Positive perturbation
                self.data.qpos[jid] += eps
                mujoco.mj_forward(self.model, self.data)
                pos_plus = self.get_gripper_pos()
                
                # Negative perturbation
                self.data.qpos[jid] -= 2 * eps
                mujoco.mj_forward(self.model, self.data)
                pos_minus = self.get_gripper_pos()
                
                # Restore
                self.data.qpos[jid] += eps
                
                jac[:, i] = (pos_plus - pos_minus) / (2 * eps)
            
            # Jacobian transpose step
            alpha = 0.5  # Step size
            dq = alpha * jac.T @ error
            
            # Apply joint limits
            for i, jid in enumerate(joint_ids):
                q[i] += dq[i]
                jnt_idx = 0
                for j_name in ["shoulder_pan", "shoulder_lift", "elbow", 
                                "wrist_pitch", "wrist_roll"]:
                    if self.model.joint(j_name).qposadr[0] == jid:
                        jnt_range = self.model.jnt_range[self.model.joint(j_name).id]
                        q[i] = np.clip(q[i], jnt_range[0], jnt_range[1])
                        break
                self.data.qpos[jid] = q[i]
        
        # Get the final joint configuration
        target_joints = q.copy()
        
        # Restore original state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_forward(self.model, self.data)
        
        return target_joints
    
    def move_to_joint_config(self, target_q, steps=500):
        """
        Smoothly move to target joint configuration using PD control.
        Returns list of control targets for each timestep.
        """
        current_q = np.array([self.data.sensordata[i] for i in range(5)])
        trajectory = []
        
        for step in range(steps):
            # Interpolate: smooth trajectory using cosine interpolation
            t = step / steps
            t_smooth = 0.5 * (1.0 - np.cos(np.pi * t))  # S-curve
            desired_q = current_q + t_smooth * (target_q - current_q)
            trajectory.append(desired_q.copy())
        
        return trajectory


class PickAndPlacePipeline:
    """Complete pick-and-place pipeline for Task 1."""
    
    def __init__(self, scene_xml, render=False, record=False, output_path=None):
        self.render = render
        self.record = record
        self.output_path = output_path
        self.frames = []
        
        # Load MuJoCo model
        if not os.path.exists(scene_xml):
            print(f"ERROR: Scene file not found: {scene_xml}")
            print("Make sure you're running from the correct directory.")
            sys.exit(1)
            
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.controller = SO101Controller(self.model, self.data)
        
        # Initialize renderer for recording
        if self.record:
            self.renderer = mujoco.Renderer(self.model, 480, 640)
        
        # State tracking
        self.phase = "INIT"
        self.success = False
        
    def step_simulation(self, n_steps=1):
        """Advance simulation by n steps."""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            
            if self.record:
                self.renderer.update_scene(self.data, camera="front_cam")
                self.frames.append(self.renderer.render().copy())
    
    def execute_trajectory(self, trajectory, gripper_width=None):
        """Execute a joint trajectory with PD control."""
        for target_q in trajectory:
            # PD control for arm joints
            for i in range(5):
                current = self.data.sensordata[i]
                velocity = self.data.qvel[i] if i < len(self.data.qvel) else 0
                self.data.ctrl[i] = target_q[i]  # Position control via actuator
            
            # Set gripper if specified
            if gripper_width is not None:
                self.controller.set_gripper(gripper_width)
            
            # Step simulation
            self.step_simulation(10)  # 10 substeps per trajectory point
    
    def phase_detect(self):
        """Phase 1: Detect the object position."""
        print("\n[Phase 1] DETECTING OBJECT...")
        
        # Let simulation settle
        self.step_simulation(100)
        
        # Read object position from sensors (simulating camera detection)
        obj_pos = self.controller.get_object_pos()
        target_pos = self.controller.get_target_pos()
        
        print(f"  Object detected at: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
        print(f"  Target zone at:     [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        return obj_pos, target_pos
    
    def phase_approach(self, obj_pos):
        """Phase 2a: Move gripper above the object."""
        print("\n[Phase 2a] APPROACHING OBJECT...")
        
        # Pre-grasp position: above the object
        approach_pos = obj_pos.copy()
        approach_pos[2] += 0.08  # 8cm above object
        
        target_joints = self.controller.inverse_kinematics_numerical(approach_pos)
        trajectory = self.controller.move_to_joint_config(target_joints, steps=300)
        
        # Open gripper during approach
        self.execute_trajectory(trajectory, gripper_width=GRIPPER_OPEN)
        
        gripper_pos = self.controller.get_gripper_pos()
        print(f"  Gripper at: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
    
    def phase_grasp(self, obj_pos):
        """Phase 2b: Lower to object and close gripper."""
        print("\n[Phase 2b] GRASPING OBJECT...")
        
        # Move down to grasp position
        grasp_pos = obj_pos.copy()
        grasp_pos[2] += 0.03  # Slightly above object center
        
        target_joints = self.controller.inverse_kinematics_numerical(grasp_pos)
        trajectory = self.controller.move_to_joint_config(target_joints, steps=200)
        
        # Keep gripper open while lowering
        self.execute_trajectory(trajectory, gripper_width=GRIPPER_OPEN)
        
        # Close gripper
        print("  Closing gripper...")
        for _ in range(200):
            self.controller.set_gripper(GRIPPER_CLOSED)
            self.step_simulation(5)
        
        gripper_pos = self.controller.get_gripper_pos()
        print(f"  Gripper at: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
    
    def phase_lift(self, obj_pos):
        """Phase 2c: Lift the object."""
        print("\n[Phase 2c] LIFTING OBJECT...")
        
        lift_pos = obj_pos.copy()
        lift_pos[2] += 0.12  # Lift 12cm above table
        
        target_joints = self.controller.inverse_kinematics_numerical(lift_pos)
        trajectory = self.controller.move_to_joint_config(target_joints, steps=300)
        self.execute_trajectory(trajectory, gripper_width=GRIPPER_CLOSED)
        
        obj_pos_after = self.controller.get_object_pos()
        print(f"  Object now at: [{obj_pos_after[0]:.3f}, {obj_pos_after[1]:.3f}, {obj_pos_after[2]:.3f}]")
        
        if obj_pos_after[2] > obj_pos[2] + 0.03:
            print("  OBJECT LIFTED SUCCESSFULLY!")
            return True
        else:
            print("  WARNING: Object may not be grasped properly")
            return False
    
    def phase_transport(self, target_pos):
        """Phase 3: Transport object to target zone."""
        print("\n[Phase 3] TRANSPORTING TO TARGET...")
        
        # Move to above target
        transit_pos = target_pos.copy()
        transit_pos[2] += 0.12  # Keep height during transit
        
        target_joints = self.controller.inverse_kinematics_numerical(transit_pos)
        trajectory = self.controller.move_to_joint_config(target_joints, steps=400)
        self.execute_trajectory(trajectory, gripper_width=GRIPPER_CLOSED)
        
        gripper_pos = self.controller.get_gripper_pos()
        print(f"  Gripper at: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
    
    def phase_place(self, target_pos):
        """Phase 4: Lower and release object at target."""
        print("\n[Phase 4] PLACING OBJECT...")
        
        # Lower to placement height
        place_pos = target_pos.copy()
        place_pos[2] += 0.04  # Just above the target surface
        
        target_joints = self.controller.inverse_kinematics_numerical(place_pos)
        trajectory = self.controller.move_to_joint_config(target_joints, steps=250)
        self.execute_trajectory(trajectory, gripper_width=GRIPPER_CLOSED)
        
        # Open gripper to release
        print("  Releasing object...")
        for _ in range(150):
            self.controller.set_gripper(GRIPPER_OPEN)
            self.step_simulation(5)
        
        # Retreat upward
        retreat_pos = place_pos.copy()
        retreat_pos[2] += 0.1
        
        target_joints = self.controller.inverse_kinematics_numerical(retreat_pos)
        trajectory = self.controller.move_to_joint_config(target_joints, steps=200)
        self.execute_trajectory(trajectory, gripper_width=GRIPPER_OPEN)
        
        # Let object settle
        self.step_simulation(300)
    
    def evaluate_result(self, target_pos):
        """Evaluate placement accuracy."""
        print("\n" + "="*50)
        print("EVALUATION")
        print("="*50)
        
        obj_final = self.controller.get_object_pos()
        error = np.linalg.norm(obj_final[:2] - target_pos[:2])  # XY error
        z_error = abs(obj_final[2] - target_pos[2])
        
        print(f"  Object final position:  [{obj_final[0]:.4f}, {obj_final[1]:.4f}, {obj_final[2]:.4f}]")
        print(f"  Target position:        [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        print(f"  XY Placement error:     {error*1000:.1f} mm")
        print(f"  Z error:                {z_error*1000:.1f} mm")
        
        if error < POSITION_TOLERANCE:
            print(f"\n  *** SUCCESS! Object placed within {POSITION_TOLERANCE*1000:.0f}mm tolerance ***")
            self.success = True
        else:
            print(f"\n  Object placed outside {POSITION_TOLERANCE*1000:.0f}mm tolerance.")
            print(f"  Consider tuning IK parameters or trajectory.")
            self.success = False
        
        return error
    
    def run(self):
        """Execute the full pick-and-place pipeline."""
        print("="*60)
        print("  PHYSICAL AI HACKATHON 2026 - TASK 1")
        print("  Object Pick and Place - SO101 Simulation")
        print("="*60)
        
        start_time = time.time()
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Phase 1: Detect
        obj_pos, target_pos = self.phase_detect()
        
        # Phase 2: Grasp (approach + grasp + lift)
        self.phase_approach(obj_pos)
        self.phase_grasp(obj_pos)
        lifted = self.phase_lift(obj_pos)
        
        if lifted:
            # Phase 3: Transport
            self.phase_transport(target_pos)
            
            # Phase 4: Place
            self.phase_place(target_pos)
        
        # Evaluate
        error = self.evaluate_result(target_pos)
        
        elapsed = time.time() - start_time
        print(f"\n  Total execution time: {elapsed:.1f}s")
        print(f"  Simulation time: {self.data.time:.1f}s")
        
        # Save video if recording
        if self.record and self.frames:
            self.save_video()
        
        return self.success
    
    def run_with_viewer(self):
        """Run the pipeline with MuJoCo's built-in viewer."""
        print("="*60)
        print("  PHYSICAL AI HACKATHON 2026 - TASK 1")
        print("  Object Pick and Place - SO101 Simulation (GUI Mode)")
        print("="*60)
        print("\nLaunching MuJoCo viewer...")
        print("The simulation will run automatically. Close the viewer to exit.\n")
        
        mujoco.mj_resetData(self.model, self.data)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Phase 1: Detect
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
            
            obj_pos = self.controller.get_object_pos()
            target_pos = self.controller.get_target_pos()
            print(f"Object at: {obj_pos}, Target at: {target_pos}")
            
            # Phase 2: Approach
            approach_pos = obj_pos.copy()
            approach_pos[2] += 0.08
            target_joints = self.controller.inverse_kinematics_numerical(approach_pos)
            trajectory = self.controller.move_to_joint_config(target_joints, steps=300)
            
            for target_q in trajectory:
                if not viewer.is_running():
                    return
                for i in range(5):
                    self.data.ctrl[i] = target_q[i]
                self.controller.set_gripper(GRIPPER_OPEN)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Lower to grasp
            grasp_pos = obj_pos.copy()
            grasp_pos[2] += 0.03
            target_joints = self.controller.inverse_kinematics_numerical(grasp_pos)
            trajectory = self.controller.move_to_joint_config(target_joints, steps=200)
            
            for target_q in trajectory:
                if not viewer.is_running():
                    return
                for i in range(5):
                    self.data.ctrl[i] = target_q[i]
                self.controller.set_gripper(GRIPPER_OPEN)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Close gripper
            for _ in range(300):
                if not viewer.is_running():
                    return
                self.controller.set_gripper(GRIPPER_CLOSED)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Lift
            lift_pos = obj_pos.copy()
            lift_pos[2] += 0.12
            target_joints = self.controller.inverse_kinematics_numerical(lift_pos)
            trajectory = self.controller.move_to_joint_config(target_joints, steps=300)
            
            for target_q in trajectory:
                if not viewer.is_running():
                    return
                for i in range(5):
                    self.data.ctrl[i] = target_q[i]
                self.controller.set_gripper(GRIPPER_CLOSED)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Transport to target
            transit_pos = target_pos.copy()
            transit_pos[2] += 0.12
            target_joints = self.controller.inverse_kinematics_numerical(transit_pos)
            trajectory = self.controller.move_to_joint_config(target_joints, steps=400)
            
            for target_q in trajectory:
                if not viewer.is_running():
                    return
                for i in range(5):
                    self.data.ctrl[i] = target_q[i]
                self.controller.set_gripper(GRIPPER_CLOSED)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Lower to place
            place_pos = target_pos.copy()
            place_pos[2] += 0.04
            target_joints = self.controller.inverse_kinematics_numerical(place_pos)
            trajectory = self.controller.move_to_joint_config(target_joints, steps=250)
            
            for target_q in trajectory:
                if not viewer.is_running():
                    return
                for i in range(5):
                    self.data.ctrl[i] = target_q[i]
                self.controller.set_gripper(GRIPPER_CLOSED)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Release
            for _ in range(200):
                if not viewer.is_running():
                    return
                self.controller.set_gripper(GRIPPER_OPEN)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Retreat and settle
            retreat_pos = place_pos.copy()
            retreat_pos[2] += 0.1
            target_joints = self.controller.inverse_kinematics_numerical(retreat_pos)
            trajectory = self.controller.move_to_joint_config(target_joints, steps=200)
            
            for target_q in trajectory:
                if not viewer.is_running():
                    return
                for i in range(5):
                    self.data.ctrl[i] = target_q[i]
                self.controller.set_gripper(GRIPPER_OPEN)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Settle
            for _ in range(500):
                if not viewer.is_running():
                    return
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.002)
            
            # Evaluate
            self.evaluate_result(target_pos)
            
            # Keep viewer open
            print("\nSimulation complete. Close the viewer window to exit.")
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)
    
    def save_video(self):
        """Save recorded frames as MP4 video."""
        if not self.frames:
            print("No frames to save.")
            return
        
        output_path = self.output_path or "task1_output.mp4"
        
        try:
            import cv2
            h, w = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))
            
            for frame in self.frames:
                # MuJoCo renders RGB, OpenCV expects BGR
                writer.write(frame[:, :, ::-1])
            
            writer.release()
            print(f"\nVideo saved to: {output_path}")
            print(f"  Frames: {len(self.frames)}, Resolution: {w}x{h}")
            
        except ImportError:
            # Fallback: save as numpy array
            np_path = output_path.replace('.mp4', '.npy')
            np.save(np_path, np.array(self.frames[::10]))  # Save every 10th frame
            print(f"\nFrames saved to: {np_path} (install opencv for MP4)")


def main():
    parser = argparse.ArgumentParser(
        description="Task 1: Object Pick and Place - SO101 Simulation")
    parser.add_argument("--render", action="store_true",
                        help="Launch MuJoCo GUI viewer")
    parser.add_argument("--record", action="store_true",
                        help="Record simulation as video")
    parser.add_argument("-o", "--output", type=str, default="task1_output.mp4",
                        help="Output video file path")
    parser.add_argument("--scene", type=str, default=SCENE_XML,
                        help="Path to MuJoCo scene XML")
    args = parser.parse_args()
    
    pipeline = PickAndPlacePipeline(
        scene_xml=args.scene,
        render=args.render,
        record=args.record,
        output_path=args.output
    )
    
    if args.render:
        pipeline.run_with_viewer()
    else:
        success = pipeline.run()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
