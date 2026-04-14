"""
Task 1 – Object Pick and Place
Physical AI Hackathon 2026 | LeRobot SO101 | MuJoCo

Autonomous pick-and-place using the real SO101 joint names from the
official URDF (/home/hacker/workspace/src/so101_description/urdf/so101.urdf)

Real SO101 joints (6 DOF):
  shoulder_pan  | -1.91986 to  1.91986 rad
  shoulder_lift | -1.74533 to  1.74533 rad
  elbow_flex    | -1.69    to  1.69    rad
  wrist_flex    | -1.65806 to  1.65806 rad
  wrist_roll    | -2.74385 to  2.84121 rad
  gripper       | -0.17453 to  1.74533 rad  (single revolute jaw)

Run inside Docker:
  python3 pick_place_sim.py --headless --record /output/task1_demo.mp4
"""

import argparse
import os
import time
import numpy as np
import mujoco

SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_scene.xml")

# ── Real SO101 actuator indices (order in XML) ────────────────────────────────
ACT_SHOULDER_PAN  = 0
ACT_SHOULDER_LIFT = 1
ACT_ELBOW_FLEX    = 2
ACT_WRIST_FLEX    = 3
ACT_WRIST_ROLL    = 4
ACT_GRIPPER       = 5

# Real joint names (match URDF exactly)
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex",   "wrist_roll",    "gripper"
]

# Gripper states — single revolute jaw
GRIPPER_OPEN   = 1.4    # rad — jaw open wide
GRIPPER_CLOSED = 0.05   # rad — jaw clamped on object

# Home pose (safe resting configuration)
HOME_POSE = np.array([0.0, -0.5, 1.0, -0.5, 0.0])   # 5 arm joints

# IK / motion parameters
IK_TOL       = 4e-4
IK_MAX_ITER  = 600
IK_STEP      = 0.4
CTRL_FREQ    = 500          # Hz (matches XML timestep 0.002)
SETTLE_STEPS = int(0.6 * CTRL_FREQ)
MOVE_STEPS   = int(1.6 * CTRL_FREQ)


# ─────────────────────────────────────────────────────────────────────────────
class SO101Controller:
    """Position controller for the real SO101 arm."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data  = data

        # Cache site ids
        self.ee_site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.obj_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "object_site")
        self.tgt_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # Cache joint ids for arm (all except gripper)
        self._arm_jnt_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES[:5]   # shoulder_pan … wrist_roll
        ]
        self._gripper_jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )

    # ── Sensor reads ──────────────────────────────────────────────────────────

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_object_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.obj_site_id].copy()

    def get_target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tgt_site_id].copy()

    def get_arm_angles(self) -> np.ndarray:
        return np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self._arm_jnt_ids
        ])

    # ── Commands ──────────────────────────────────────────────────────────────

    def set_arm(self, q: np.ndarray):
        """Send position command to 5 arm joints."""
        for i, val in enumerate(q):
            lo = self.model.actuator_ctrlrange[i, 0]
            hi = self.model.actuator_ctrlrange[i, 1]
            self.data.ctrl[i] = float(np.clip(val, lo, hi))

    def set_gripper(self, angle: float):
        """Set gripper jaw angle (open≈1.4 rad, closed≈0.05 rad)."""
        lo = self.model.actuator_ctrlrange[ACT_GRIPPER, 0]
        hi = self.model.actuator_ctrlrange[ACT_GRIPPER, 1]
        self.data.ctrl[ACT_GRIPPER] = float(np.clip(angle, lo, hi))

    def open_gripper(self):
        self.set_gripper(GRIPPER_OPEN)

    def close_gripper(self):
        self.set_gripper(GRIPPER_CLOSED)

    # ── Numerical IK (Jacobian damped-least-squares) ──────────────────────────

    def ik(self, target_pos: np.ndarray,
           q_init: np.ndarray | None = None) -> np.ndarray:
        """Solve position IK for 5 arm joints → place EE at target_pos."""
        model, data = self.model, self.data

        # Work on a scratch copy
        data_tmp = mujoco.MjData(model)
        mujoco.mj_copyData(data_tmp, model, data)

        q = q_init.copy() if q_init is not None else self.get_arm_angles().copy()

        for _ in range(IK_MAX_ITER):
            # Apply q to scratch
            for k, jid in enumerate(self._arm_jnt_ids):
                data_tmp.qpos[model.jnt_qposadr[jid]] = q[k]
            mujoco.mj_kinematics(model, data_tmp)
            mujoco.mj_comPos(model, data_tmp)

            ee  = data_tmp.site_xpos[self.ee_site_id].copy()
            err = target_pos - ee
            if np.linalg.norm(err) < IK_TOL:
                break

            # Jacobian (3 × nv)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data_tmp, jacp, jacr, self.ee_site_id)

            dof_ids = [model.jnt_dofadr[jid] for jid in self._arm_jnt_ids]
            J       = jacp[:, dof_ids]

            lam = 1e-3
            dq  = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
            q   = q + IK_STEP * dq

            # Clip to joint limits
            for k, jid in enumerate(self._arm_jnt_ids):
                q[k] = np.clip(q[k],
                                model.jnt_range[jid, 0],
                                model.jnt_range[jid, 1])
        return q

    # ── Blocking motions ──────────────────────────────────────────────────────

    def move_to_pos(self, target_pos: np.ndarray,
                    viewer=None, steps: int = MOVE_STEPS):
        """Smooth interpolation to IK solution of target_pos."""
        q_start  = self.get_arm_angles()
        q_target = self.ik(target_pos, q_init=q_start)

        for i in range(steps):
            alpha   = (i + 1) / steps
            alpha_s = alpha * alpha * (3 - 2 * alpha)   # smooth-step
            self.set_arm(q_start + alpha_s * (q_target - q_start))
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()

    def settle(self, viewer=None, steps: int = SETTLE_STEPS):
        """Step without changing commands — let physics settle."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
def pick_and_place(model, data, ctrl: SO101Controller,
                   viewer=None, verbose: bool = True) -> dict:
    """Execute one full pick-and-place episode. Returns result dict."""

    def log(msg):
        if verbose:
            print(f"  [SO101] {msg}")

    HOVER_H = 0.13   # m above table while approaching
    GRASP_H = 0.025  # m above object centre for grasp

    # Phase 0 — Home
    log("Phase 0  Home …")
    ctrl.set_arm(HOME_POSE)
    ctrl.open_gripper()
    ctrl.settle(viewer, steps=SETTLE_STEPS * 2)

    # Phase 1 — Detect
    obj_pos = ctrl.get_object_pos()
    tgt_pos = ctrl.get_target_pos()
    log(f"Phase 1  Object @ {obj_pos.round(3)}")
    log(f"         Target @ {tgt_pos.round(3)}")

    # Phase 2 — Move above object
    log("Phase 2  Approach above object …")
    ctrl.open_gripper()
    ctrl.move_to_pos(obj_pos + [0, 0, HOVER_H], viewer)
    ctrl.settle(viewer)

    # Phase 3 — Lower to grasp height
    log("Phase 3  Lower to grasp …")
    ctrl.move_to_pos(obj_pos + [0, 0, GRASP_H], viewer,
                     steps=int(MOVE_STEPS * 0.6))
    ctrl.settle(viewer, steps=int(SETTLE_STEPS * 0.5))

    # Phase 4 — Close gripper
    log("Phase 4  Close gripper …")
    ctrl.close_gripper()
    ctrl.settle(viewer)

    # Phase 5 — Lift
    log("Phase 5  Lift …")
    ctrl.move_to_pos(obj_pos + [0, 0, HOVER_H + 0.06], viewer)
    ctrl.settle(viewer)

    # Phase 6 — Transport to above target
    log("Phase 6  Transport to target …")
    ctrl.move_to_pos(tgt_pos + [0, 0, HOVER_H], viewer)
    ctrl.settle(viewer)

    # Phase 7 — Lower to place
    log("Phase 7  Lower to place …")
    ctrl.move_to_pos(tgt_pos + [0, 0, GRASP_H], viewer,
                     steps=int(MOVE_STEPS * 0.6))
    ctrl.settle(viewer, steps=int(SETTLE_STEPS * 0.5))

    # Phase 8 — Open gripper (release)
    log("Phase 8  Release …")
    ctrl.open_gripper()
    ctrl.settle(viewer)

    # Phase 9 — Retreat to home
    log("Phase 9  Retreat …")
    ctrl.move_to_pos(tgt_pos + [0, 0, HOVER_H], viewer,
                     steps=int(MOVE_STEPS * 0.5))
    ctrl.set_arm(HOME_POSE)
    ctrl.settle(viewer, steps=SETTLE_STEPS * 2)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    final_obj = ctrl.get_object_pos()
    place_err = float(np.linalg.norm(final_obj[:2] - tgt_pos[:2]))
    height_ok = final_obj[2] > (tgt_pos[2] - 0.04)
    success   = (place_err < 0.05) and height_ok

    log("─" * 48)
    log(f"Result          : {'✓ SUCCESS' if success else '✗ FAILED'}")
    log(f"Placement error : {place_err*100:.1f} cm  (threshold 5 cm)")
    log(f"Object height   : {final_obj[2]:.3f} m")
    log("─" * 48)

    return dict(success=success,
                placement_error_m=place_err,
                object_final_pos=final_obj.tolist(),
                target_pos=tgt_pos.tolist())


# ─────────────────────────────────────────────────────────────────────────────
def reset_episode(model, data, rng, randomise=True):
    mujoco.mj_resetData(model, data)
    if randomise:
        fj_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        adr    = model.jnt_qposadr[fj_id]
        data.qpos[adr + 0] = rng.uniform(0.13, 0.24)   # x
        data.qpos[adr + 1] = rng.uniform(-0.10, 0.10)  # y
        data.qpos[adr + 2] = 0.432                      # z (on table)
        data.qpos[adr + 3] = 1.0                        # qw
        data.qpos[adr + 4:adr + 7] = 0.0
    mujoco.mj_forward(model, data)


# ─────────────────────────────────────────────────────────────────────────────
def make_recorder(path, model, data, width=640, height=480, fps=30):
    try:
        import cv2
    except ImportError:
        print("opencv-python not found — recording disabled.")
        return None, lambda: None

    renderer    = mujoco.Renderer(model, height=height, width=width)
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    writer      = cv2.VideoWriter(path, fourcc, fps, (width, height))
    step_cnt    = [0]
    record_every = max(1, CTRL_FREQ // fps)

    def record():
        step_cnt[0] += 1
        if step_cnt[0] % record_every == 0:
            renderer.update_scene(data)
            frame = renderer.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    renderer._writer = writer
    return renderer, record


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SO101 Pick-and-Place — MuJoCo simulation")
    parser.add_argument("--headless",    action="store_true")
    parser.add_argument("--record",      type=str, default=None,
                        metavar="PATH.mp4")
    parser.add_argument("--num_trials",  type=int, default=1)
    parser.add_argument("--no_random",   action="store_true")
    parser.add_argument("--scene",       type=str, default=SCENE_XML)
    args = parser.parse_args()

    print("=" * 58)
    print("  Physical AI Hackathon 2026  |  Task 1: Pick and Place")
    print("  Robot: LeRobot SO101  |  Simulator: MuJoCo")
    print("=" * 58)
    print(f"\nLoading: {args.scene}")

    model = mujoco.MjModel.from_xml_path(args.scene)
    data  = mujoco.MjData(model)
    ctrl  = SO101Controller(model, data)
    rng   = np.random.default_rng(42)

    _, record_fn = (None, lambda: None)
    if args.record:
        _, record_fn = make_recorder(args.record, model, data)
        print(f"Recording → {args.record}")

    # Patch mj_step globally so recorder fires on every step
    _orig_step = mujoco.mj_step
    def _patched_step(m, d):
        _orig_step(m, d)
        record_fn()
    mujoco.mj_step = _patched_step

    results = []

    def run(viewer):
        for t in range(args.num_trials):
            print(f"\n{'─'*58}")
            print(f"  Trial {t+1}/{args.num_trials}")
            print(f"{'─'*58}")
            reset_episode(model, data, rng, not args.no_random)
            results.append(
                pick_and_place(model, data, ctrl, viewer=viewer)
            )

    if args.headless:
        run(None)
    else:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                run(viewer)
        except Exception as e:
            print(f"Viewer unavailable ({e}), running headless.")
            run(None)

    mujoco.mj_step = _orig_step   # restore

    # Summary
    n_ok = sum(r["success"] for r in results)
    print(f"\n{'='*58}")
    print(f"  SUMMARY  |  {n_ok}/{args.num_trials} success  "
          f"({n_ok/args.num_trials*100:.0f}%)")
    if results:
        avg = np.mean([r["placement_error_m"] for r in results]) * 100
        print(f"  Avg placement error: {avg:.1f} cm")
    print(f"{'='*58}")

    if args.record and hasattr(_, "_writer"):
        _._writer.release()
        print(f"\nVideo saved → {args.record}")


if __name__ == "__main__":
    main()
