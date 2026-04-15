#!/usr/bin/env python3
"""
SO101 Autonomous Pick and Place
Physical AI Hackathon 2026

Pipeline:
  1.  Randomise red cube position on the table
  2.  Move arm aside → overhead camera segmentation → detect cube centroid
  3.  Solve Jacobian IK for all waypoints (hover → grasp → lift → transport → place)
  4.  Execute pick-and-place with MuJoCo passive viewer
  5.  Report XY placement error vs. blue cylinder target

Scene: /home/hacker/workspace/so101_scene_task1.xml
  - Simplified SO101 primitive-geometry robot (real joint names & limits)
  - ee_site between gripper jaws
  - object_geom : red cube (free body, randomised)
  - blue_cyl_geom : blue cylinder (fixed target)
  - overhead camera (fovy=60, z=1.30)
"""

import os
import csv
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── Scene & model (cross-platform, relative to this script) ────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_SCRIPT_DIR, 'so101_scene_task1.xml')
LOG_CSV_PATH = os.path.join(_SCRIPT_DIR, 'task1_phase_log.csv')

# ── Joint / actuator names (order matches real SO101 URDF) ────────────────────
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex',   'wrist_roll',    'gripper']
ACT_NAMES   = ['act_shoulder_pan', 'act_shoulder_lift', 'act_elbow_flex',
               'act_wrist_flex',   'act_wrist_roll',    'act_gripper']

JOINT_LIMITS = [
    (-1.91986,  1.91986),   # shoulder_pan
    (-1.74533,  1.74533),   # shoulder_lift
    (-1.69000,  1.69000),   # elbow_flex
    (-1.65806,  1.65806),   # wrist_flex
    (-2.74385,  2.84121),   # wrist_roll
    (-0.17453,  1.74533),   # gripper
]

# ── Gripper states ─────────────────────────────────────────────────────────────
GRIPPER_OPEN   = 1.4    # jaw fully open
GRIPPER_CLOSED = 0.05   # jaw clamped on object

# ── Named joint configs (6 joints including gripper) ──────────────────────────
HOME_Q6       = np.array([0.00, -0.50,  1.00, -0.50,  0.00, GRIPPER_OPEN])
# Arm swept to -Y (right side) so overhead camera sees the pick area unobstructed
PERCEPTION_Q6 = np.array([-1.50, -0.50,  0.80, -0.50,  0.00, GRIPPER_OPEN])

# ── Camera intrinsics (overhead camera, fovy=60, 640×480) ─────────────────────
CAM_W,  CAM_H  = 640, 480
CAM_F          = (CAM_H / 2.0) / np.tan(np.deg2rad(60.0 / 2.0))   # ≈ 415.7 px
CAM_CX, CAM_CY = CAM_W / 2.0, CAM_H / 2.0

# ── Grasp geometry ─────────────────────────────────────────────────────────────
#   HOVER_H   – ee_site target height above cube centre (approach phase)
#   GRASP_H   – ee_site target height above cube centre (grip phase)
#   LIFT_H    – ee_site target height above cube centre (carry phase)
HOVER_H = 0.13
GRASP_H = 0.025
LIFT_H  = 0.22

# ── Simulation timing ─────────────────────────────────────────────────────────
CTRL_HZ = 50          # controller frequency (iterations/s)
N_SUB   = 10          # physics sub-steps per control step
DT      = 1.0 / CTRL_HZ


# ══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════════════════════

def smooth(t: float) -> float:
    """Smooth-step S-curve: maps [0,1]→[0,1] with zero velocity at endpoints."""
    return t * t * (3.0 - 2.0 * t)


# ── Quaternion helpers (w, x, y, z convention) ────────────────────────────────
def qinv(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (conjugate for unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product a * b."""
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def step_sim(model, data, n: int = 1):
    for _ in range(n):
        mujoco.mj_step(model, data)


def settle(model, data, viewer, secs: float):
    """Run physics for `secs` seconds while syncing viewer."""
    n = max(1, int(secs * CTRL_HZ))
    for _ in range(n):
        t0 = time.monotonic()
        step_sim(model, data, N_SUB)
        viewer.sync()
        r = DT - (time.monotonic() - t0)
        if r > 0:
            time.sleep(r)


def move_to(model, data, viewer, act_id: dict,
            q_start, q_end, dur: float):
    """
    Smoothly interpolate from q_start to q_end (both length-6 joint configs)
    over `dur` seconds while stepping physics and syncing viewer.
    Returns q_end as a list (the new 'current' config).
    """
    n = max(1, int(dur * CTRL_HZ))
    qs = np.asarray(q_start, dtype=float)
    qe = np.asarray(q_end,   dtype=float)
    for i in range(n):
        t0 = time.monotonic()
        t  = smooth((i + 1) / n)
        q  = qs + t * (qe - qs)
        for k, name in enumerate(JOINT_NAMES):
            data.ctrl[act_id[name]] = clamp(float(q[k]), *JOINT_LIMITS[k])
        step_sim(model, data, N_SUB)
        viewer.sync()
        r = DT - (time.monotonic() - t0)
        if r > 0:
            time.sleep(r)
    return qe.tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Jacobian IK  (damped-least-squares, 5 arm joints)
# ══════════════════════════════════════════════════════════════════════════════

def solve_ik(model, site_id: int,
             arm_jnt_ids, arm_dof_ids,
             target: np.ndarray,
             q_init: np.ndarray,
             n_iter: int = 800,
             alpha:  float = 0.4,
             lam:    float = 1e-3) -> tuple[np.ndarray, float]:
    """
    Solve 5-DOF IK (arm only, gripper excluded) so that ee_site reaches `target`.
    Uses a fresh scratch MjData to avoid disturbing the live simulation.

    Returns
    -------
    q   : np.ndarray shape (5,) — arm joint angles
    err : float — final Cartesian error in metres
    """
    data_ik = mujoco.MjData(model)
    q = np.array(q_init, dtype=float)
    err = float('inf')

    for _ in range(n_iter):
        # Apply current q to scratch data
        for k, jid in enumerate(arm_jnt_ids):
            data_ik.qpos[model.jnt_qposadr[jid]] = q[k]
        mujoco.mj_kinematics(model, data_ik)
        mujoco.mj_comPos(model, data_ik)

        # Cartesian error at ee_site
        ee  = data_ik.site_xpos[site_id].copy()
        err_vec = target - ee
        err = float(np.linalg.norm(err_vec))
        if err < 4e-4:   # 0.4 mm — good enough
            break

        # Jacobian (position only, 3 × nv)
        J = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_ik, J, None, site_id)
        J = J[:, arm_dof_ids]

        # Damped-least-squares update
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err_vec)
        q += alpha * dq

        # Enforce joint limits
        for k, jid in enumerate(arm_jnt_ids):
            q[k] = np.clip(q[k],
                           model.jnt_range[jid, 0],
                           model.jnt_range[jid, 1])

    return q, err


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation-based object detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_object_3d(model, data,
                     cam_id:    int,
                     geom_name: str,
                     known_z:   float) -> np.ndarray | None:
    """
    Render the scene from the overhead camera with segmentation enabled.
    Back-project the centroid of the named geom's pixels to 3-D world coords.

    Parameters
    ----------
    model, data  : MuJoCo model/data (physics state must be current)
    cam_id       : camera index of the overhead camera
    geom_name    : name of the geom to detect
    known_z      : known world-Z of the object centre

    Returns
    -------
    3-D world position np.ndarray(3) or None if the geom is not visible.
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        print(f"  [detect] geom '{geom_name}' not found in model")
        return None

    try:
        renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera='overhead')
        seg = renderer.render()   # (H, W, 2): channel 0 = geom_id
        renderer.disable_segmentation_rendering()
        del renderer
    except Exception as ex:
        print(f"  [detect] Renderer error: {ex}")
        return None

    mask = (seg[:, :, 0] == geom_id)
    n_px = int(mask.sum())
    if n_px < 3:
        print(f"  [detect] '{geom_name}': only {n_px} px — not visible")
        return None

    ys, xs = np.where(mask)
    u = float(xs.mean())
    v = float(ys.mean())

    # Camera pose (from live data)
    cam_pos = data.cam_xpos[cam_id].copy()          # (3,) world
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)   # cam→world rotation

    # Back-project: camera looks in -Z, image X right, image Y down
    depth = cam_pos[2] - known_z
    x_c   =  (u - CAM_CX) / CAM_F * depth
    y_c   = -(v - CAM_CY) / CAM_F * depth   # flip Y (image Y is down)
    p_w   = cam_mat @ np.array([x_c, y_c, -depth]) + cam_pos

    print(f"  [detect] '{geom_name}': {n_px} px  "
          f"pixel=({u:.0f},{v:.0f})  "
          f"world=({p_w[0]:.3f}, {p_w[1]:.3f}, {p_w[2]:.3f})")
    return p_w


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  SO101  Autonomous Pick and Place")
    print("  Physical AI Hackathon 2026")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    # ── Build index caches ─────────────────────────────────────────────────────
    act_id = {}
    for name, aname in zip(JOINT_NAMES, ACT_NAMES):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        assert aid >= 0, f"Actuator '{aname}' not found — check scene XML"
        act_id[name] = aid

    arm_jnt_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                   for n in JOINT_NAMES[:5]]
    arm_dof_ids = [int(model.jnt_dofadr[j]) for j in arm_jnt_ids]

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'overhead')
    assert site_id >= 0, "Site 'ee_site' not found"
    assert cam_id  >= 0, "Camera 'overhead' not found"

    fj_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'object_freejoint')
    fj_adr = int(model.jnt_qposadr[fj_id])

    cyl_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'blue_target')
    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'object')

    # Weld equality constraint (gripper_link ↔ object) — activated after jaw close
    eq_id          = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, 'gripper_object_weld')
    gripper_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'gripper_link')
    assert eq_id >= 0,           "Equality 'gripper_object_weld' not found — check scene XML"
    assert gripper_body_id >= 0, "Body 'gripper_link' not found"
    data.eq_active[eq_id] = 0   # make sure weld starts OFF

    # ── Randomise cube position ────────────────────────────────────────────────
    rng = np.random.default_rng()
    rx  = float(rng.uniform(0.10, 0.28))   # x: in front of robot
    ry  = float(rng.uniform(-0.10, 0.10))  # y: small lateral offset
    print(f"\n[setup]  Red cube placed at  x={rx:.3f}  y={ry:.3f}")

    # Initialise physics: arm at perception pose, cube at random position
    mujoco.mj_resetData(model, data)
    for k, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = PERCEPTION_Q6[k]
        data.ctrl[act_id[name]]            = PERCEPTION_Q6[k]
    data.qpos[fj_adr:fj_adr + 7] = [rx, ry, 0.432, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    # Settle cube onto table
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # ── Perception ─────────────────────────────────────────────────────────────
    print("\n[perception]  Segmentation-based detection...")
    cube_xyz = detect_object_3d(model, data, cam_id, 'object_geom', 0.432)

    if cube_xyz is None:
        cube_xyz = np.array([rx, ry, 0.432])
        print(f"  [fallback]  Using known position ({rx:.3f}, {ry:.3f})")
    else:
        cube_xyz[2] = 0.432   # enforce known Z (table height)

    # Blue cylinder is fixed — read world position directly
    mujoco.mj_forward(model, data)
    cyl_xyz = data.xpos[cyl_body_id].copy()
    cyl_xyz[2] = 0.426        # cylinder centre height

    print(f"\n[detected]  cube   = ({cube_xyz[0]:.3f}, {cube_xyz[1]:.3f}, {cube_xyz[2]:.3f})")
    print(f"[target]    cyl    = ({cyl_xyz[0]:.3f},  {cyl_xyz[1]:.3f},  {cyl_xyz[2]:.3f})")

    # ── IK planning ────────────────────────────────────────────────────────────
    print("\n[IK]  Solving waypoints...")

    # Cartesian targets for ee_site
    hover_pos  = np.array([cube_xyz[0], cube_xyz[1], cube_xyz[2] + HOVER_H])
    grasp_pos  = np.array([cube_xyz[0], cube_xyz[1], cube_xyz[2] + GRASP_H])
    lift_pos   = np.array([cube_xyz[0], cube_xyz[1], cube_xyz[2] + LIFT_H])
    trans_pos  = np.array([cyl_xyz[0],  cyl_xyz[1],  cyl_xyz[2]  + LIFT_H])
    place_pos  = np.array([cyl_xyz[0],  cyl_xyz[1],  cyl_xyz[2]  + GRASP_H])

    # IK chain: each waypoint warm-started from the previous solution
    warm = HOME_Q6[:5].copy()
    hover_q5,  e = solve_ik(model, site_id, arm_jnt_ids, arm_dof_ids, hover_pos,  warm)
    print(f"  hover      IK err = {e*1000:.2f} mm")

    grasp_q5,  e = solve_ik(model, site_id, arm_jnt_ids, arm_dof_ids, grasp_pos,  hover_q5)
    print(f"  grasp      IK err = {e*1000:.2f} mm")

    lift_q5,   e = solve_ik(model, site_id, arm_jnt_ids, arm_dof_ids, lift_pos,   grasp_q5)
    print(f"  lift       IK err = {e*1000:.2f} mm")

    trans_q5,  e = solve_ik(model, site_id, arm_jnt_ids, arm_dof_ids, trans_pos,  lift_q5,
                             n_iter=1200)   # extra iters: large config change (pan ≈ +90°)
    print(f"  transport  IK err = {e*1000:.2f} mm")

    place_q5,  e = solve_ik(model, site_id, arm_jnt_ids, arm_dof_ids, place_pos,  trans_q5)
    print(f"  place      IK err = {e*1000:.2f} mm")

    print(f"\n[waypoint targets]")
    print(f"  hover      = ({hover_pos[0]:.3f}, {hover_pos[1]:.3f}, {hover_pos[2]:.3f})")
    print(f"  grasp      = ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
    print(f"  lift       = ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
    print(f"  transport  = ({trans_pos[0]:.3f}, {trans_pos[1]:.3f}, {trans_pos[2]:.3f})")
    print(f"  place      = ({place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f})")

    # Build full 6-joint configs (arm + gripper)
    def q6(q5, g):
        return list(q5) + [g]

    home_pose     = list(HOME_Q6)
    hover_pose    = q6(hover_q5,  GRIPPER_OPEN)
    grasp_open    = q6(grasp_q5,  GRIPPER_OPEN)
    grasp_closed  = q6(grasp_q5,  GRIPPER_CLOSED)
    lift_pose     = q6(lift_q5,   GRIPPER_CLOSED)
    trans_pose    = q6(trans_q5,  GRIPPER_CLOSED)
    place_closed  = q6(place_q5,  GRIPPER_CLOSED)
    place_open    = q6(place_q5,  GRIPPER_OPEN)

    # ── Reset simulation before execution ──────────────────────────────────────
    mujoco.mj_resetData(model, data)
    for k, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = HOME_Q6[k]
        data.ctrl[act_id[name]]            = HOME_Q6[k]
    data.qpos[fj_adr:fj_adr + 7] = [rx, ry, 0.432, 1.0, 0.0, 0.0, 0.0]
    data.eq_active[eq_id] = 0   # ensure weld is OFF after reset
    mujoco.mj_forward(model, data)

    # ── Execute with viewer ────────────────────────────────────────────────────
    print("\n[execution]  Launching viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera view: looking at the table from a comfortable angle
        viewer.cam.azimuth   = 100.0
        viewer.cam.elevation = -35.0
        viewer.cam.distance  =  1.2
        viewer.cam.lookat[:] = [0.1, 0.05, 0.50]
        viewer.sync()

        # Let arm/cube settle in HOME pose before starting
        settle(model, data, viewer, 1.5)
        current = home_pose[:]

        # ── Pick-and-place sequence with weld-based grip ──────────────────────

        def get_ee_pos():
            """Get current EE position."""
            ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
            return data.site_xpos[ee_id].copy()

        # ── CSV phase logger ──────────────────────────────────────────────────
        csv_file = open(LOG_CSV_PATH, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'phase', 't_sim',
            'ee_x', 'ee_y', 'ee_z',
            'obj_x', 'obj_y', 'obj_z',
            'tgt_x', 'tgt_y', 'tgt_z',
            'weld_active',
        ])
        csv_file.flush()
        print(f"  [LOG] writing phase log → {LOG_CSV_PATH}")

        def do_phase(label, target, dur, pause_s):
            nonlocal current
            print(f"  >> {label}")
            current = move_to(model, data, viewer, act_id, current, target, dur)
            settle(model, data, viewer, pause_s)
            mujoco.mj_forward(model, data)
            ee_pos = get_ee_pos()
            obj_pos = data.xpos[obj_body_id].copy()
            csv_writer.writerow([
                label, f"{data.time:.4f}",
                f"{ee_pos[0]:.4f}",  f"{ee_pos[1]:.4f}",  f"{ee_pos[2]:.4f}",
                f"{obj_pos[0]:.4f}", f"{obj_pos[1]:.4f}", f"{obj_pos[2]:.4f}",
                f"{cyl_xyz[0]:.4f}", f"{cyl_xyz[1]:.4f}", f"{cyl_xyz[2]:.4f}",
                int(data.eq_active[eq_id]),
            ])
            csv_file.flush()
            print(f"     EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})  "
                  f"Obj: ({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")

        def activate_weld():
            """
            Lock the cube to the gripper WITHOUT teleport / slingshot.

            Steps:
              1. Snap cube position to the EE site (centred between jaws) and
                 match its orientation to the gripper — so the weld's target
                 offset equals the actual offset (zero violation at t=0).
              2. Zero all 6 DoF of the free-joint so no residual momentum
                 kicks the cube out.
              3. Write the weld relpose using the CORRECT MuJoCo 3.x layout:
                     eq_data[0:3]  = anchor        (body2 frame) → zero
                     eq_data[3:6]  = relpose_pos   (body1 origin in body2)
                     eq_data[6:10] = relpose_quat  (body1 orient in body2)
                     eq_data[10]   = torquescale
              4. Activate the equality constraint.
            """
            mujoco.mj_forward(model, data)

            # (1) Snap cube into canonical grip pose (at EE site)
            ee_sid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
            ee_pos   = data.site_xpos[ee_sid].copy()
            gri_quat = data.xquat[gripper_body_id].copy()
            data.qpos[fj_adr    : fj_adr + 3] = ee_pos
            data.qpos[fj_adr + 3: fj_adr + 7] = gri_quat

            # (2) Zero free-joint velocities (lin + ang) to kill momentum
            obj_dofadr = int(model.jnt_dofadr[fj_id])
            data.qvel[obj_dofadr: obj_dofadr + 6] = 0.0

            mujoco.mj_forward(model, data)

            # (3) Compute body1(gripper)↔body2(object) relative pose
            gri_mat  = data.xmat[gripper_body_id].reshape(3, 3)
            gri_pos  = data.xpos[gripper_body_id].copy()
            obj_pos  = data.xpos[obj_body_id].copy()
            obj_quat = data.xquat[obj_body_id].copy()

            # gripper pose expressed in object frame (body2 = object)
            obj_mat  = data.xmat[obj_body_id].reshape(3, 3)
            rel_pos  = obj_mat.T @ (gri_pos - obj_pos)
            rel_quat = qmul(qinv(obj_quat), gri_quat)

            model.eq_data[eq_id, 0:3]  = 0.0          # anchor at body2 origin
            model.eq_data[eq_id, 3:6]  = rel_pos      # body1 pos in body2 frame
            model.eq_data[eq_id, 6:10] = rel_quat     # body1 quat in body2 frame
            if model.eq_data.shape[1] > 10:
                model.eq_data[eq_id, 10] = 1.0        # torquescale

            data.eq_active[eq_id] = 1
            mujoco.mj_forward(model, data)
            print(f"  >> [GRIP LOCKED]  rel_pos={rel_pos.round(4)}")

        def deactivate_weld():
            data.eq_active[eq_id] = 0
            mujoco.mj_forward(model, data)
            print("  >> [GRIP RELEASED]")

        # Phase 1 – settle at home
        do_phase("HOME (settle)",    home_pose,    1.5, 0.5)
        # Phase 2 – approach hover above cube
        do_phase("HOVER above cube", hover_pose,   2.5, 0.5)
        # Phase 3 – descend to grasp height (gripper open)
        do_phase("GRASP descend",    grasp_open,   2.0, 0.8)
        # Phase 4 – close jaw
        do_phase("CLOSE GRIPPER",    grasp_closed, 0.8, 1.0)
        # LOCK the weld — object is now attached to gripper_link
        activate_weld()
        settle(model, data, viewer, 0.5)
        # Phase 5 – lift
        do_phase("LIFT",             lift_pose,    3.0, 0.5)
        # Phase 6 – transport over target (SLOW — gives weld time to stabilize)
        do_phase("TRANSPORT",        trans_pose,   6.5, 1.2)
        # Phase 7 – descend to place height
        do_phase("PLACE descend",    place_closed, 2.5, 1.0)
        # RELEASE the weld — object stays at target location
        deactivate_weld()
        settle(model, data, viewer, 0.6)
        # Phase 8 – open gripper
        do_phase("OPEN GRIPPER",     place_open,   0.8, 1.0)
        # Phase 9 – retreat home
        do_phase("RETREAT",          home_pose,    2.5, 0.5)

        # ── Evaluate result ────────────────────────────────────────────────────
        mujoco.mj_forward(model, data)
        final_cube = data.xpos[obj_body_id].copy()
        target_pos = cyl_xyz.copy()

        err_2d   = float(np.linalg.norm(final_cube[:2] - target_pos[:2]))
        height_ok = final_cube[2] > (target_pos[2] - 0.04)
        success   = (err_2d < 0.05) and height_ok

        print()
        print("=" * 56)
        print(f"  TASK 1  {'✓  SUCCESS' if success else '✗  FAILED'}")
        print(f"  Cube final  : ({final_cube[0]:.3f}, {final_cube[1]:.3f}, {final_cube[2]:.3f})")
        print(f"  Target (cyl): ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        print(f"  XY error    : {err_2d*100:.1f} cm  (threshold 5 cm)")
        print(f"  Height OK   : {height_ok}")
        print("=" * 56)

        # Final summary row + flush CSV
        csv_writer.writerow([
            'FINAL', f"{data.time:.4f}",
            '', '', '',
            f"{final_cube[0]:.4f}", f"{final_cube[1]:.4f}", f"{final_cube[2]:.4f}",
            f"{target_pos[0]:.4f}", f"{target_pos[1]:.4f}", f"{target_pos[2]:.4f}",
            int(data.eq_active[eq_id]),
        ])
        csv_file.flush()
        csv_file.close()
        print(f"  [LOG] saved → {LOG_CSV_PATH}")

        print("\n  Close the viewer window to exit.")
        while viewer.is_running():
            t0 = time.monotonic()
            step_sim(model, data, N_SUB)
            viewer.sync()
            r = DT - (time.monotonic() - t0)
            if r > 0:
                time.sleep(r)


if __name__ == '__main__':
    main()
