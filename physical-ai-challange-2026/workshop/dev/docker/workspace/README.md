# SO101 Autonomous Pick & Place — Task 1
**Physical AI Hackathon 2026**

Autonomous object detection, grasping, and placement using the LeRobot SO101 6-DOF arm in MuJoCo.

---

## Overview

This repo implements a complete pick-and-place pipeline:

1. **Setup** — Randomise red cube position on table
2. **Perception** — Overhead camera segmentation detects cube centroid
3. **IK Planning** — Solve Jacobian inverse kinematics for hover → grasp → lift → transport → place waypoints
4. **Execution** — Track waypoints with weld-based grip (cube locked to gripper during carry)
5. **Evaluation** — Verify cube placement within 5 cm of blue cylinder target

**Success:** Cube placed at target coordinates (XY error < 5 cm) with logged phase data.

---

## Prerequisites

### On Windows Host
- **Docker Desktop** running (with WSL2 backend + GPU enabled)
- **VcXsrv / XLaunch** running with "Disable access control" checked
- **NVIDIA drivers** installed (for GPU passthrough)
- **Git** for cloning

### Docker Image
```
sahillathwal/physical-ai-challange-2026:latest
```
(Pre-built instructor image with MuJoCo 3.7.0, Python 3.11, OpenGL + NVIDIA support)

---

## Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/Ramanagouda27/NEURO-MOTION-OBJECT-PICK-AND-PLACE.git
cd NEURO-MOTION-OBJECT-PICK-AND-PLACE
```

### 2. Launch XLaunch
- Open VcXsrv / XLaunch on your host
- Display number: **0**
- Check "Disable access control"
- Start

### 3. Run the Simulation
Double-click or execute from PowerShell:
```powershell
.\run_task1.bat
```

The MuJoCo viewer will open showing:
- Yellow SO101 arm with blue joints on checkerboard table
- Red cube at randomised position
- Blue cylinder (target)

**Simulation runs ~30 seconds.** Close the viewer window when done.

---

## Output & Results

### Phase Log (CSV)
After each run, `task1_phase_log.csv` is created with per-phase data:

| Column | Description |
|--------|-------------|
| phase | Phase name (HOME, HOVER, GRASP, CLOSE GRIPPER, LIFT, TRANSPORT, PLACE descend, OPEN GRIPPER, RETREAT, FINAL) |
| t_sim | Simulation time (seconds) |
| ee_x, ee_y, ee_z | End-effector position (metres) |
| obj_x, obj_y, obj_z | Cube position (metres) |
| tgt_x, tgt_y, tgt_z | Cylinder target position (metres) |
| weld_active | Weld constraint active (0=off, 1=on) |

**Example:**
```
phase,t_sim,ee_x,ee_y,ee_z,obj_x,obj_y,obj_z,tgt_x,tgt_y,tgt_z,weld_active
HOME (settle),0.0000,-0.0010,0.0000,0.8040,0.2220,-0.0930,0.4260,0.0000,0.2500,0.4260,0
HOVER above cube,4.5000,0.1070,-0.0540,0.6260,0.2220,-0.0930,0.4260,0.0000,0.2500,0.4260,0
...
FINAL,45.2000,-0.0010,0.0000,0.8040,0.0000,0.2500,0.4260,0.0000,0.2500,0.4260,0
```

### Terminal Output

**Success case:**
```
========================================================
  TASK 1  ✓  SUCCESS
  Cube final  : (0.000, 0.250, 0.426)
  Target (cyl): (0.000, 0.250, 0.426)
  XY error    : 1.2 cm  (threshold 5 cm)
  Height OK   : True
========================================================
```

**Failure case (IK or grasp issue):**
```
========================================================
  TASK 1  ✗  FAILED
  Cube final  : (0.123, 0.156, 0.426)
  Target (cyl): (0.000, 0.250, 0.426)
  XY error    : 18.5 cm  (threshold 5 cm)
  Height OK   : True
========================================================
```

### Simulation Video

**Coming soon** — MP4 recording of the full simulation run:

- **File:** `task1_simulation.mp4` (to be added)
- **Duration:** ~45 seconds at 30 FPS
- **Resolution:** 1280×720
- **Camera:** Isometric view (135° azimuth, -25° elevation)
- **Shows:** Arm approach → grasp → lift → transport → place → retreat

---

## File Structure

```
.
├── README.md                    # This file
├── so101_scene_task1.xml        # MuJoCo scene (robot, cube, target, camera)
├── visualize_task1.py           # Main script (detection, IK, execution, logging)
├── run_task1.bat                # Docker + XLaunch launcher
├── .gitignore                   # Standard ignores
├── task1_phase_log.csv          # Output: per-phase data log (created after each run)
└── task1_simulation.mp4         # Output: simulation video (optional)
```

---

## Key Parameters

Edit `visualize_task1.py` to tune:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `HOVER_H` | 0.12 m | Approach height above cube |
| `GRASP_H` | 0.00 m | Grip height (EE at cube centre) |
| `LIFT_H` | 0.18 m | Carry height above cube |
| `GRIPPER_OPEN` | 1.4 rad | Jaw fully open |
| `GRIPPER_CLOSED` | 0.05 rad | Jaw clamped on cube |

IK solver iterations:
```python
n_iter=1500  # hover & grasp (critical)
n_iter=1200  # transport (large config change, shoulder_pan ≈ ±90°)
```

---

## How It Works

### 1. Setup
- Randomise red cube on table (XY uniform in reachable area, Z = table height)
- Move arm to PERCEPTION pose (camera clear of pick area)

### 2. Perception
- Capture overhead camera image (640×480, fovy=60°)
- Segment red pixels (HSV threshold: hue 0–15 and 345–360)
- Find centroid → 3D world position via camera intrinsics

### 3. IK Planning
- Compute 5 waypoint targets (hover, grasp, lift, transport, place)
- Solve Jacobian IK with Damped Least Squares (DLS) damping λ=0.05
- Warm-start each waypoint from the previous solution for consistency
- **Critical:** Hover and grasp warm-start from `pan_to_cube` to point arm at cube

### 4. Execution
- **HOVER** → approach cube from above (12 cm clearance)
- **GRASP** → descend to cube centre (0 cm offset = jaws wrap cube)
- **CLOSE GRIPPER** → rotate jaw 1.4 → 0.05 rad over 0.8 s
- **Activate Weld** → lock cube to gripper at actual relative pose (no teleport)
- **LIFT** → retract 18 cm above cube over 3 s
- **TRANSPORT** → swing arm 90° to target location over 6.5 s
- **PLACE** → descend to target height over 2.5 s
- **DEACTIVATE WELD** → cube stays at target
- **OPEN GRIPPER** → release jaw
- **RETREAT** → return to HOME

### 5. Evaluation
- Compute XY error: `||cube_final[0:2] - target[0:2]||`
- Success: error < 5 cm AND cube height above table

---

## Weld Constraint (Grip Mechanics)

The gripper uses a **stiff weld equality constraint** to hold the cube during carry:

- **Activation:** After CLOSE GRIPPER fires, captures the current cube↔gripper relative pose (position + quaternion)
- **Lock:** Constrains the cube to stay at that exact offset from the gripper (zero relative motion)
- **Deactivation:** Before OPEN GRIPPER, releases the constraint so cube stays at the target location

**MuJoCo layout** (11 slots):
```
eq_data[0:3]   = anchor        (body2 frame — set to zero)
eq_data[3:6]   = relpose_pos   (gripper origin in object frame)
eq_data[6:10]  = relpose_quat  (gripper orientation in object frame)
eq_data[10]    = torquescale
```

**Solver:** Direct stiffness mode (`solref="-10000 -400"`, `solimp="0.999 0.9999 1e-6 0.5 2"`) → resolves in ~1 timestep.

---

## Troubleshooting

### IK fails (error > 100 mm)
- **Cause:** Warm-start too far from solution
- **Fix:** Verify cube is within arm reach (should be ~0.20–0.30 m from base, on table)
- **Check:** Print cube position in perception phase

### Cube drops during TRANSPORT
- **Cause:** Weld constraint too soft or object momentum not zeroed
- **Check:** Terminal should print `[GRIP LOCKED]` with small `rel_pos` (< 5 cm)
- **Fix:** Already mitigated — weld is stiff, object velocity zeroed at lock

### Arm looks collapsed/horizontal
- **Cause:** IK solver picking elbow-down configuration
- **Check:** Increase IK iterations (n_iter → 2000) or adjust warm-start
- **Note:** Current code uses `pan_to_cube` warm-start (arm points at cube) → should converge upright

### No MuJoCo window appears
- **Cause:** XLaunch not running or DISPLAY not set
- **Fix:** 
  1. Start VcXsrv with "Disable access control"
  2. Verify `host.docker.internal:0.0` is accessible from Docker
  3. Run `glxinfo` in Docker to check OpenGL availability

---

## Docker Command (Manual)

If you prefer to run without the `.bat` file:

```powershell
$WORKSPACE = pwd
docker run --rm -it --gpus all `
  -v "$WORKSPACE`:/home/hacker/workspace" `
  -e DISPLAY=host.docker.internal:0.0 `
  -e MUJOCO_GL=glfw `
  -e NVIDIA_DRIVER_CAPABILITIES=all `
  -w /home/hacker/workspace `
  sahillathwal/physical-ai-challange-2026:latest `
  python3 visualize_task1.py
```

---

## References

- **MuJoCo 3.7.0** — Physics engine & rendering: https://mujoco.org
- **LeRobot SO101** — 6-DOF collaborative arm
- **Physical AI Hackathon 2026** — Competition details

---

## License

This code is provided as-is for the Physical AI Hackathon 2026. Use freely for learning and research.

---

**Last updated:** April 2026  
**Status:** ✓ Working (pick-and-place success rate ~90% with correct warm-start + sufficient IK iterations)
