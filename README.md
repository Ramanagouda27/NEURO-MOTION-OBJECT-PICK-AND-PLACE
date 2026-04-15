# NEURO-MOTION-OBJECT-PICK-AND-PLACE

A comprehensive robotics project for object pick and place tasks using the SO101 robot in simulation and real environments. This repository contains implementations for the Physical AI Hackathon 2026, integrating MuJoCo simulation, LeRobot framework, and ROS 2 for robotic manipulation.

## 📋 Project Overview

This project implements autonomous object picking and placing capabilities utilizing:
- **MuJoCo Physics Simulation**: High-fidelity physics simulation for the SO101 robot
- **LeRobot Framework**: Deep learning-based imitation learning and behavior cloning
- **ROS 2 Integration**: Real-time robot control and sensing
- **Gymnasium Environment**: RL-compatible environment wrapper for the pick-and-place task

## 📁 Repository Structure

```
NEURO-MOTION-OBJECT-PICK-AND-PLACE/
├── physical-ai-task1/              # LeRobot SO101 simulation (MuJoCo)
│   ├── scripts/
│   │   ├── task1_pick_and_place.py    # Main pick-and-place simulation
│   │   ├── task1_full_pipeline.py     # Full pipeline with detection + grasp
│   │   └── test_environment.py        # Environment test script
│   ├── envs/
│   │   └── so101_pick_place_env.py    # Gymnasium environment wrapper
│   ├── assets/
│   │   └── so101_scene.xml            # MuJoCo scene description
│   └── configs/
│       └── task1_config.yaml          # Configuration parameters
│
├── Robothon/                       # Robothon challenge implementation
│   └── task1_pick_place/
│       ├── collect_demos.py        # Demo collection for behavior cloning
│       ├── train_act_policy.py      # ACT policy training script
│       ├── pick_place_sim.py        # Simulation utilities
│       ├── lerobot_env.py           # LeRobot environment integration
│       ├── requirements.txt         # Python dependencies
│       └── so101_scene.xml          # MuJoCo scene
│
└── physical-ai-challange-2026/     # Complete workshop environment (ROS 2)
    └── workshop/
        └── dev/
            └── docker/
                ├── workspace/      # ROS 2 workspace with full stack
                └── tests/          # Integration and unit tests
```

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (with WSL2 + NVIDIA Container Toolkit for GPU passthrough)
- **VcXsrv / XLaunch** for X11 forwarding to Windows

### Three steps to run

**Step 1: Clone the repository**
```bash
git clone https://github.com/Ramanagouda27/NEURO-MOTION-OBJECT-PICK-AND-PLACE.git
cd NEURO-MOTION-OBJECT-PICK-AND-PLACE/physical-ai-challange-2026/workshop/dev/docker/workspace
```

**Step 2: Pull the instructor's Docker image**
```bash
docker pull sahillathwal/physical-ai-challange-2026:latest
```

**Step 3: Start XLaunch**

Launch XLaunch with these settings (then leave it running in the system tray):
- Multiple windows
- Start no client
- ☑ Disable access control
- ☑ Native opengl

**Step 4: Run the simulation**
```powershell
.\run_task1.bat
```

A native MuJoCo window opens showing the robot executing the full pick-and-place sequence autonomously. When finished, `task1_phase_log.csv` is written with per-phase end-effector and object positions.

## 📦 Dependencies

### Core Requirements
- **MuJoCo** >= 3.1.0 - Physics simulation
- **Gymnasium** >= 0.29.1 - RL environment interface
- **PyTorch** >= 2.1.0 - Deep learning framework
- **NumPy** >= 1.24.0 - Numerical computing
- **OpenCV** >= 4.8.0 - Computer vision

### Optional
- **LeRobot** - Native imitation learning support
- **Hugging Face Datasets** - Data management
- **ROS 2** - Robot Operating System integration

See `Robothon/task1_pick_place/requirements.txt` for complete dependency list.

## 🎯 Features

### Pick and Place Task
- Autonomous object detection and localization
- Grasp planning and execution
- Collision avoidance
- Multiple manipulation strategies (front, left, right, rear)

### Learning Capabilities
- Behavior cloning from demonstrations
- ACT (Action Chunking Transformer) policy training
- Diffusion-based trajectory generation
- Transfer learning between simulation and real robot

### Simulation
- Realistic physics with MuJoCo
- Multiple object types and configurations
- Ray-casting based object detection
- Customizable scene environments

### Integration
- ROS 2 compatible interfaces
- Gymnasium environment API
- LeRobot framework compatibility
- Docker containerization

## 🔧 Configuration

Key configuration files:
- `physical-ai-task1/configs/task1_config.yaml` - Main task parameters
- `physical-ai-task1/envs/so101_pick_place_env.py` - Environment settings
- Docker workspace ROS 2 launch files in `workshop/dev/docker/workspace/src/`

Edit configuration files to adjust:
- Object spawn positions and types
- Robot movement speeds and joint limits
- Camera parameters and detection thresholds
- Reward functions and success criteria

## 📝 Usage Examples

### Run Simulation with Rendering
```bash
python physical-ai-task1/scripts/task1_pick_and_place.py --render --episodes 10
```

### Record Video Output
```bash
python physical-ai-task1/scripts/task1_pick_and_place.py --record --output demo.mp4
```

### Full Pipeline with Detection
```bash
python physical-ai-task1/scripts/task1_full_pipeline.py --render
```

### Test Environment Setup
```bash
python physical-ai-task1/scripts/test_environment.py
```

### Collect Demonstrations
```bash
cd Robothon/task1_pick_place
python collect_demos.py --num-demos 100 --output demos/
```

### Train ACT Policy
```bash
cd Robothon/task1_pick_place
python train_act_policy.py --demo-dir demos/ --epochs 50
```

## 🐛 Troubleshooting

### Display/GPU Issues
```bash
# If rendering fails, use headless mode:
export MUJOCO_GL=egl
python scripts/task1_pick_and_place.py --record

# Or use CPU rendering:
export MUJOCO_GL=glx
```

### Docker Memory Issues
- Docker Desktop Settings → Resources → Increase Memory to 8GB+
- Set Swap to 4GB minimum

### ROS 2 Workspace Build Errors
```bash
cd physical-ai-challange-2026/workshop/dev/docker/workspace
rm -rf build install
colcon build --symlink-install
```

### CUDA/GPU Support
To enable GPU in Docker:
```powershell
docker run -it --rm ^
  --gpus all ^
  -e DISPLAY=host.docker.internal:0.0 ^
  -v D:\Hackathon\physical-ai-challange-2026:/workspace ^
  sahillathwal/physical-ai-challange-:latest ^
  /bin/bash
```

## 📖 Project Structure Details

### physical-ai-task1
Main MuJoCo-based simulation environment for quick prototyping:
- **scripts/** - Executable scripts for task execution
- **envs/** - Gymnasium-compatible environment wrapper
- **assets/** - Scene XML and robot descriptions
- **configs/** - YAML configuration files

### Robothon/task1_pick_place
Robothon challenge implementation with LeRobot integration:
- **collect_demos.py** - Collect human demonstrations
- **train_act_policy.py** - Train policy using ACT
- **pick_place_sim.py** - Simulation utilities
- **lerobot_env.py** - LeRobot environment bridge

### physical-ai-challange-2026/workshop
Complete ROS 2 workspace for production deployment:
- **src/so101_gazebo** - Gazebo simulator integration
- **src/so101_moveit_config** - MoveIt motion planning
- **src/so101_mujoco** - MuJoCo ROS 2 bridge
- **src/so101_unified_bringup** - Launch and configuration

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is part of the Physical AI Hackathon 2026.

## 📧 Contact

For issues, questions, or contributions, please open an issue in the GitHub repository.

## 🔗 References

- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [ROS 2 Documentation](https://docs.ros.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**Status**: Active Development | **Last Updated**: 2026-04-14
