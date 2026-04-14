# Physical AI Hackathon 2026 - Task 1: Object Pick and Place
## LeRobot SO101 Simulation using MuJoCo

---

## Prerequisites Completed
- [x] Cloned repo to `D:\Hackathon\physical-ai-challange-2026`
- [x] Docker Desktop installed with `sahillathwal/physical-ai-challange-:latest` image
- [x] VcXsrv installed

---

## Step 1: Launch VcXsrv (X Server for GUI Display)

1. Open **XLaunch** (VcXsrv)
2. Select **Multiple windows** → Next
3. Select **Start no client** → Next
4. **CHECK** "Disable access control" → Next → Finish

## Step 2: Run the Docker Container

Open **PowerShell** or **Command Prompt** and run:

```powershell
# Get your IP address first
ipconfig
# Note your IPv4 Address (e.g., 192.168.x.x or 172.x.x.x)

# Set DISPLAY variable and run container
docker run -it --rm ^
  -e DISPLAY=host.docker.internal:0.0 ^
  -v D:\Hackathon\physical-ai-challange-2026:/workspace ^
  -v D:\Hackathon\physical-ai-task1:/task1 ^
  --name physical-ai ^
  sahillathwal/physical-ai-challange-:latest ^
  /bin/bash
```

**Alternative (if above doesn't work):**
```powershell
docker run -it --rm ^
  -e DISPLAY=<YOUR_IP>:0.0 ^
  -v D:\Hackathon\physical-ai-challange-2026:/workspace ^
  -v D:\Hackathon\physical-ai-task1:/task1 ^
  --name physical-ai ^
  sahillathwal/physical-ai-challange-:latest ^
  /bin/bash
```

## Step 3: Inside Docker - Install Dependencies

```bash
# Inside the container
pip install mujoco gymnasium opencv-python-headless numpy

# If lerobot is not already installed:
pip install lerobot

# Test MuJoCo works
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

## Step 4: Copy Task 1 Files

```bash
# Copy the task1 solution files into the workspace
cp -r /task1/* /workspace/task1/
cd /workspace/task1
```

## Step 5: Run the Simulation

```bash
# Option A: Run with MuJoCo GUI viewer (needs VcXsrv)
python scripts/task1_pick_and_place.py --render

# Option B: Run headless (no GUI, saves video)
python scripts/task1_pick_and_place.py --record --output video_output.mp4

# Option C: Run the full pipeline with object detection
python scripts/task1_full_pipeline.py --render
```

## Step 6: Connect with VS Code (Optional)

To use VS Code with the Docker container:

1. Install **"Dev Containers"** extension in VS Code
2. Open VS Code → Press `Ctrl+Shift+P` → "Dev Containers: Attach to Running Container"
3. Select `physical-ai`
4. VS Code will open connected to the container
5. Open `/workspace/task1` folder

---

## Project Structure

```
physical-ai-task1/
├── README.md                    # This file
├── scripts/
│   ├── task1_pick_and_place.py  # Main pick-and-place simulation
│   ├── task1_full_pipeline.py   # Full pipeline with detection + grasp
│   └── test_environment.py      # Quick test to verify setup
├── envs/
│   └── so101_pick_place_env.py  # Gymnasium environment wrapper
├── assets/
│   └── so101_scene.xml          # MuJoCo scene description
└── configs/
    └── task1_config.yaml        # Configuration parameters
```

## Troubleshooting

### VcXsrv / Display Issues
- Make sure "Disable access control" is checked in XLaunch
- Try `host.docker.internal:0.0` as DISPLAY value
- If still failing, run headless mode with `--record` flag

### Docker Memory Issues
- Docker Desktop → Settings → Resources → Set Memory to at least 8GB
- Set Swap to 4GB

### MuJoCo Rendering Issues
- Inside container: `export MUJOCO_GL=egl` for headless rendering
- Or: `export MUJOCO_GL=glx` for X11 forwarding
