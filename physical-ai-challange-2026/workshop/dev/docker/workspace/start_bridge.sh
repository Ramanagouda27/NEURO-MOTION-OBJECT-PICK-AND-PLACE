#!/bin/bash
source /opt/ros/humble/setup.bash
source /home/hacker/workspace/install/setup.bash
cd /home/hacker/workspace
echo "============================================"
echo "  SO101 MuJoCo Bridge - Starting (headless)"
echo "============================================"
python3 src/so101_mujoco/scripts/so101_mujoco_bridge.py \
    --model src/so101_mujoco/mujoco/scene.xml \
    --no-viewer
