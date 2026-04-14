#!/bin/bash
cd /home/hacker/workspace
echo "============================================"
echo "  SO101 Autonomous Pick and Place"
echo "  Waiting 10 seconds for bridge to load..."
echo "============================================"
sleep 10
python3 src/so101_mujoco/scripts/task1_pick_place.py
