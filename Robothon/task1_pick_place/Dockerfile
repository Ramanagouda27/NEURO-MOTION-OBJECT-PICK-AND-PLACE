# ============================================================
#  Task 1 – Object Pick and Place
#  Physical AI Hackathon 2026
#  Extends the official hackathon Docker image
# ============================================================
FROM sahillathwal/physical-ai-challange-2026:latest

# Set working directory inside container
WORKDIR /task1

# Copy all task1 Python scripts and scene XML
COPY so101_scene.xml    ./
COPY pick_place_sim.py  ./
COPY lerobot_env.py     ./
COPY collect_demos.py   ./
COPY train_act_policy.py ./
COPY run_task1.py       ./

# Install any extra Python packages not already in the image
# (mujoco, numpy, gymnasium, torch, opencv already present in base image)
RUN python3 -m pip install --quiet --no-cache-dir \
        gymnasium \
        opencv-python-headless \
    || true

# Output folder for videos / checkpoints
RUN mkdir -p /output /task1/demo_data /task1/checkpoints

# Set MuJoCo to use software rendering (no GPU required)
ENV MUJOCO_GL=osmesa
ENV DISPLAY=:0
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH

# Default command — run headless simulation and save video
CMD ["python3", "pick_place_sim.py", \
     "--headless", \
     "--num_trials", "3", \
     "--record", "/output/task1_demo.mp4"]
