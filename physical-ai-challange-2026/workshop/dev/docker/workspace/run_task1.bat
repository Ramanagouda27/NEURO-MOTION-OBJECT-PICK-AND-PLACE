@echo off
REM ============================================================================
REM  SO101 Task 1 - Autonomous Pick & Place
REM  Launches MuJoCo live viewer inside the instructor Docker image via XLaunch.
REM
REM  Prerequisites on the Windows host:
REM    1. Docker Desktop running (WSL2 backend + GPU enabled)
REM    2. VcXsrv / XLaunch running with "Disable access control" checked
REM    3. NVIDIA Container Toolkit installed
REM ============================================================================
setlocal
set WORKSPACE=%~dp0
set IMAGE=sahillathwal/physical-ai-challange-2026:latest
set DISPLAY_ADDR=host.docker.internal:0.0

echo [INFO] Workspace : %WORKSPACE%
echo [INFO] Docker    : %IMAGE%
echo [INFO] DISPLAY   : %DISPLAY_ADDR%
echo.

docker run --rm -it --gpus all ^
  -v "%WORKSPACE%:/home/hacker/workspace" ^
  -e DISPLAY=%DISPLAY_ADDR% ^
  -e MUJOCO_GL=glfw ^
  -e NO_MP4=1 ^
  -e LIBGL_ALWAYS_INDIRECT=0 ^
  -e NVIDIA_DRIVER_CAPABILITIES=all ^
  -e NVIDIA_VISIBLE_DEVICES=all ^
  -w /home/hacker/workspace ^
  %IMAGE% ^
  bash -c "echo '=== GPU ==='; nvidia-smi -L; echo '=== GL ==='; glxinfo 2^>/dev/null ^| grep -E 'OpenGL vendor^|OpenGL renderer'; echo '=== RUN ==='; python3 -u /home/hacker/workspace/visualize_task1.py"

endlocal
