@echo off
:: ============================================================
::  Task 1 – Object Pick and Place
::  Physical AI Hackathon 2026
::  Windows launcher — runs everything inside Docker
:: ============================================================

:: Make sure output folder exists on Windows side
if not exist "output" mkdir output

echo.
echo ============================================================
echo   Physical AI Hackathon 2026 ^| Task 1: Pick and Place
echo   Robot: LeRobot SO101 ^| Simulator: MuJoCo ^| Docker
echo ============================================================
echo.

:: Parse argument
set MODE=%1
if "%MODE%"=="" set MODE=sim

if "%MODE%"=="sim" (
    echo [1/1] Running scripted simulation ^(3 trials^) ...
    echo       Video will be saved to: output\task1_demo.mp4
    echo.
    docker run --rm ^
        -v "%CD%:/task1" ^
        -v "%CD%\output:/output" ^
        -w /task1 ^
        -e MUJOCO_GL=osmesa ^
        -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages ^
        sahillathwal/physical-ai-challange-2026:latest ^
        python3 pick_place_sim.py --headless --num_trials 3 --record /output/task1_demo.mp4
    goto done
)

if "%MODE%"=="collect" (
    echo [1/1] Collecting 50 demonstration episodes ...
    echo.
    docker run --rm ^
        -v "%CD%:/task1" ^
        -w /task1 ^
        -e MUJOCO_GL=osmesa ^
        -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages ^
        sahillathwal/physical-ai-challange-2026:latest ^
        python3 collect_demos.py --num_demos 50 --headless --output_dir /task1/demo_data
    goto done
)

if "%MODE%"=="train" (
    echo [1/1] Training ACT policy on collected demos ...
    echo.
    docker run --rm ^
        -v "%CD%:/task1" ^
        -w /task1 ^
        -e MUJOCO_GL=osmesa ^
        -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages ^
        sahillathwal/physical-ai-challange-2026:latest ^
        python3 train_act_policy.py --mode standalone --data_dir /task1/demo_data --checkpoint_dir /task1/checkpoints --epochs 200
    goto done
)

if "%MODE%"=="eval" (
    echo [1/1] Evaluating trained policy ...
    echo.
    docker run --rm ^
        -v "%CD%:/task1" ^
        -w /task1 ^
        -e MUJOCO_GL=osmesa ^
        -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages ^
        sahillathwal/physical-ai-challange-2026:latest ^
        python3 train_act_policy.py --mode eval --checkpoint /task1/checkpoints/best.pt
    goto done
)

if "%MODE%"=="all" (
    echo Running full pipeline: collect → train → eval
    call %0 collect
    call %0 train
    call %0 eval
    goto done
)

if "%MODE%"=="smoke" (
    echo Running environment smoke-test ...
    docker run --rm ^
        -v "%CD%:/task1" ^
        -w /task1 ^
        -e MUJOCO_GL=osmesa ^
        -e PYTHONPATH=/usr/local/lib/python3.10/dist-packages ^
        sahillathwal/physical-ai-challange-2026:latest ^
        python3 lerobot_env.py
    goto done
)

echo Usage:
echo   run_docker.bat           -- run simulation (default)
echo   run_docker.bat sim       -- run 3-trial simulation, save video
echo   run_docker.bat collect   -- collect 50 demo episodes
echo   run_docker.bat train     -- train ACT policy
echo   run_docker.bat eval      -- evaluate trained policy
echo   run_docker.bat all       -- collect + train + eval
echo   run_docker.bat smoke     -- quick environment smoke-test

:done
echo.
echo Done! Check the output\ folder for results.
