@echo off
:: ============================================================
::  Task 1 - SO101 Autonomous Pick and Place
::  Choose mode:
::    run_task1.bat          -> visual standalone sim (recommended)
::    run_task1.bat headless -> headless bridge + pick-place script
:: ============================================================

set MODE=%1
if "%MODE%"=="" set MODE=visual

echo.
echo ============================================================
echo   Task 1 - SO101 Autonomous Pick and Place
echo ============================================================
echo.

:: Check container
docker inspect -f "{{.State.Running}}" lerobot_hackathon_env 2>nul | findstr "true" >nul
if errorlevel 1 (
    echo [ERROR] Container is not running. Start it first:
    echo   cd D:\Hackathon\physical-ai-challange-2026\physical-ai-challange-2026\workshop\dev\docker
    echo   docker compose up -d
    echo.
    pause
    exit /b 1
)
echo [OK] Container is running.
echo.

if "%MODE%"=="visual" (
    echo Mode: VISUAL - standalone MuJoCo viewer
    echo Make sure VcXsrv ^(XLaunch^) is running with Disable access control checked.
    echo.
    pause
    start "SO101 Visual Sim" cmd /k docker exec -it -e DISPLAY=host.docker.internal:0.0 lerobot_hackathon_env python3 /home/hacker/workspace/visualize_task1.py
    echo Viewer window should open. Robot will move automatically.
    goto done
)

if "%MODE%"=="headless" (
    echo Mode: HEADLESS - bridge + pick-place ^(no viewer^)
    echo.
    start "MuJoCo Bridge" cmd /k docker exec -it lerobot_hackathon_env bash /home/hacker/workspace/start_bridge.sh
    timeout /t 8 /nobreak >nul
    start "Pick and Place" cmd /k docker exec -it lerobot_hackathon_env bash /home/hacker/workspace/start_pick_place.sh
    echo Bridge and pick-place windows launched.
    goto done
)

:done
echo.
pause
