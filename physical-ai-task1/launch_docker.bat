@echo off
REM ============================================================
REM Physical AI Hackathon 2026 - Task 1 Docker Launcher
REM ============================================================
REM 
REM Prerequisites:
REM   1. Docker Desktop running
REM   2. VcXsrv (XLaunch) running with "Disable access control" checked
REM   3. Docker image: sahillathwal/physical-ai-challange-:latest
REM
REM Usage: Double-click this file or run from PowerShell
REM ============================================================

echo ============================================================
echo  Physical AI Hackathon 2026 - Task 1 Launcher
echo ============================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo Docker is running. Starting container...
echo.
echo IMPORTANT: Make sure VcXsrv (XLaunch) is running with:
echo   - Multiple windows
echo   - Start no client  
echo   - "Disable access control" CHECKED
echo.
pause

REM Launch container with display forwarding and volume mounts
docker run -it --rm ^
  -e DISPLAY=host.docker.internal:0.0 ^
  -e MUJOCO_GL=egl ^
  -v "%~dp0:/task1" ^
  -v "D:\Hackathon\physical-ai-challange-2026:/workspace" ^
  --name physical-ai-task1 ^
  sahillathwal/physical-ai-challange-:latest ^
  /bin/bash -c "echo '=== Container Ready ===' && echo 'Run: cd /task1 && python scripts/test_environment.py' && /bin/bash"

echo.
echo Container exited.
pause
