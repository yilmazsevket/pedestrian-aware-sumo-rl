@echo off
REM Interactive launcher for train_ppo.py
setlocal ENABLEDELAYEDEXPANSION

REM Ensure we're in the script directory
cd /d "%~dp0"

REM Activate venv if present (optional)
if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

python -u train_ppo.py

endlocal
