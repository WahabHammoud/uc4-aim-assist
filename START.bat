@echo off
cd /d %~dp0
python main.py --capture-card --auto-detect --show-feed --no-gamepad
pause
