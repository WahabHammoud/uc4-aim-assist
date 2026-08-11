@echo off
cd /d %~dp0
python main.py --capture-card --device-index 0 --show-feed --no-gamepad
pause
