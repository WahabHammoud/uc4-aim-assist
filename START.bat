@echo off
cd /d %~dp0

REM Normal diagnostic / visualization launch.
REM The captured feed is shown in a windowed OpenCV window with the detection
REM box drawn directly on the frame.  Press ESC inside the window to quit.
REM
REM If you use a UVC capture card, add:  --capture-card --device-index 0
REM If you have a 4K monitor, add:       --windowed
REM For the transparent overlay only:    python main.py --overlay
python main.py --show-feed --windowed
pause
