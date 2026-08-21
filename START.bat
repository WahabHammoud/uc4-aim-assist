@echo off
cd /d %~dp0

REM Kill any running python.exe to prevent leftover feed windows from a
REM previous run.  Errors suppressed — it is fine if nothing is running.
taskkill /f /im python.exe >nul 2>&1

REM Brief pause so Windows releases the cv2 window handle before we open a new one.
timeout /t 1 /nobreak >nul

REM Normal diagnostic / visualization launch.
REM The captured feed is shown in a windowed OpenCV window with the detection
REM box drawn directly on the frame.  Press ESC inside the window to quit.
REM
REM If you use a UVC capture card, add:  --capture-card --device-index 0
REM For the transparent overlay only:    python main.py --overlay
python main.py --show-feed --no-gamepad --windowed
pause
