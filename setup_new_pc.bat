@echo off
:: ============================================================
:: UC4 Aim Assist - New PC Setup (HP OMEN TG03-0014nx)
:: ============================================================
:: Run this ONCE after cloning the repo on the new machine.
:: Installs all dependencies and sets up GPU acceleration.
:: ============================================================
cd /d %~dp0

echo ============================================================
echo  UC4 Aim Assist - New PC Setup
echo ============================================================
echo.

:: Step 1: Install Python dependencies
echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Make sure Python is in PATH.
    pause
    exit /b 1
)
echo [1/4] Done.
echo.

:: Step 2: GPU setup (swap onnxruntime for onnxruntime-gpu)
echo [2/4] Setting up GPU acceleration...
call setup_gpu.bat
echo [2/4] Done.
echo.

:: Step 3: Verify CUDA
echo [3/4] Final CUDA verification...
python -c "import torch; cuda = torch.cuda.is_available(); name = torch.cuda.get_device_name(0) if cuda else 'N/A'; print(f'  GPU: {name}'); print(f'  CUDA: {cuda}')"
echo [3/4] Done.
echo.

:: Step 4: Next steps
echo [4/4] Setup complete! Next steps:
echo.
echo   A) Export TensorRT engine for maximum performance (do this first):
echo      python tools\export_tensorrt.py
echo      (takes 3-10 min on first run, then loads instantly)
echo.
echo   B) Or skip TensorRT and run directly (uses ONNX + CUDA):
echo      START.bat
echo.
echo   The system will automatically use the best available provider:
echo     .engine file present  ->  TensorRT (fastest)
echo     CUDA available        ->  ONNX + CUDAExecutionProvider
echo     No CUDA               ->  ONNX + CPUExecutionProvider
echo.
pause
