@echo off
:: ============================================================
:: UC4 Aim Assist - GPU Setup (RTX 5060 / onnxruntime-gpu)
:: ============================================================
:: Switches from CPU onnxruntime to the GPU-accelerated build.
:: Run this ONCE on the new PC after running setup_new_pc.bat.
:: ============================================================
cd /d %~dp0

echo [GPU Setup] Uninstalling CPU-only onnxruntime...
pip uninstall onnxruntime -y 2>nul
echo [GPU Setup] Installing onnxruntime-gpu...
pip install onnxruntime-gpu --upgrade

echo.
echo [GPU Setup] Verifying CUDA availability via onnxruntime...
python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('[GPU Setup] Available providers:', providers); cuda_ok = 'CUDAExecutionProvider' in providers; print('[GPU Setup] CUDA ready:', cuda_ok)"

echo.
echo [GPU Setup] Verifying CUDA via PyTorch...
python -c "import torch; print('[GPU Setup] PyTorch CUDA available:', torch.cuda.is_available()); print('[GPU Setup] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo [GPU Setup] Done.
echo   If CUDA is ready: run tools\export_tensorrt.py for maximum performance.
echo   If CUDA shows False: check NVIDIA drivers and CUDA toolkit installation.
echo.
pause
