"""
Export uc4_enemy_v2_final.onnx to TensorRT .engine for RTX 5060.

Usage:
    python tools/export_tensorrt.py

Saves models/uc4_enemy_v2_final.engine (FP16, imgsz=640).
First run takes 3-10 minutes to build the engine. Subsequent runs load instantly.

Requirements:
    - NVIDIA GPU with CUDA (RTX 5060 or similar)
    - TensorRT installed (comes with CUDA toolkit / nvidia-tensorrt)
    - pip install ultralytics
"""

from pathlib import Path


def main():
    onnx_path = Path("models/uc4_enemy_v2_final.onnx")
    engine_path = Path("models/uc4_enemy_v2_final.engine")

    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found at {onnx_path}")
        print("Place uc4_enemy_v2_final.onnx in the models/ directory and retry.")
        return

    print(f"Exporting {onnx_path} → {engine_path}")
    print("Settings: FP16=True, imgsz=640, device=0")
    print("This may take 3-10 minutes on first run...")
    print()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return

    model = YOLO(str(onnx_path), task="detect")
    export_path = model.export(
        format="engine",
        half=True,
        imgsz=640,
        device=0,
        simplify=True,
    )

    import shutil
    shutil.copy(export_path, engine_path)

    print()
    print(f"TensorRT engine saved: {engine_path}")
    print()
    print("The engine will be loaded automatically on next START.bat run.")
    print("No config changes needed — detector.py checks for .engine automatically.")


if __name__ == "__main__":
    main()
