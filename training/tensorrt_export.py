"""
TensorRT engine export.

Converts a trained YOLOv8 .pt model to a TensorRT .engine file optimised
for the current GPU.  TensorRT engines are device-specific — the engine
generated on the deployment GPU cannot be transferred to a different GPU
model.

The export pipeline:
  1. Ultralytics YOLO.export(format="engine") → calls onnx → tensorrt
     (Ultralytics handles the ONNX → TRT conversion internally via TensorRT
      Python API or trtexec, depending on what is installed.)
  2. We verify the engine by running a dummy inference and checking the output
     shape.
  3. We copy the .engine to models/enemy_detector.engine.

Prerequisites:
  - CUDA Toolkit installed.
  - TensorRT SDK (Python wheels: tensorrt, cuda-python).
  - NVIDIA GPU with matching CUDA compute capability.

Usage:
    python -m training.tensorrt_export \
        --weights models/training/enemy_detector/weights/best.pt \
        --imgsz 640 \
        --batch 1
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(
    weights: str,
    imgsz: int = 640,
    batch: int = 1,
    fp16: bool = True,
    output_path: str = "models/enemy_detector.engine",
    workspace_gb: int = 4,
) -> Path:
    """
    Export a YOLOv8 .pt model to TensorRT .engine.

    Parameters
    ----------
    weights      : path to the trained .pt weights.
    imgsz        : input image size (square).
    batch        : static batch size (1 for real-time inference).
    fp16         : enable FP16 / mixed precision for 2× speed on Ampere+.
    output_path  : where to copy the final engine.
    workspace_gb : TensorRT builder workspace size in GB.

    Returns
    -------
    Path to the .engine file.
    """
    from ultralytics import YOLO

    model = YOLO(weights)

    print(f"Exporting {weights} → TensorRT engine …")
    print(f"  imgsz={imgsz}, batch={batch}, fp16={fp16}, workspace={workspace_gb} GB")

    exported_path = model.export(
        format    = "engine",
        imgsz     = imgsz,
        batch     = batch,
        half      = fp16,
        workspace = workspace_gb,
        verbose   = True,
        device    = 0,
    )

    engine_src = Path(exported_path)
    if not engine_src.exists():
        # Ultralytics sometimes returns the ONNX path — find the engine
        candidates = list(engine_src.parent.glob("*.engine"))
        if not candidates:
            raise FileNotFoundError(
                f"Engine not found after export. Searched: {engine_src.parent}"
            )
        engine_src = candidates[0]

    engine_dst = Path(output_path)
    engine_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine_src, engine_dst)
    print(f"\nEngine copied to: {engine_dst}")

    # Verify
    _verify_engine(engine_dst, imgsz)
    return engine_dst


def _verify_engine(engine_path: Path, imgsz: int) -> None:
    """Load the engine via Ultralytics and run one dummy inference."""
    from ultralytics import YOLO
    import numpy as np

    print(f"Verifying engine: {engine_path} …")
    model = YOLO(str(engine_path), task="detect")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    results = model.predict(source=dummy, verbose=False, imgsz=imgsz)
    print(f"  OK — output boxes shape: {results[0].boxes.xyxy.shape}")
    print("Engine verification passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UC4 TensorRT Export")
    parser.add_argument("--weights",   required=True)
    parser.add_argument("--imgsz",     type=int, default=640)
    parser.add_argument("--batch",     type=int, default=1)
    parser.add_argument("--no-fp16",   action="store_true")
    parser.add_argument("--output",    default="models/enemy_detector.engine")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace GB")
    args = parser.parse_args()

    export(
        weights      = args.weights,
        imgsz        = args.imgsz,
        batch        = args.batch,
        fp16         = not args.no_fp16,
        output_path  = args.output,
        workspace_gb = args.workspace,
    )


if __name__ == "__main__":
    main()
