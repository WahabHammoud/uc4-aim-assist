"""
Export YOLOv8n to ONNX format for faster CPU inference.

Usage:
    python tools/export_onnx.py [--imgsz 320]

The exported model is saved to models/yolov8n.onnx.
Run inference with: python main.py --capture-card --device-index 0 --show-feed
(the detector auto-detects .onnx extension and uses onnxruntime).
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size (default: 320)")
    args = parser.parse_args()

    from ultralytics import YOLO

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    print(f"Exporting yolov8n.pt to ONNX at {args.imgsz}x{args.imgsz}...")
    model = YOLO("yolov8n.pt")
    export_path = model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    print(f"Exported to: {export_path}")

    import shutil
    dest = out_dir / f"yolov8n_{args.imgsz}.onnx"
    shutil.copy(export_path, dest)
    print(f"Copied to: {dest}")
    print()
    print("To use in config.yaml, set:")
    print("  detection:")
    print(f"    model_path: 'models/yolov8n_{args.imgsz}.onnx'")
    print(f"    input_size: [{args.imgsz}, {args.imgsz}]")
    print("    detector_mode: 'coco_person'")


if __name__ == "__main__":
    main()
