"""
Dataset organisation pipeline.

Takes a flat directory of extracted frames, optionally applies auto-splitting
into train / val / test, and generates the data.yaml that YOLOv8 expects.

Classes
-------
  0 : enemy   — red-marked enemy player

We train on a SINGLE class because the classifier separates enemy from
teammate AFTER detection using colour analysis.  This keeps the detection
model fast (binary) and lets the colour-based classifier handle ambiguous
cases without penalising the model's recall.

Usage:
    python -m training.dataset_pipeline \
        --frames dataset/frames/raw \
        --labels dataset/frames/labels \
        --output dataset/yolo \
        --split 0.80 0.10 0.10
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml

CLASSES = ["enemy"]


def build_yolo_dataset(
    frames_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    split: Tuple[float, float, float] = (0.80, 0.10, 0.10),
    seed: int = 42,
) -> Path:
    """
    Organise annotated frames into YOLOv8 dataset format.

    Parameters
    ----------
    frames_dir : directory with .jpg / .png images.
    labels_dir : directory with corresponding .txt YOLO annotations.
    output_dir : where to create the YOLO dataset tree.
    split      : (train_frac, val_frac, test_frac) — must sum to 1.0.

    Returns
    -------
    Path to the generated data.yaml.
    """
    assert abs(sum(split) - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    # Collect annotated images (must have a matching label file)
    image_paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img in sorted(frames_dir.glob(ext)):
            lbl = labels_dir / (img.stem + ".txt")
            if lbl.exists():
                image_paths.append(img)

    if not image_paths:
        print(f"[WARNING] No annotated frames found in {frames_dir}")
        print(f"          Expected labels at {labels_dir}/<stem>.txt")
        return _write_data_yaml(output_dir)

    random.seed(seed)
    random.shuffle(image_paths)

    n = len(image_paths)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])

    splits = {
        "train": image_paths[:n_train],
        "val":   image_paths[n_train: n_train + n_val],
        "test":  image_paths[n_train + n_val:],
    }

    for split_name, paths in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in paths:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)

        print(f"  {split_name:5s}: {len(paths)} images -> {img_out}")

    data_yaml = _write_data_yaml(output_dir)
    print(f"\nDataset ready.  data.yaml: {data_yaml}")
    print(f"Total images: {n}  (train={n_train}, val={n_val}, test={n-n_train-n_val})")
    return data_yaml


def _write_data_yaml(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UC4 Dataset Pipeline")
    parser.add_argument("--frames",  default="dataset/frames/raw")
    parser.add_argument("--labels",  default="dataset/frames/labels")
    parser.add_argument("--output",  default="dataset/yolo")
    parser.add_argument("--split",   nargs=3, type=float, default=[0.80, 0.10, 0.10])
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    build_yolo_dataset(
        frames_dir = Path(args.frames),
        labels_dir = Path(args.labels),
        output_dir = Path(args.output),
        split      = tuple(args.split),
        seed       = args.seed,
    )


if __name__ == "__main__":
    main()
