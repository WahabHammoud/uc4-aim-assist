"""
Automatic annotation pipeline for UC4 enemy frames.

Pipeline per frame:
  1. YOLOv8n (COCO pretrained) detects all "person" bounding boxes.
  2. For each person bbox, sample the region just ABOVE the head where
     Uncharted 4 renders the floating coloured health marker.
  3. HSV colour analysis:
       red pixels  ≥ threshold  → confirmed ENEMY   → write label
       blue pixels ≥ threshold  → teammate / sidekick → skip
       neither                  → motion filter (frame-diff vs previous frame)
                                   moving  → write label (likely enemy)
                                   static  → skip     (statue / painting)
  4. Labels saved in YOLO format: dataset/frames/labels/<stem>.txt

After this script finishes, run a quick spot-check review:
    python tools/annotate_gui.py --frames dataset/frames/raw --labels dataset/frames/labels

Usage:
    python tools/auto_annotator.py
    python tools/auto_annotator.py --frames dataset/frames/raw --labels dataset/frames/labels
    python tools/auto_annotator.py --conf 0.35 --min-red 30 --workers 4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── YOLO person class ID in COCO ──────────────────────────────────────────────
COCO_PERSON_CLASS = 0
ENEMY_CLASS_ID    = 0   # our single class

# ── HSV ranges for red (enemy) marker ─────────────────────────────────────────
RED_LOWER1 = np.array([0,   110, 80],  np.uint8)
RED_UPPER1 = np.array([12,  255, 255], np.uint8)
RED_LOWER2 = np.array([162, 110, 80],  np.uint8)
RED_UPPER2 = np.array([180, 255, 255], np.uint8)

# ── HSV range for blue (teammate) marker ──────────────────────────────────────
BLUE_LOWER = np.array([95,  100, 70],  np.uint8)
BLUE_UPPER = np.array([130, 255, 255], np.uint8)


# ── Per-frame annotation logic ────────────────────────────────────────────────

def _hsv_counts(crop_hsv: np.ndarray) -> Tuple[int, int]:
    """Return (red_pixel_count, blue_pixel_count) in an HSV crop."""
    red_mask  = (cv2.inRange(crop_hsv, RED_LOWER1, RED_UPPER1) |
                 cv2.inRange(crop_hsv, RED_LOWER2, RED_UPPER2))
    blue_mask = cv2.inRange(crop_hsv, BLUE_LOWER, BLUE_UPPER)
    return int(np.count_nonzero(red_mask)), int(np.count_nonzero(blue_mask))


def _marker_crop(
    hsv: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    H: int, W: int,
    above_ratio: float = 0.40,
    width_ratio:  float = 0.55,
) -> Optional[np.ndarray]:
    """Extract the HSV region above the bbox where the floating marker appears."""
    bbox_h   = y2 - y1
    search_h = max(4, int(bbox_h * above_ratio))
    cx       = (x1 + x2) // 2
    hw       = max(8, int((x2 - x1) * width_ratio / 2))

    rx1 = max(0, cx - hw)
    rx2 = min(W - 1, cx + hw)
    ry2 = max(0, y1)
    ry1 = max(0, ry2 - search_h)

    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return hsv[ry1:ry2, rx1:rx2]


def _motion_in_bbox(
    curr_gray: np.ndarray,
    prev_gray: Optional[np.ndarray],
    x1: int, y1: int, x2: int, y2: int,
    min_changed: int = 100,
) -> bool:
    """True if enough pixels changed inside the bbox vs the previous frame."""
    if prev_gray is None:
        return True   # can't check → assume moving

    # Clamp to frame bounds
    fy, fx = curr_gray.shape
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fx, x2), min(fy, y2)
    if x2c <= x1c or y2c <= y1c:
        return False

    diff = cv2.absdiff(curr_gray[y1c:y2c, x1c:x2c],
                       prev_gray[y1c:y2c, x1c:x2c])
    _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    return int(np.count_nonzero(thresh)) >= min_changed


def annotate_frame(
    frame: np.ndarray,
    prev_gray: Optional[np.ndarray],
    model,
    conf_thresh: float,
    min_red: int,
    min_blue: int,
    min_motion: int,
    min_area: int,
    max_area: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Annotate a single frame.

    Returns
    -------
    labels    : list of YOLO label strings  "0 cx cy w h"
    curr_gray : greyscale of this frame (pass as prev_gray to next call)
    """
    H, W = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = model.predict(
        source=frame, conf=conf_thresh, classes=[COCO_PERSON_CLASS],
        verbose=False, imgsz=640,
    )

    labels: List[str] = []

    if not results or results[0].boxes is None:
        return labels, curr_gray

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = [int(v) for v in box]

        # ── geometric sanity ──────────────────────────────────────────────────
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            continue
        aspect = (x2 - x1) / max(y2 - y1, 1)
        if aspect > 2.0 or aspect < 0.10:
            continue

        # ── HUD exclusion (top bar, bottom bar, minimap) ─────────────────────
        cx_n = ((x1 + x2) / 2) / W
        cy_n = ((y1 + y2) / 2) / H
        if cy_n < 0.07 or cy_n > 0.90 or (cx_n < 0.14 and cy_n < 0.22):
            continue

        # ── colour marker check ───────────────────────────────────────────────
        crop = _marker_crop(hsv, x1, y1, x2, y2, H, W)
        if crop is not None and crop.size > 0:
            red_count, blue_count = _hsv_counts(crop)
            if blue_count >= min_blue:
                continue                      # teammate — skip
            if red_count < min_red:
                # No clear red marker → use motion
                if not _motion_in_bbox(curr_gray, prev_gray, x1, y1, x2, y2, min_motion):
                    continue                  # static object — skip
        else:
            # Marker region out of frame — trust motion only
            if not _motion_in_bbox(curr_gray, prev_gray, x1, y1, x2, y2, min_motion):
                continue

        # ── write YOLO label ──────────────────────────────────────────────────
        bx = ((x1 + x2) / 2) / W
        by = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        labels.append(f"{ENEMY_CLASS_ID} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")

    return labels, curr_gray


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(
    frames_dir: Path,
    labels_dir: Path,
    conf_thresh:  float = 0.35,
    min_red:      int   = 30,
    min_blue:     int   = 30,
    min_motion:   int   = 100,
    min_area:     int   = 300,
    max_area:     int   = 280_000,
    overwrite:    bool  = False,
) -> None:
    from ultralytics import YOLO

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Load the COCO-pretrained model (downloads automatically on first run)
    print("Loading YOLOv8n (COCO pretrained)…")
    model = YOLO("yolov8n.pt")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)

    if not frames:
        print(f"[ERROR] No images found in {frames_dir}")
        return

    print(f"Frames to process : {len(frames)}")
    print(f"Labels output     : {labels_dir}")
    print(f"Conf threshold    : {conf_thresh}")
    print(f"Min red pixels    : {min_red}")
    print()

    t0 = time.perf_counter()
    total_labels  = 0
    frames_with   = 0
    frames_skip   = 0
    prev_gray: Optional[np.ndarray] = None

    for i, img_path in enumerate(frames, 1):
        lbl_path = labels_dir / (img_path.stem + ".txt")

        if not overwrite and lbl_path.exists():
            frames_skip += 1
            prev_gray = None   # can't guarantee frame order when skipping
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        labels, prev_gray = annotate_frame(
            frame      = frame,
            prev_gray  = prev_gray,
            model      = model,
            conf_thresh  = conf_thresh,
            min_red      = min_red,
            min_blue     = min_blue,
            min_motion   = min_motion,
            min_area     = min_area,
            max_area     = max_area,
        )

        if labels:
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels) + "\n")
            total_labels += len(labels)
            frames_with  += 1
        else:
            # Write empty file so the dataset pipeline knows this frame was processed
            lbl_path.write_text("")

        if i % 200 == 0 or i == len(frames):
            elapsed = time.perf_counter() - t0
            fps     = i / elapsed
            eta     = (len(frames) - i) / fps if fps > 0 else 0
            print(f"  [{i:>5}/{len(frames)}]  "
                  f"enemies={total_labels}  "
                  f"frames_with_enemy={frames_with}  "
                  f"fps={fps:.1f}  ETA={eta:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"Auto-annotation complete in {elapsed:.1f}s")
    print(f"  Frames processed    : {len(frames) - frames_skip}")
    print(f"  Frames skipped      : {frames_skip} (already labelled)")
    print(f"  Frames with enemies : {frames_with}")
    print(f"  Total enemy boxes   : {total_labels}")
    print(f"  Avg enemies/frame   : {total_labels / max(frames_with, 1):.2f}")
    print(f"\nNext steps:")
    print(f"  1. Quick review (optional but recommended):")
    print(f"     python tools/annotate_gui.py --frames {frames_dir} --labels {labels_dir}")
    print(f"  2. Build dataset splits:")
    print(f"     python -m training.dataset_pipeline --frames {frames_dir} --labels {labels_dir} --output dataset/yolo")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UC4 Auto-Annotator")
    parser.add_argument("--frames",     default="dataset/frames/raw")
    parser.add_argument("--labels",     default="dataset/frames/labels")
    parser.add_argument("--conf",       type=float, default=0.35,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--min-red",    type=int,   default=30,
                        help="Min red pixels to confirm enemy marker")
    parser.add_argument("--min-blue",   type=int,   default=30,
                        help="Min blue pixels to flag as teammate")
    parser.add_argument("--min-motion", type=int,   default=100,
                        help="Min changed pixels to pass motion filter")
    parser.add_argument("--min-area",   type=int,   default=300)
    parser.add_argument("--max-area",   type=int,   default=280_000)
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-annotate frames that already have labels")
    args = parser.parse_args()

    run(
        frames_dir  = Path(args.frames),
        labels_dir  = Path(args.labels),
        conf_thresh = args.conf,
        min_red     = args.min_red,
        min_blue    = args.min_blue,
        min_motion  = args.min_motion,
        min_area    = args.min_area,
        max_area    = args.max_area,
        overwrite   = args.overwrite,
    )


if __name__ == "__main__":
    main()
