"""
Demo Video Generator — uses the PRODUCTION pipeline (ByteTrackWrapper + TargetLock).

Display follows Ahmed's reference images:
  • ONE red bounding box per frame, on the enemy being actively engaged.
  • No box when target cannot be confidently determined.
  • No boxes on teammates, bystanders, or non-engaged enemies.
  • No crosshairs, status text, or any other overlays.

Engagement is inferred from video geometry (r2_held=None → demo mode):
  the box appears only when exactly ONE stable enemy is in the strict
  engagement zone around screen centre.  In the live system the same
  algorithm uses the real R2 trigger signal instead.

Usage:
    python tools/demo_generator.py --model models/training/enemy_detector/weights/best.pt
    python tools/demo_generator.py --model path/to/best.pt --video path/to/gameplay.mp4
    python tools/demo_generator.py --model path/to/best.pt --video gameplay.mp4 --skip 92616
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml

from src.detection.detector import EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.tracking.target_lock import LockState, TargetLock


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "config.yaml"


def _load_cfg(config_path: Path = _DEFAULT_CONFIG) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Rendering — clean red box only, matching Ahmed's reference images
# ------------------------------------------------------------------

def draw_frame(
    frame: np.ndarray,
    locked_box: Optional[Tuple[int, int, int, int]],
    lock_state: LockState,
) -> np.ndarray:
    """
    Render exactly ONE red bounding box on the engaged enemy.
    Reads the pre-clamped locked_box from TargetLock directly — no ByteTrack
    list search — so the box is drawn even during Kalman dropout frames.
    Nothing else is drawn — no gray boxes, no crosshairs, no labels.
    """
    if lock_state == LockState.NO_BOX or locked_box is None:
        return frame

    out = frame.copy()
    cv2.rectangle(out, (locked_box[0], locked_box[1]), (locked_box[2], locked_box[3]), (0, 0, 255), 2)
    return out


# ------------------------------------------------------------------
# Pipeline builders
# ------------------------------------------------------------------

def _build_pipeline(model_path: Path, cfg: dict, device: str):
    det_cfg = dict(cfg.get("detection", {}))
    det_cfg["device"] = device
    # In coco_person mode the model_path CLI arg is irrelevant; let detector.load() handle it.
    # In finetuned mode, force the CLI --model path as the fallback.
    if det_cfg.get("detector_mode", "finetuned") != "coco_person":
        det_cfg["model_path"]    = "models/no_engine.engine"
        det_cfg["fallback_model"] = str(model_path)
    detector = EnemyDetector(det_cfg)
    detector.load()
    detector.warmup(n_iters=5)

    classifier = EnemyClassifier(cfg.get("enemy_classification", {}))
    obj_filter = ObjectFilter(cfg.get("object_filter", {}))

    tracker = ByteTrackWrapper(cfg.get("tracking", {}))
    tracker.load()

    return detector, classifier, obj_filter, tracker


# ------------------------------------------------------------------
# Video processing
# ------------------------------------------------------------------

def run_on_video(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    conf: float = 0.45,
    max_frames: int = 0,
    skip_frames: int = 0,
) -> None:
    cfg = _load_cfg()
    cfg["detection"]["confidence_threshold"] = conf

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    if skip_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
        print(f"Skipping to frame {skip_frames} ({skip_frames / src_fps:.1f}s into video)")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps, (W, H),
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    detector, classifier, obj_filter, tracker = _build_pipeline(model_path, cfg, device)

    lock_sm = TargetLock(
        config=cfg.get("target_lock", {}),
        frame_width=W,
        frame_height=H,
        aim_point_ratio=cfg.get("roi", {}).get("aim_point_ratio", 0.30),
    )

    print(f"Processing: {video_path.name}")
    print(f"Output:     {output_path}")
    print(f"Resolution: {W}x{H}  src_fps={src_fps:.1f}")
    print(f"Device:     {device}")

    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        raw     = detector.detect(frame)
        classed = classifier.classify(frame, raw)
        enemies = obj_filter.filter(classed, W, H)
        tracked = tracker.update(enemies)

        # r2_held=None → demo mode: infer engagement from video geometry
        _, lock_state = lock_sm.update(tracked, l2_held=True, r2_held=None)

        out_frame = draw_frame(frame, lock_sm.locked_box, lock_state)
        writer.write(out_frame)

        if (frame_idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            fps = (frame_idx + 1) / max(elapsed, 1e-6)
            print(f"  [{frame_idx + 1}]  state={lock_state.name}  "
                  f"locked_id={lock_sm.locked_id}  fps={fps:.1f}")

        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    fps = frame_idx / max(elapsed, 1e-6)
    print(f"\nDone — {frame_idx} frames in {elapsed:.1f}s → {output_path}  ({fps:.1f} avg FPS)")


def run_on_frames(
    frames_dir: Path,
    model_path: Path,
    output_path: Path,
    conf: float = 0.45,
    max_frames: int = 900,
    output_fps: float = 30.0,
) -> None:
    cfg = _load_cfg()
    cfg["detection"]["confidence_threshold"] = conf

    exts   = {".jpg", ".jpeg", ".png"}
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise RuntimeError(f"No images found in {frames_dir}")

    sample = cv2.imread(str(frames[0]))
    H, W   = sample.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps, (W, H),
    )

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    detector, classifier, obj_filter, tracker = _build_pipeline(model_path, cfg, device)

    lock_sm = TargetLock(
        config=cfg.get("target_lock", {}),
        frame_width=W,
        frame_height=H,
        aim_point_ratio=cfg.get("roi", {}).get("aim_point_ratio", 0.30),
    )

    print(f"Processing {len(frames)} frames from {frames_dir}")
    print(f"Output: {output_path}  ({W}x{H} @ {output_fps} fps)  device={device}")

    t0 = time.perf_counter()

    for i, img_path in enumerate(frames):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        raw     = detector.detect(frame)
        classed = classifier.classify(frame, raw)
        enemies = obj_filter.filter(classed, W, H)
        tracked = tracker.update(enemies)

        _, lock_state = lock_sm.update(tracked, l2_held=True, r2_held=None)

        out_frame = draw_frame(frame, lock_sm.locked_box, lock_state)
        writer.write(out_frame)

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            fps = (i + 1) / max(elapsed, 1e-6)
            print(f"  [{i + 1}/{len(frames)}]  state={lock_state.name}  fps={fps:.1f}")

    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"\nDone — {len(frames)} frames in {elapsed:.1f}s → {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UC4 Demo Video Generator (production pipeline: ByteTrack + TargetLock)"
    )
    parser.add_argument("--model",      required=True,
                        help="Path to best.pt (trained YOLOv8n)")
    parser.add_argument("--video",      default=None,
                        help="Input gameplay video (.mp4). If omitted, uses --frames dir.")
    parser.add_argument("--frames",     default="dataset/frames/raw",
                        help="Directory of extracted frames (used when --video is not set)")
    parser.add_argument("--output",     default="demo_production.mp4",
                        help="Output video path")
    parser.add_argument("--conf",       type=float, default=0.45,
                        help="YOLO confidence threshold")
    parser.add_argument("--max-frames", type=int,   default=900,
                        help="Max frames to process (0 = all)")
    parser.add_argument("--skip",       type=int,   default=0,
                        help="Skip this many frames from start (jump into gameplay)")
    parser.add_argument("--fps",        type=float, default=30.0,
                        help="Output FPS when using --frames mode")
    args = parser.parse_args()

    if args.video:
        run_on_video(
            video_path  = Path(args.video),
            model_path  = Path(args.model),
            output_path = Path(args.output),
            conf        = args.conf,
            max_frames  = args.max_frames,
            skip_frames = args.skip,
        )
    else:
        run_on_frames(
            frames_dir  = Path(args.frames),
            model_path  = Path(args.model),
            output_path = Path(args.output),
            conf        = args.conf,
            max_frames  = args.max_frames,
            output_fps  = args.fps,
        )


if __name__ == "__main__":
    main()
