"""
find_combat_segments.py

Scans all gameplay videos at low sample rate (every 60th frame = 1 sample/sec).
For each 10-second window, records average enemy detections per frame.
Reports the top windows ranked by:
  1. Max simultaneous enemies in any single frame (multi-enemy firefights)
  2. Average enemies per sampled frame (sustained combat)

Run this first to find the right skip values before generating demos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml
import torch

from src.detection.detector import EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL  = "models/training/enemy_detector/weights/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_EVERY  = 60    # sample 1 frame per second (60fps source)
WINDOW_FRAMES = 600   # 10-second window = 10 samples at 1/sec
OVERLAP       = 300   # step between windows = 5 seconds (50% overlap)

VIDEOS = [
    "dataset/videos/U8tiME2kLok.f399.mp4",
    "dataset/videos/5OYh3vlqTcY.f299.mp4",
    "dataset/videos/FuIJnd1plI0.f299.mp4",
    "dataset/videos/Na7su9ZsqCc.f299.mp4",
    "dataset/videos/v8CTaY0j_aY.f299.mp4",
    "dataset/videos/v9YTZhLf4zM.f299.mp4",
]

# Skip the first 2 minutes (often menu/cutscene) and last 2 minutes
SKIP_START = 7200
SKIP_END   = 7200


def scan_video(video_path: str, detector, classifier, obj_filter) -> list[dict]:
    name = Path(video_path).name
    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {name}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  {name}  ({total_frames} frames, {total_frames/fps/60:.1f} min)")

    scan_start = SKIP_START
    scan_end   = total_frames - SKIP_END

    windows: list[dict] = []

    # Collect all samples for this video first
    samples: list[tuple[int, int, list]] = []   # (frame_idx, n_enemies, dets)
    sample_idx = scan_start

    cap.set(cv2.CAP_PROP_POS_FRAMES, scan_start)
    cur_pos = scan_start

    while cur_pos < scan_end:
        # Jump to next sample position if needed
        if cur_pos != sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
            cur_pos = sample_idx

        ret, frame = cap.read()
        if not ret:
            break
        cur_pos += 1

        raw     = detector.detect(frame)
        classed = classifier.classify(frame, raw)
        enemies = obj_filter.filter(classed, W, H)

        samples.append((sample_idx, len(enemies), enemies))
        sample_idx += SAMPLE_EVERY

    cap.release()

    # Slide window over samples
    # Each sample covers SAMPLE_EVERY frames; window = WINDOW_FRAMES frames
    samples_per_window = WINDOW_FRAMES // SAMPLE_EVERY   # 10
    step_samples       = OVERLAP // SAMPLE_EVERY          # 5

    for i in range(0, len(samples) - samples_per_window + 1, step_samples):
        window_samples = samples[i : i + samples_per_window]
        counts         = [s[1] for s in window_samples]
        start_frame    = window_samples[0][0]
        max_count      = max(counts)
        avg_count      = sum(counts) / len(counts)
        multi_frames   = sum(1 for c in counts if c >= 2)   # frames with 2+ enemies

        windows.append({
            "video":        video_path,
            "name":         name,
            "start":        start_frame,
            "start_sec":    start_frame / fps,
            "max_enemies":  max_count,
            "avg_enemies":  avg_count,
            "multi_frames": multi_frames,   # how many 1-sec windows had 2+ enemies
            "counts":       counts,
        })

    return windows


def main():
    print(f"Loading detector on {DEVICE} …")
    detector = EnemyDetector({
        "model_path":            "models/no_engine.engine",
        "fallback_model":        MODEL,
        "confidence_threshold":  0.45,
        "device":                DEVICE,
    })
    detector.load()
    detector.warmup(n_iters=2)

    classifier = EnemyClassifier(CFG.get("enemy_classification", {}))
    obj_filter = ObjectFilter(CFG.get("object_filter", {}))

    all_windows: list[dict] = []

    for vid in VIDEOS:
        print(f"\nScanning {Path(vid).name} …")
        windows = scan_video(vid, detector, classifier, obj_filter)
        all_windows.extend(windows)
        print(f"  {len(windows)} windows scanned")

    # Rank by: max_enemies DESC, then multi_frames DESC, then avg DESC
    ranked = sorted(
        all_windows,
        key=lambda w: (w["max_enemies"], w["multi_frames"], w["avg_enemies"]),
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("TOP COMBAT SEGMENTS  (ranked by max simultaneous enemies)")
    print("=" * 80)
    print(f"{'#':>3}  {'Video':32s}  {'Skip':>7}  {'Time':>7}  "
          f"{'MaxEnemy':>9}  {'Avg':>5}  {'Multi10s':>9}  Counts")
    print("-" * 80)

    seen = set()
    shown = 0
    for w in ranked:
        # De-duplicate: skip if we already showed a window within 300 frames of this one from same video
        key = (w["video"], w["start"] // OVERLAP)
        if key in seen:
            continue
        seen.add(key)

        t = w["start_sec"]
        print(
            f"{shown+1:>3}  {w['name']:32s}  {w['start']:>7d}  "
            f"{int(t)//60:>3d}m{int(t)%60:02d}s  "
            f"{w['max_enemies']:>9d}  {w['avg_enemies']:>5.2f}  "
            f"{w['multi_frames']:>9d}  {w['counts']}"
        )
        shown += 1
        if shown >= 20:
            break

    print("=" * 80)
    print()
    print("Use the 'Skip' column values in demo_generator.py --skip to generate demos.")
    print("Prioritise rows where MaxEnemy >= 2 and Multi10s >= 3.")


if __name__ == "__main__":
    main()
