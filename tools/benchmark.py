"""
Performance benchmarking tool.

Measures end-to-end latency of the detection + tracking pipeline on a set
of test images or a video file.  Reports per-stage breakdown so bottlenecks
can be identified and addressed.

Usage:
    python tools/benchmark.py \
        --source dataset/frames/raw \
        --model models/enemy_detector.engine \
        --iterations 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np


def benchmark_pipeline(
    source: str,
    model_path: str,
    iterations: int = 500,
    config_path: str = "config/config.yaml",
) -> None:
    import yaml
    from src.detection.detector import EnemyDetector
    from src.detection.enemy_classifier import EnemyClassifier
    from src.detection.object_filter import ObjectFilter
    from src.tracking.bytetrack_wrapper import ByteTrackWrapper
    from src.utils.profiler import FrameProfiler

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Override model path
    cfg["detection"]["model_path"]    = model_path
    cfg["detection"]["fallback_model"] = model_path

    detector   = EnemyDetector(cfg["detection"])
    classifier = EnemyClassifier(cfg["enemy_classification"])
    obj_filter = ObjectFilter(cfg["object_filter"])
    tracker    = ByteTrackWrapper(cfg["tracking"])

    detector.load()
    detector.warmup(n_iters=20)
    tracker.load()

    frames = _load_frames(source, iterations)
    if not frames:
        print(f"[ERROR] No frames found at {source}")
        return

    profiler = FrameProfiler(log_interval_frames=iterations)
    print(f"Benchmarking {len(frames)} frames × {iterations // max(len(frames),1) + 1} cycles …")

    frame_pool = frames * (iterations // len(frames) + 1)
    frame_pool = frame_pool[:iterations]

    for frame in frame_pool:
        profiler.begin_frame()
        H, W = frame.shape[:2]

        with profiler.section("detection"):
            dets = detector.detect(frame)

        with profiler.section("classification"):
            classified = classifier.classify(frame, dets)

        with profiler.section("filter"):
            enemies = obj_filter.filter(classified, W, H)

        with profiler.section("tracking"):
            tracked = tracker.update(enemies)

        profiler.end_frame()

    print(profiler.report())
    print(f"\nTarget: sub-10 ms total")
    total_avg = profiler.total_avg_ms
    if total_avg <= 10.0:
        print(f"✓  PASS  ({total_avg:.2f} ms avg)")
    else:
        print(f"✗  FAIL  ({total_avg:.2f} ms avg — {total_avg - 10:.2f} ms over budget)")


def _load_frames(source: str, n: int) -> List[np.ndarray]:
    """Load up to *n* frames from a directory or video file."""
    src = Path(source)
    frames = []

    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                img = cv2.imread(str(p))
                if img is not None:
                    frames.append(img)
                if len(frames) >= n:
                    break
    elif src.is_file() and src.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}:
        cap = cv2.VideoCapture(str(src))
        while len(frames) < n:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UC4 Pipeline Benchmark")
    parser.add_argument("--source",     required=True)
    parser.add_argument("--model",      default="models/enemy_detector.engine")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--config",     default="config/config.yaml")
    args = parser.parse_args()

    benchmark_pipeline(
        source     = args.source,
        model_path = args.model,
        iterations = args.iterations,
        config_path= args.config,
    )


if __name__ == "__main__":
    main()
