"""
quick_scan.py — Fast probe for multi-enemy combat segments.

Strategy: for each video, probe N evenly-spaced timestamps.
At each timestamp, grab 5 consecutive frames and run detection.
Report timestamps where any frame shows >= 2 enemies simultaneously.

Total cost: 6 videos * 20 probes * 5 frames = 600 detection passes (~2 min on CPU).
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

# 5 frames at each probe point is enough to get a burst sample
FRAMES_PER_PROBE = 5
N_PROBES_PER_VIDEO = 25   # evenly spaced across the video
SKIP_START = 7200          # 2 min: skip menus
SKIP_END   = 7200          # 2 min: skip outro

VIDEOS = [
    "dataset/videos/U8tiME2kLok.f399.mp4",
    "dataset/videos/5OYh3vlqTcY.f299.mp4",
    "dataset/videos/FuIJnd1plI0.f299.mp4",
    "dataset/videos/Na7su9ZsqCc.f299.mp4",
    "dataset/videos/v8CTaY0j_aY.f299.mp4",
    "dataset/videos/v9YTZhLf4zM.f299.mp4",
]


def probe_video(video_path, detector, classifier, obj_filter) -> list[dict]:
    name = Path(video_path).name
    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Cannot open {name}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scannable = total - SKIP_START - SKIP_END
    if scannable <= 0:
        cap.release()
        return []

    step   = scannable // N_PROBES_PER_VIDEO
    probes = [SKIP_START + i * step for i in range(N_PROBES_PER_VIDEO)]

    results = []
    for probe_start in probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, probe_start)
        counts = []
        dets_list = []
        for _ in range(FRAMES_PER_PROBE):
            ret, frame = cap.read()
            if not ret:
                break
            raw     = detector.detect(frame)
            classed = classifier.classify(frame, raw)
            enemies = obj_filter.filter(classed, W, H)
            counts.append(len(enemies))
            dets_list.append(enemies)

        if not counts:
            continue

        max_c = max(counts)
        avg_c = sum(counts) / len(counts)
        t     = probe_start / fps
        results.append({
            "video":       video_path,
            "name":        name,
            "start":       probe_start,
            "time_str":    f"{int(t)//60}m{int(t)%60:02d}s",
            "max_enemies": max_c,
            "avg_enemies": avg_c,
            "counts":      counts,
        })

    cap.release()
    return results


def main():
    print(f"Loading detector on {DEVICE} …\n")
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

    all_results = []
    for vid in VIDEOS:
        print(f"Probing {Path(vid).name} …", end=" ", flush=True)
        r = probe_video(vid, detector, classifier, obj_filter)
        all_results.extend(r)
        hot = [x for x in r if x["max_enemies"] >= 2]
        print(f"{len(r)} probes, {len(hot)} with 2+ enemies")

    # Sort: max_enemies DESC, then avg_enemies DESC
    ranked = sorted(all_results, key=lambda x: (x["max_enemies"], x["avg_enemies"]), reverse=True)

    print("\n" + "=" * 80)
    print("MULTI-ENEMY SEGMENTS  (probe shows >= 1 frame with 2+ enemies simultaneously)")
    print("=" * 80)
    print(f"{'#':>3}  {'Video':32s}  {'Skip':>7}  {'Time':>7}  {'Max':>5}  {'Avg':>5}  Counts")
    print("-" * 80)

    shown = 0
    for r in ranked:
        if r["max_enemies"] < 2:
            break
        print(f"{shown+1:>3}  {r['name']:32s}  {r['start']:>7d}  {r['time_str']:>7s}  "
              f"{r['max_enemies']:>5d}  {r['avg_enemies']:>5.2f}  {r['counts']}")
        shown += 1

    if shown == 0:
        print("  No probe returned 2+ simultaneous enemies.")
        print("  Single-enemy segments (any detection):")
        for r in ranked[:10]:
            if r["max_enemies"] >= 1:
                print(f"     {r['name']}  skip={r['start']}  @{r['time_str']}  "
                      f"max={r['max_enemies']}  counts={r['counts']}")

    print("=" * 80)
    print("\nAll probes with at least 1 enemy detected:")
    print(f"{'Video':32s}  {'Skip':>7}  {'Time':>7}  {'Max':>5}  Counts")
    print("-" * 75)
    for r in sorted(all_results, key=lambda x: x["max_enemies"], reverse=True):
        if r["max_enemies"] >= 1:
            print(f"{r['name']:32s}  {r['start']:>7d}  {r['time_str']:>7s}  "
                  f"{r['max_enemies']:>5d}  {r['counts']}")


if __name__ == "__main__":
    main()
