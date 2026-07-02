"""
Run Method A (TargetSelector) on 3 separate combat sequences from 3 different
UC4 gameplay videos. Produces one annotated demo video per sequence and prints
per-sequence lock statistics to the console.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import yaml
import torch

from src.detection.detector import EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.detection.target_selector import TargetSelector, SelectState

# ── Config ──────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL    = "models/training/enemy_detector/weights/best.pt"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES = 300   # 5 s @ 60 fps

SEQUENCES = [
    # (label,  video_file,                                skip_frames,  description)
    ("SEQ-A",
     "dataset/videos/U8tiME2kLok.f399.mp4",
     36000,
     "DevoManiac UC4  @10 min (different segment from our demo_temporal)"),
    ("SEQ-B",
     "dataset/videos/5OYh3vlqTcY.f299.mp4",
     21600,
     "Different UC4 gameplay video  @6 min"),
    ("SEQ-C",
     "dataset/videos/Na7su9ZsqCc.f299.mp4",
     14400,
     "Third UC4 gameplay video  @4 min"),
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_frame(frame, target, fidx, proc_fps):
    out  = frame.copy()
    H, W = out.shape[:2]
    cx, cy = W // 2, H // 2

    cv2.drawMarker(out, (cx, cy), (200, 200, 200),
                   cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

    if target is not None:
        x1, y1, x2, y2 = target.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = f"TARGET  score={target.score:.2f}  conf={target.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(lbl, FONT, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 180, 0), -1)
        cv2.putText(out, lbl, (x1 + 2, y1 - 4), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        ax, ay = int(target.aim_point[0]), int(target.aim_point[1])
        cv2.drawMarker(out, (ax, ay), (0, 255, 255),
                       cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    state = "LOCKED" if target is not None else "SEARCHING"
    col   = (0, 220, 0) if target is not None else (0, 140, 255)
    hud   = f"Frame {fidx:05d}  |  {state}  |  {proc_fps:.1f} FPS"
    cv2.rectangle(out, (0, H - 28), (W, H), (20, 20, 20), -1)
    cv2.putText(out, hud, (8, H - 8), FONT, 0.5, col, 1, cv2.LINE_AA)
    return out


def run_sequence(label, video, skip, description, detector, classifier, obj_filter):
    print(f"\n{'='*60}")
    print(f"{label}  |  {description}")
    print(f"  Video : {Path(video).name}")
    print(f"  Start : frame {skip}  ({skip // 3600}m {(skip // 60) % 60:02d}s)")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 60.0

    out_path = Path("eval_results") / f"demo_{label.lower()}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src, (W, H),
    )

    ts_cfg   = CFG.get("target_selector", {})
    selector = TargetSelector(
        max_lost_frames  = ts_cfg.get("max_lost_frames", 8),
        lock_radius_px   = ts_cfg.get("lock_radius_px", 150),
        iou_match_thresh = ts_cfg.get("iou_match_thresh", 0.25),
    )

    locked_frames  = 0
    n_acquisitions = 0
    prev_state     = SelectState.UNLOCKED
    lock_durations = []
    cur_duration   = 0
    n_dets_total   = 0
    frames_with_enemy = 0
    fi = -1
    t0 = time.perf_counter()

    for fi in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        raw     = detector.detect(frame)
        classed = classifier.classify(frame, raw)
        enemies = obj_filter.filter(classed, W, H)
        target  = selector.update(enemies, W, H)

        n_dets_total += len(enemies)
        if len(enemies) > 0:
            frames_with_enemy += 1

        if selector.state == SelectState.LOCKED:
            locked_frames += 1
            cur_duration  += 1
            if prev_state == SelectState.UNLOCKED:
                n_acquisitions += 1
        else:
            if prev_state == SelectState.LOCKED and cur_duration > 0:
                lock_durations.append(cur_duration)
                cur_duration = 0
        prev_state = selector.state

        elapsed  = time.perf_counter() - t0
        proc_fps = (fi + 1) / max(elapsed, 1e-6)
        writer.write(draw_frame(frame, target, skip + fi, proc_fps))

        if (fi + 1) % 100 == 0:
            print(f"  [{fi+1}/{N_FRAMES}]  state={selector.state.name}  "
                  f"locked={locked_frames}  acq={n_acquisitions}  "
                  f"proc={proc_fps:.1f}fps")

    if cur_duration > 0:
        lock_durations.append(cur_duration)

    cap.release()
    writer.release()

    n_processed = fi + 1
    pct         = 100 * locked_frames // max(n_processed, 1)
    enemy_pct   = 100 * frames_with_enemy // max(n_processed, 1)
    avg_e       = n_dets_total / max(n_processed, 1)
    avg_d       = sum(lock_durations) // max(len(lock_durations), 1)
    max_d       = max(lock_durations) if lock_durations else 0

    print()
    print(f"  Frames processed   : {n_processed}")
    print(f"  Frames with enemy  : {frames_with_enemy} ({enemy_pct}%) — how often any enemy was on screen")
    print(f"  Avg enemies/frame  : {avg_e:.2f}")
    print(f"  LOCKED frames      : {locked_frames}/{n_processed} ({pct}%)")
    print(f"  Lock acquisitions  : {n_acquisitions}")
    if lock_durations:
        print(f"  Lock durations     : {lock_durations}")
        print(f"  Avg lock duration  : {avg_d} frames  ({avg_d / fps_src * 1000:.0f} ms)")
        print(f"  Max lock duration  : {max_d} frames  ({max_d / fps_src * 1000:.0f} ms)")
    else:
        print(f"  Lock durations     : no locks achieved")
    print(f"  Output video       : {out_path}")

    return {
        "label":       label,
        "n_frames":    n_processed,
        "enemy_pct":   enemy_pct,
        "avg_e":       avg_e,
        "locked_pct":  pct,
        "acquisitions": n_acquisitions,
        "avg_dur":     avg_d,
        "max_dur":     max_d,
        "fps_src":     fps_src,
    }


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

    results = []
    for label, video, skip, desc in SEQUENCES:
        r = run_sequence(label, video, skip, desc, detector, classifier, obj_filter)
        if r:
            results.append(r)

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("CROSS-SEQUENCE SUMMARY — Method A (bbox-contains + temporal)")
    print("=" * 70)
    hdr = f"{'Seq':6s}  {'Enemy%':>7s}  {'Det/f':>6s}  {'Lock%':>6s}  {'Acq':>4s}  {'AvgDur':>8s}  {'MaxDur':>8s}"
    print(hdr)
    print("-" * 70)
    for r in results:
        fps = r["fps_src"]
        print(
            f"{r['label']:6s}  "
            f"{r['enemy_pct']:6d}%  "
            f"{r['avg_e']:6.2f}  "
            f"{r['locked_pct']:5d}%  "
            f"{r['acquisitions']:4d}  "
            f"{r['avg_dur']:5d}f/{r['avg_dur']/fps*1000:.0f}ms  "
            f"{r['max_dur']:5d}f/{r['max_dur']/fps*1000:.0f}ms"
        )
    print("=" * 70)
    print()
    print("Demo videos written to:  eval_results/demo_seq-*.mp4")
    print()
    print("How to interpret:")
    print("  Enemy%   — how often at least one enemy is on screen (detection density)")
    print("  Lock%    — fraction of frames where a target was selected")
    print("  Acq      — number of fresh lock-on events (fewer + longer = more stable)")
    print("  AvgDur   — average frames per lock (longer = more stable)")
    print("  MaxDur   — longest single lock (useful for judging coherent engagement)")


if __name__ == "__main__":
    main()
