"""
visual_eval.py — Visual correctness evaluation for Method A.

Shows ALL detected enemies (gray boxes) + the selected target (green box)
so you can judge frame-by-frame whether the selector picked the right enemy.

Segments chosen by quick_scan.py — these are the highest multi-enemy
density windows found across all 6 gameplay videos.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import time
import yaml
import torch

from src.detection.detector import EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.detection.target_selector import TargetSelector, SelectState, TargetInfo

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL  = "models/training/enemy_detector/weights/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Segments from quick_scan.py — sorted by multi-enemy density
SEGMENTS = [
    # (label,  video,                                        skip,   description)
    ("EVAL-1", "dataset/videos/Na7su9ZsqCc.f299.mp4",       33633,  "Na7su9 @9m20s — 3/5 probes had 2 enemies"),
    ("EVAL-2", "dataset/videos/U8tiME2kLok.f399.mp4",       92616,  "U8tiM  @25m43s — 3/5 probes had 2 enemies"),
    ("EVAL-3", "dataset/videos/5OYh3vlqTcY.f299.mp4",       55220,  "5OYh3  @15m20s — 2 enemies in 1 probe"),
    ("EVAL-4", "dataset/videos/Na7su9ZsqCc.f299.mp4",       71814,  "Na7su9 @19m56s — 2 enemies in probe"),
    ("EVAL-5", "dataset/videos/FuIJnd1plI0.f299.mp4",       38400,  "FuIJnd @10m40s — sustained single-enemy (accuracy check)"),
]

N_FRAMES = 150   # 2.5 s — enough to see multiple engagements per segment

FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
GRAY  = (140, 140, 140)
RED   = (0, 60, 255)
CYAN  = (255, 220, 0)
WHITE = (255, 255, 255)


def draw_eval_frame(frame, enemies, target, fidx, proc_fps, label):
    out  = frame.copy()
    H, W = out.shape[:2]
    cx, cy = W // 2, H // 2

    # Screen-centre crosshair (aim reference point)
    cv2.drawMarker(out, (cx, cy), WHITE, cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

    # Draw ALL detected enemies in gray with index labels
    for i, det in enumerate(enemies):
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        det_cx, det_cy = int(det.cx), int(det.cy)

        # Is this the selected target?
        is_target = (
            target is not None and
            abs(det_cx - target.center[0]) < 5 and
            abs(det_cy - target.center[1]) < 5
        )

        if is_target:
            # Green bold box — selected target
            cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 3)
            lbl = f"TARGET  score={target.score:.2f}  conf={det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(lbl, FONT, 0.42, 1)
            cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 160, 0), -1)
            cv2.putText(out, lbl, (x1 + 2, y1 - 4), FONT, 0.42, WHITE, 1, cv2.LINE_AA)
            # Aim point
            ax, ay = int(target.aim_point[0]), int(target.aim_point[1])
            cv2.drawMarker(out, (ax, ay), CYAN, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        else:
            # Gray thin box — other detected enemy (not selected)
            cv2.rectangle(out, (x1, y1), (x2, y2), GRAY, 1)
            cv2.putText(out, f"enemy{i+1}  {det.confidence:.2f}",
                        (x1, y1 - 4), FONT, 0.38, GRAY, 1, cv2.LINE_AA)

    # State badge
    if target is not None:
        state_txt = "LOCKED"
        state_col = GREEN
    else:
        state_txt = "SEARCHING"
        state_col = (0, 140, 255)

    n_others = len(enemies) - (1 if target is not None else 0)
    multi_txt = f"  +{n_others} other(s)" if n_others > 0 else ""

    # Bottom HUD
    hud = f"{label}  |  Frame {fidx:05d}  |  {state_txt}{multi_txt}  |  {proc_fps:.1f}fps"
    cv2.rectangle(out, (0, H - 28), (W, H), (20, 20, 20), -1)
    cv2.putText(out, hud, (8, H - 8), FONT, 0.45, state_col, 1, cv2.LINE_AA)

    # Top-left info: enemy count
    cv2.rectangle(out, (0, 0), (260, 24), (0, 0, 0), -1)
    cv2.putText(out, f"Enemies detected: {len(enemies)}   {'MULTI!' if len(enemies) >= 2 else ''}",
                (6, 17), FONT, 0.45,
                RED if len(enemies) >= 2 else WHITE, 1, cv2.LINE_AA)

    return out


def run_eval_segment(label, video, skip, desc, detector, classifier, obj_filter):
    print(f"\n{'='*60}")
    print(f"{label}  {desc}")
    print(f"  {Path(video).name}  @frame {skip}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 60.0

    out_path = Path("eval_results") / f"visual_{label.lower()}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H)
    )

    ts_cfg   = CFG.get("target_selector", {})
    selector = TargetSelector(
        max_lost_frames  = ts_cfg.get("max_lost_frames", 8),
        lock_radius_px   = ts_cfg.get("lock_radius_px", 150),
        iou_match_thresh = ts_cfg.get("iou_match_thresh", 0.25),
    )

    multi_frames   = 0   # frames where 2+ enemies visible
    locked_correct = 0   # not measurable automatically — for notes
    wrong_candidates = 0  # frames where selector chose amid 2+ enemies
    locked_total   = 0
    t0 = time.perf_counter()
    fi = -1

    for fi in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        raw     = detector.detect(frame)
        classed = classifier.classify(frame, raw)
        enemies = obj_filter.filter(classed, W, H)
        target  = selector.update(enemies, W, H)

        if len(enemies) >= 2:
            multi_frames += 1
            if target is not None:
                wrong_candidates += 1   # selector had to choose among multiple — inspect these

        if target is not None:
            locked_total += 1

        elapsed  = time.perf_counter() - t0
        proc_fps = (fi + 1) / max(elapsed, 1e-6)
        writer.write(draw_eval_frame(frame, enemies, target, skip + fi, proc_fps, label))

    cap.release()
    writer.release()

    n = fi + 1
    print(f"  Frames             : {n}")
    print(f"  Multi-enemy frames : {multi_frames}  ({100*multi_frames//max(n,1)}%)  ← frames where selector had to choose")
    print(f"  LOCKED frames      : {locked_total}  ({100*locked_total//max(n,1)}%)")
    print(f"  Chose amid 2+ det  : {wrong_candidates}  — inspect these in video")
    print(f"  Output             : {out_path}")
    print(f"  >> Watch for: green box on wrong enemy when gray boxes also visible")


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

    for label, video, skip, desc in SEGMENTS:
        run_eval_segment(label, video, skip, desc, detector, classifier, obj_filter)

    print("\n" + "="*60)
    print("All visual evaluation videos written to eval_results/")
    print()
    print("How to review each video:")
    print("  GREEN box  = what the selector chose as the target")
    print("  GRAY box   = other detected enemies the selector did NOT choose")
    print("  CYAN cross = aim point (30% from top of target bbox)")
    print("  White +    = screen centre")
    print()
    print("Key question for each MULTI! frame:")
    print("  Is the GREEN box on the enemy DevoManiac is actually shooting at?")
    print("  Or is it on a bystander / background enemy?")
    print("="*60)


if __name__ == "__main__":
    main()
