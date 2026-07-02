#!/usr/bin/env python3
"""
Engagement algorithm validation script.

Runs the prototype strict-engagement algorithm on existing footage WITHOUT
modifying any production code. Produces:
  - Per-frame CSV with engagement decisions and reasons
  - Side-by-side comparison video (left: all detected enemies; right: new algo)
  - Console metrics summary with focus on multi-enemy sequences

Usage:
    python tools/validate_engagement.py \
        --source   dataset/videos/U8tiME2kLok.f399.mp4 \
        --start    92616 \
        --frames   900 \
        --output   eval_results/engagement_validation \
        --config   config/config.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import Detection, EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─── Algorithm parameters ────────────────────────────────────────────────────
STRICT_RADIUS_PX   = 380    # Circular zone: enemy center must be within this px from screen center
MIN_STABLE_FRAMES  = 8      # Track must be visible for this many frames before engaging
HOLD_FRAMES        = 45     # Frames to keep box after engage conditions no longer met

# ─── Colours ─────────────────────────────────────────────────────────────────
COL_RED   = (0, 0, 220)
COL_GRAY  = (90, 90, 90)
COL_WHITE = (255, 255, 255)
COL_YELLOW= (0, 200, 220)


# ─────────────────────────────────────────────────────────────────────────────
# Prototype EngagementDetector
# ─────────────────────────────────────────────────────────────────────────────
class EngagementDetector:
    """
    Single-algorithm engagement detector.

    Activation gate:
      - Live:  l2_held AND r2_held (caller passes real controller booleans)
      - Demo:  gate is inferred from strict geometry + stability (r2_held=None)

    Everything after the gate (selection, hold, exit) is identical for both modes.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        strict_radius_px: float = STRICT_RADIUS_PX,
        min_stable_frames: int  = MIN_STABLE_FRAMES,
        hold_frames: int        = HOLD_FRAMES,
    ):
        self._cx = frame_width  / 2.0
        self._cy = frame_height / 2.0
        self._radius     = strict_radius_px
        self._min_stable = min_stable_frames
        self._hold       = hold_frames

        self._locked_id: Optional[int] = None
        self._hold_cnt  = 0
        self._track_age: Dict[int, int] = defaultdict(int)

    # ── Main update ─────────────────────────────────────────────────────────

    def update(
        self,
        enemies: List[Detection],
        l2_held: bool = True,
        r2_held: Optional[bool] = None,   # None → infer from video geometry
    ) -> Tuple[Optional[Detection], str]:
        """
        Returns (engaged_detection_or_None, reason_string).
        reason is used for CSV logging and debugging.
        """
        # Update track age counters
        for d in enemies:
            if d.track_id is not None:
                self._track_age[d.track_id] += 1

        # Hard gate: must be in ADS mode
        if not l2_held:
            self._release()
            return None, "L2_OFF"

        # Find candidates in strict zone
        candidates = [e for e in enemies if self._in_zone(e)]

        # Determine engagement gate
        if r2_held is not None:
            # Live mode: use real R2 signal
            gate_open = r2_held
        else:
            # Demo mode: infer from geometry
            # Gate opens only when exactly ONE enemy is in the strict zone AND stable
            if len(candidates) == 1 and self._stable(candidates[0]):
                gate_open = True
            else:
                gate_open = False

        # ── Engagement logic ────────────────────────────────────────────────
        if gate_open and len(candidates) == 1:
            cand = candidates[0]
            if self._stable(cand):
                # Engage (or maintain engagement on same enemy)
                if self._locked_id != cand.track_id:
                    log.debug("Engage: track #%d dist=%.0f age=%d",
                              cand.track_id,
                              self._dist(cand),
                              self._track_age.get(cand.track_id, 0))
                self._locked_id = cand.track_id
                self._hold_cnt  = self._hold
                return cand, "ENGAGED"
            else:
                age = self._track_age.get(cand.track_id, 0)
                return self._try_hold(enemies, f"UNSTABLE(age={age}/{self._min_stable})")

        elif len(candidates) == 0:
            return self._try_hold(enemies, "NO_CANDIDATE")

        elif len(candidates) > 1:
            ids = [str(c.track_id) for c in candidates]
            return self._try_hold(enemies, f"AMBIGUOUS({len(candidates)})")

        else:
            # gate_open is False (R2 not pressed in live mode)
            return self._try_hold(enemies, "R2_OFF")

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _try_hold(
        self,
        enemies: List[Detection],
        reason: str,
    ) -> Tuple[Optional[Detection], str]:
        """Maintain hold box if we're counting down after losing engagement."""
        if self._hold_cnt > 0 and self._locked_id is not None:
            locked = next((e for e in enemies if e.track_id == self._locked_id), None)
            if locked is not None:
                self._hold_cnt -= 1
                return locked, f"HOLDING({self._hold_cnt}):{reason}"
        self._release()
        return None, f"NO_BOX:{reason}"

    def _in_zone(self, det: Detection) -> bool:
        return self._dist(det) <= self._radius

    def _dist(self, det: Detection) -> float:
        return math.hypot(det.cx - self._cx, det.cy - self._cy)

    def _stable(self, det: Detection) -> bool:
        return self._track_age.get(det.track_id, 0) >= self._min_stable

    def _release(self) -> None:
        self._locked_id = None
        self._hold_cnt  = 0


# ─────────────────────────────────────────────────────────────────────────────
# Frame rendering
# ─────────────────────────────────────────────────────────────────────────────
def draw_left(frame: np.ndarray, enemies: List[Detection]) -> np.ndarray:
    """Left panel: current behavior — all detected enemies, gray boxes."""
    out = frame.copy()
    for d in enemies:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_GRAY, 2)
        if d.track_id is not None:
            cv2.putText(out, f"#{d.track_id}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GRAY, 1)
    cv2.putText(out, "CURRENT (all boxes)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_WHITE, 2)
    return out


def draw_right(
    frame: np.ndarray,
    enemies: List[Detection],
    engaged: Optional[Detection],
    reason: str,
    strict_radius_px: float,
) -> np.ndarray:
    """Right panel: new behavior — one red box or nothing."""
    out = frame.copy()
    cx, cy = out.shape[1] / 2, out.shape[0] / 2

    # Draw strict zone circle (faint)
    cv2.circle(out, (int(cx), int(cy)), int(strict_radius_px),
               (40, 40, 40), 1)

    if engaged is not None:
        x1, y1, x2, y2 = int(engaged.x1), int(engaged.y1), int(engaged.x2), int(engaged.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_RED, 3)
        tag = f"#{engaged.track_id}"
        cv2.putText(out, tag, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_RED, 2)

    # State label
    state_str = reason.split(":")[0]
    col = COL_RED if "ENGAGED" in state_str else (
          COL_YELLOW if "HOLDING" in state_str else (80, 80, 80))
    cv2.putText(out, f"NEW: {state_str}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(out, reason[:60], (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # Enemy count label
    n = len(enemies)
    if n > 1:
        cv2.putText(out, f"{n} enemies in frame", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_YELLOW, 1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  required=True)
    ap.add_argument("--start",   type=int, default=0)
    ap.add_argument("--frames",  type=int, default=900)
    ap.add_argument("--output",  default="eval_results/engagement_validation")
    ap.add_argument("--config",  default="config/config.yaml")
    ap.add_argument("--radius",  type=float, default=STRICT_RADIUS_PX)
    ap.add_argument("--stable",  type=int,   default=MIN_STABLE_FRAMES)
    ap.add_argument("--hold",    type=int,   default=HOLD_FRAMES)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build pipeline (mirrors demo_generator._build_pipeline) ─────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(args.config).parent.parent / "models/training/enemy_detector/weights/best.pt"
    detector = EnemyDetector({
        "model_path":           "models/no_engine.engine",
        "fallback_model":       str(model_path),
        "confidence_threshold": cfg["detection"]["confidence_threshold"],
        "device":               device,
    })
    detector.load()
    detector.warmup(n_iters=3)

    classifier = EnemyClassifier(cfg.get("enemy_classification", {}))
    obj_filter = ObjectFilter(cfg.get("object_filter", {}))
    tracker    = ByteTrackWrapper(cfg.get("tracking", {}))
    tracker.load()

    # ── Open video first to get actual frame dimensions ──────────────────────
    cap_probe = cv2.VideoCapture(args.source)
    fw = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()
    if fw == 0:
        fw, fh = 1920, 1080

    eng_det = EngagementDetector(
        frame_width       = fw,
        frame_height      = fh,
        strict_radius_px  = args.radius,
        min_stable_frames = args.stable,
        hold_frames       = args.hold,
    )

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.source}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # Side-by-side output: half width each panel
    side_w = fw // 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(
        str(out_dir / "comparison.mp4"),
        fourcc, 6.0, (fw, fh),
    )

    csv_rows = []
    stats = defaultdict(int)
    multi_enemy_details = []

    t0 = time.time()
    frame_idx = 0

    while frame_idx < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != fw or frame.shape[0] != fh:
            frame = cv2.resize(frame, (fw, fh))

        # ── Detection pipeline ───────────────────────────────────────────────
        dets  = detector.detect(frame)
        dets  = classifier.classify(frame, dets)
        dets  = obj_filter.filter(dets, frame_width=fw, frame_height=fh)
        enemies = tracker.update(dets)

        # ── New engagement algorithm ─────────────────────────────────────────
        engaged, reason = eng_det.update(enemies, l2_held=True, r2_held=None)

        # ── Stats ────────────────────────────────────────────────────────────
        n_enemies = len(enemies)
        state_tag = reason.split(":")[0].split("(")[0]
        stats["total_frames"] += 1
        stats[f"state_{state_tag}"] += 1
        if n_enemies > 0:
            stats["frames_with_enemy"] += 1
        if n_enemies > 1:
            stats["multi_enemy_frames"] += 1
            multi_enemy_details.append({
                "frame": args.start + frame_idx,
                "n_enemies": n_enemies,
                "result": state_tag,
                "reason": reason,
                "engaged_id": engaged.track_id if engaged else None,
            })

        # ── CSV row ──────────────────────────────────────────────────────────
        csv_rows.append({
            "frame_abs": args.start + frame_idx,
            "frame_rel": frame_idx,
            "n_enemies": n_enemies,
            "state": state_tag,
            "reason": reason,
            "engaged_id": engaged.track_id if engaged else "",
            "engaged_cx": f"{engaged.cx:.1f}" if engaged else "",
            "engaged_cy": f"{engaged.cy:.1f}" if engaged else "",
            "engaged_dist_px": f"{math.hypot(engaged.cx - fw/2, engaged.cy - fh/2):.1f}" if engaged else "",
        })

        # ── Render ───────────────────────────────────────────────────────────
        left  = draw_left(frame, enemies)
        right = draw_right(frame, enemies, engaged, reason, args.radius)
        combined = np.hstack([
            cv2.resize(left,  (side_w, fh)),
            cv2.resize(right, (side_w, fh)),
        ])
        video_out.write(combined)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{frame_idx}/{args.frames}] {elapsed:.1f}s  "
                  f"last={reason[:50]}")

    cap.release()
    video_out.release()
    elapsed = time.time() - t0

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)

    # ── Metrics summary ──────────────────────────────────────────────────────
    total = stats["total_frames"]
    with_enemy = stats["frames_with_enemy"]
    multi = stats["multi_enemy_frames"]
    engaged_frames = stats["state_ENGAGED"] + stats["state_HOLDING"]
    no_box_frames  = total - engaged_frames

    print("\n" + "="*60)
    print("ENGAGEMENT VALIDATION RESULTS")
    print("="*60)
    print(f"Source          : {args.source}  (start={args.start})")
    print(f"Frames analyzed : {total}")
    print(f"Runtime         : {elapsed:.1f}s  ({total/elapsed:.1f} fps)")
    print()
    print(f"Frames with ≥1 enemy    : {with_enemy} ({100*with_enemy/total:.1f}%)")
    print(f"Frames with >1 enemy    : {multi}  ({100*multi/total:.1f}%)")
    print()
    print(f"Box shown (ENGAGED/HOLD): {engaged_frames} ({100*engaged_frames/total:.1f}%)")
    print(f"No box shown            : {no_box_frames} ({100*no_box_frames/total:.1f}%)")
    print()
    print("Breakdown of no-box reasons:")
    for key, val in sorted(stats.items()):
        if key.startswith("state_NO_BOX"):
            sub = key.replace("state_NO_BOX:", "  ").replace("state_", "  ")
            print(f"  {key.replace('state_NO_BOX:','').replace('state_',''): <30} {val}")
    print()
    print(f"ENGAGED frames          : {stats['state_ENGAGED']}")
    print(f"HOLDING frames          : {stats['state_HOLDING']}")
    print()

    # Multi-enemy analysis
    if multi_enemy_details:
        print(f"Multi-enemy frame analysis ({len(multi_enemy_details)} frames):")
        result_counts: Dict[str, int] = defaultdict(int)
        for d in multi_enemy_details:
            result_counts[d["result"]] += 1
        for result, cnt in sorted(result_counts.items()):
            print(f"  {result: <25} {cnt} ({100*cnt/len(multi_enemy_details):.1f}%)")

        print()
        print("Sample multi-enemy frames:")
        for d in multi_enemy_details[:15]:
            print(f"  frame={d['frame']}  n={d['n_enemies']}  "
                  f"result={d['result']: <25}  "
                  f"engaged_id={d['engaged_id']}")

    print()
    print(f"Comparison video : {out_dir/'comparison.mp4'}")
    print(f"Per-frame CSV    : {csv_path}")
    print("="*60)


if __name__ == "__main__":
    main()
