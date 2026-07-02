"""
eval_selector.py — Evidence-based comparison of two target-selection strategies.

Compares on real UC4 footage WITHOUT requiring ground-truth labels.
Instead, measures what can be objectively measured:

  - Lock stability   (longer, fewer switches = better)
  - Disagreement cases (where the methods differ — critical frames to inspect)
  - Visual comparison  (annotated video for human review)

Because we have no ground truth, this script does NOT claim which method is
correct.  It surfaces the frames where they disagree so the user can judge.

Method A: bbox-contains priority + 20-frame temporal window  (current)
Method B: camera-tracking relative velocity + 30-frame temporal window  (proposed)

Usage
-----
  python tools/eval_selector.py \\
      --model  models/training/enemy_detector/weights/best.pt \\
      --video  dataset/videos/U8tiME2kLok.f399.mp4 \\
      --skip   10800 --frames 900

Outputs
-------
  eval_results/comparison.mp4    annotated comparison video
  eval_results/report.txt        per-sequence statistics + analysis
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.detection.detector import Detection, EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.detection.target_selector import SelectState, TargetInfo, TargetSelector


# ─────────────────────────────────────────────────────────────────
# Method B: Camera-Tracking Selector
# (proposed — isolated here for eval only, not yet in production)
# ─────────────────────────────────────────────────────────────────

@dataclass
class _TrkRecord:
    track_id:       int
    last_cx:        float = 0.0
    last_cy:        float = 0.0
    frames_visible: int   = 0
    frames_absent:  int   = 0
    score_window: deque = field(default_factory=lambda: deque(maxlen=30))

    @property
    def temporal_score(self) -> float:
        if not self.score_window:
            return 0.0
        return sum(self.score_window) / len(self.score_window)

    @property
    def visibility_score(self) -> float:
        return min(self.frames_visible / 60.0, 1.0)

    @property
    def combined_score(self) -> float:
        return 0.70 * self.temporal_score + 0.30 * self.visibility_score


def _bbox_t(det: Detection) -> Tuple[int, int, int, int]:
    return (int(det.x1), int(det.y1), int(det.x2), int(det.y2))


def _iou_t(a: Tuple, b: Optional[Tuple]) -> float:
    if b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    i   = (ix2 - ix1) * (iy2 - iy1)
    aa  = (ax2 - ax1) * (ay2 - ay1)
    ab  = (bx2 - bx1) * (by2 - by1)
    return i / (aa + ab - i + 1e-6)


class CameraTrackingSelector:
    """
    Selects the enemy the player's camera is actively following.

    Core signal: camera-corrected relative screen velocity.
      camera_velocity  = mean displacement of all visible enemies (frame N → N+1)
      relative_velocity[i] = enemy_i displacement − camera_velocity
      frame_stability[i]   = 1 / (1 + |relative_velocity[i]| / PIXEL_SCALE)

    Rolling 30-frame mean of frame_stability = temporal_tracking_score.
    Combined with a visibility score (how long the enemy has been in frame).

    No fixed screen zones.  No crosshair geometry.  Purely kinematic.
    """

    _PIXEL_SCALE   = 10.0   # px/frame: relative speed that halves the score
    _W_TRACKING    = 0.70   # weight for temporal tracking score
    _W_VISIBILITY  = 0.30   # weight for continuous visibility
    _MIN_VISIBLE   = 10     # gate: frames enemy must be visible before acquisition
    _ACQ_THRESHOLD = 0.35   # min combined score for lock acquisition
    _NN_RADIUS_PX  = 64.0
    _IOU_THRESH    = 0.25
    _PROX_RADIUS   = 200.0  # px: centre-proximity fallback search radius

    def __init__(
        self,
        max_lost_frames:  int   = 8,
        iou_match_thresh: float = 0.25,
        aim_point_ratio:  float = 0.30,
    ) -> None:
        self._max_lost   = max_lost_frames
        self._iou_thresh = iou_match_thresh
        self._aim_ratio  = aim_point_ratio

        self._state:      SelectState                      = SelectState.UNLOCKED
        self._locked_id:  Optional[int]                    = None
        self._locked_box: Optional[Tuple[int,int,int,int]] = None
        self._lost_count: int                              = 0
        self._records:    Dict[int, _TrkRecord]            = {}
        self._next_id:    int                              = 3_000_000

        # Populated after every update() — used by the evaluation report
        self.diag: dict = {}

    @property
    def state(self) -> SelectState:
        return self._state

    @property
    def locked_id(self) -> Optional[int]:
        return self._locked_id

    def reset(self) -> None:
        self._state      = SelectState.UNLOCKED
        self._locked_id  = None
        self._locked_box = None
        self._lost_count = 0
        self._records.clear()

    def update(
        self,
        enemies:     List[Detection],
        screen_w:    int,
        screen_h:    int,
        crosshair_x: Optional[float] = None,
        crosshair_y: Optional[float] = None,
        frame:       Optional[np.ndarray] = None,
    ) -> Optional[TargetInfo]:

        # ── Step 1: ID assignment + compute per-enemy displacements ──
        assigned: List[Tuple[Detection, int]]   = []
        disps:    List[Tuple[float, float]]     = []
        has_prev: List[bool]                    = []
        claimed:  set                           = set()

        for det in enemies:
            if det.track_id is not None:
                tid = det.track_id
                if tid not in self._records:
                    self._records[tid] = _TrkRecord(tid, det.cx, det.cy)
                    prev = False
                else:
                    prev = self._records[tid].frames_visible > 0
                self._records[tid].frames_absent = 0
                claimed.add(tid)
            else:
                best_id, best_d = None, self._NN_RADIUS_PX
                for tid2, rec2 in self._records.items():
                    if rec2.frames_absent > 0 or tid2 in claimed:
                        continue
                    d = math.hypot(rec2.last_cx - det.cx, rec2.last_cy - det.cy)
                    if d < best_d:
                        best_d, best_id = d, tid2
                if best_id is not None:
                    tid = best_id
                    prev = self._records[tid].frames_visible > 0
                    self._records[tid].frames_absent = 0
                    claimed.add(tid)
                else:
                    tid = self._next_id
                    self._next_id += 1
                    self._records[tid] = _TrkRecord(tid, det.cx, det.cy)
                    prev = False
                    claimed.add(tid)

            rec = self._records[tid]
            if prev:
                disp = (det.cx - rec.last_cx, det.cy - rec.last_cy)
            else:
                disp = (0.0, 0.0)

            assigned.append((det, tid))
            disps.append(disp)
            has_prev.append(prev)

        # ── Step 2: Estimate camera velocity from mean displacement ──
        # Use only enemies with a previous position (not first-frame appearances)
        valid_disps = [disps[i] for i in range(len(disps)) if has_prev[i]]
        if valid_disps:
            cam_dx = sum(d[0] for d in valid_disps) / len(valid_disps)
            cam_dy = sum(d[1] for d in valid_disps) / len(valid_disps)
        else:
            cam_dx, cam_dy = 0.0, 0.0
        cam_speed = math.hypot(cam_dx, cam_dy)

        # ── Step 3: Per-enemy relative velocity → frame stability score ──
        rel_vels: Dict[int, float] = {}
        scores:   Dict[int, float] = {}

        for i, (det, tid) in enumerate(assigned):
            rec = self._records[tid]
            if has_prev[i]:
                rel_dx   = disps[i][0] - cam_dx
                rel_dy   = disps[i][1] - cam_dy
                rel_spd  = math.hypot(rel_dx, rel_dy)
            else:
                rel_spd  = 0.0   # first appearance → neutral score

            rel_vels[tid] = rel_spd
            stab = 1.0 / (1.0 + rel_spd / self._PIXEL_SCALE)
            rec.score_window.append(stab)
            rec.frames_visible += 1
            rec.last_cx, rec.last_cy = det.cx, det.cy
            scores[tid] = rec.combined_score

        # Age absent records
        stale = []
        for tid2, rec2 in self._records.items():
            if tid2 not in claimed:
                rec2.frames_absent += 1
                if rec2.frames_absent > self._max_lost:
                    stale.append(tid2)
        for tid2 in stale:
            del self._records[tid2]

        # ── Diagnostics ──
        self.diag = {
            "cam_dx":    cam_dx,
            "cam_dy":    cam_dy,
            "cam_speed": cam_speed,
            "rel_vels":  rel_vels,
            "scores":    scores,
            "n_enemies": len(enemies),
        }

        if self._state == SelectState.UNLOCKED:
            return self._acquire(assigned)
        return self._maintain(assigned)

    # ── Private ────────────────────────────────────────────────────

    def _acquire(
        self, assigned: List[Tuple[Detection, int]]
    ) -> Optional[TargetInfo]:
        best_det, best_tid = None, None
        best_score: float  = self._ACQ_THRESHOLD

        for det, tid in assigned:
            rec = self._records[tid]
            if rec.frames_visible < self._MIN_VISIBLE:
                continue
            s = rec.combined_score
            if s > best_score:
                best_score, best_det, best_tid = s, det, tid

        if best_det is None:
            return None

        self._locked_id  = best_tid
        self._locked_box = _bbox_t(best_det)
        self._lost_count = 0
        self._state      = SelectState.LOCKED
        return self._info(best_det, best_tid)

    def _maintain(
        self, assigned: List[Tuple[Detection, int]]
    ) -> Optional[TargetInfo]:
        found = self._find(assigned)
        if found is None:
            self._lost_count += 1
            if self._lost_count >= self._max_lost:
                self.reset()
            return None
        det, tid           = found
        self._locked_box   = _bbox_t(det)
        self._locked_id    = tid
        self._lost_count   = 0
        return self._info(det, tid)

    def _find(
        self, assigned: List[Tuple[Detection, int]]
    ) -> Optional[Tuple[Detection, int]]:
        if self._locked_id is None or self._locked_box is None:
            return None
        for det, tid in assigned:
            if tid == self._locked_id:
                return det, tid
        best_iou, best_det, best_tid = self._iou_thresh, None, None
        for det, tid in assigned:
            iou = _iou_t(_bbox_t(det), self._locked_box)
            if iou > best_iou:
                best_iou, best_det, best_tid = iou, det, tid
        if best_det is not None:
            return best_det, best_tid
        lx1, ly1, lx2, ly2 = self._locked_box
        lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        best_d, best_det, best_tid = self._PROX_RADIUS, None, None
        for det, tid in assigned:
            d = math.hypot(det.cx - lcx, det.cy - lcy)
            if d < best_d:
                best_d, best_det, best_tid = d, det, tid
        return (best_det, best_tid) if best_det else None

    def _info(self, det: Detection, tid: int) -> TargetInfo:
        bh    = det.y2 - det.y1
        aim_y = det.y1 + bh * self._aim_ratio
        rec   = self._records.get(tid)
        return TargetInfo(
            bbox       = _bbox_t(det),
            center     = (det.cx, det.cy),
            aim_point  = (det.cx, aim_y),
            confidence = det.confidence,
            track_id   = tid,
            state      = self._state,
            score      = rec.combined_score if rec else 0.0,
        )


# ─────────────────────────────────────────────────────────────────
# Detection extraction
# ─────────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    idx:       int
    dets:      List[Detection]
    n_enemies: int


def extract_detections(
    video_path: Path,
    model_path: Path,
    skip:       int,
    max_frames: int,
    conf:       float = 0.45,
) -> Tuple[List[FrameData], int, int, float]:
    """
    GPU pass: run detector + classifier + filter on every frame.
    Returns (frame_data_list, width, height, src_fps).
    The raw frames are NOT stored — we re-read the video for the comparison video.
    """
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = EnemyDetector({
        "model_path":           "models/no_engine.engine",
        "fallback_model":       str(model_path),
        "confidence_threshold": conf,
        "device":               device,
    })
    detector.load()
    detector.warmup(n_iters=2)
    classifier = EnemyClassifier(cfg.get("enemy_classification", {}))
    obj_filter = ObjectFilter(cfg.get("object_filter", {}))

    cap     = cv2.VideoCapture(str(video_path))
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if skip:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip)

    results: List[FrameData] = []
    t0 = time.perf_counter()
    print(f"  Extracting detections ({max_frames} frames) …")

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        raw     = detector.detect(frame)
        classed = classifier.classify(frame, raw)
        enemies = obj_filter.filter(classed, W, H)
        results.append(FrameData(idx=skip + i, dets=enemies, n_enemies=len(enemies)))

        if (i + 1) % 100 == 0:
            fps = (i + 1) / max(time.perf_counter() - t0, 1e-6)
            print(f"    {i+1}/{max_frames}  ({fps:.1f} fps)")

    cap.release()
    print(f"  Done.  {len(results)} frames, {sum(f.n_enemies for f in results)} total detections.")
    return results, W, H, src_fps


# ─────────────────────────────────────────────────────────────────
# Replay selectors against cached detections
# ─────────────────────────────────────────────────────────────────

@dataclass
class Decision:
    frame_idx:  int
    target:     Optional[TargetInfo]
    state:      SelectState
    n_enemies:  int
    reason:     str   # human-readable explanation for this decision


def replay_method_a(
    frames: List[FrameData], W: int, H: int
) -> List[Decision]:
    """Replay the current TargetSelector (bbox-contains + 20-frame temporal)."""
    sel   = TargetSelector()
    cx, cy = W / 2.0, H / 2.0
    out: List[Decision] = []

    for fd in frames:
        t = sel.update(fd.dets, W, H)

        # Reconstruct reason
        if t is None:
            reason = "no-target"
            if fd.n_enemies > 0:
                reason = f"no-target  ({fd.n_enemies} enemies, none near crosshair)"
        else:
            x1, y1, x2, y2 = t.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                reason = f"bbox-contains  score={t.score:.2f}"
            else:
                reason = f"temporal-proximity  score={t.score:.2f}"
            if t.state == SelectState.LOCKED:
                reason = "LOCKED  " + reason

        out.append(Decision(fd.idx, t, sel.state, fd.n_enemies, reason))

    return out


def replay_method_b(
    frames: List[FrameData], W: int, H: int
) -> List[Decision]:
    """Replay CameraTrackingSelector (relative velocity + 30-frame temporal)."""
    sel   = CameraTrackingSelector()
    out: List[Decision] = []

    for fd in frames:
        t = sel.update(fd.dets, W, H)
        d = sel.diag

        if t is None:
            reason = "no-target"
            if fd.n_enemies > 0:
                # Find best candidate score to explain why it didn't lock
                best_score = max(d["scores"].values()) if d.get("scores") else 0.0
                reason = (
                    f"no-target  ({fd.n_enemies} enemies, "
                    f"cam={d.get('cam_speed',0):.1f}px/f, "
                    f"best_score={best_score:.2f} < threshold)"
                )
        else:
            tid     = t.track_id
            rel_vel = d.get("rel_vels", {}).get(tid, 0.0)
            cam_spd = d.get("cam_speed", 0.0)
            reason  = (
                f"tracking  score={t.score:.2f}  "
                f"rel_vel={rel_vel:.1f}px/f  "
                f"cam={cam_spd:.1f}px/f"
            )
            if t.state == SelectState.LOCKED:
                reason = "LOCKED  " + reason

        out.append(Decision(fd.idx, t, sel.state, fd.n_enemies, reason))

    return out


# ─────────────────────────────────────────────────────────────────
# Agreement analysis
# ─────────────────────────────────────────────────────────────────

def targets_agree(a: Optional[TargetInfo], b: Optional[TargetInfo]) -> Tuple[bool, str]:
    """
    Return (agree, disagreement_type).
    Same enemy = IoU > 0.40 between selected bboxes.
    """
    if a is None and b is None:
        return True, "both-none"
    if a is None:
        return False, "A=none  B=target"
    if b is None:
        return False, "A=target  B=none"
    iou = _iou_t(a.bbox, b.bbox)
    if iou >= 0.40:
        return True, "same-enemy"
    return False, "different-enemy"


@dataclass
class SeqStats:
    name:          str
    n_frames:      int
    a_locked:      int
    b_locked:      int
    a_lock_events: int
    b_lock_events: int
    agreements:    int
    type_both_none:      int = 0
    type_a_none_b_tgt:   int = 0
    type_a_tgt_b_none:   int = 0
    type_diff_enemy:     int = 0

    @property
    def a_locked_pct(self):  return 100 * self.a_locked / max(self.n_frames, 1)
    @property
    def b_locked_pct(self):  return 100 * self.b_locked / max(self.n_frames, 1)
    @property
    def agree_pct(self):     return 100 * self.agreements / max(self.n_frames, 1)
    @property
    def disagree_pct(self):  return 100 - self.agree_pct


def analyze_sequence(
    name:     str,
    dec_a:    List[Decision],
    dec_b:    List[Decision],
) -> Tuple[SeqStats, List[int]]:
    """Compute metrics for one sequence. Returns (stats, disagree_frame_indices)."""
    a_locked = a_events = 0
    b_locked = b_events = 0
    agreements = 0
    disagree_indices: List[int] = []
    prev_a_state = SelectState.UNLOCKED
    prev_b_state = SelectState.UNLOCKED
    stats = SeqStats(name=name, n_frames=len(dec_a),
                     a_locked=0, b_locked=0,
                     a_lock_events=0, b_lock_events=0,
                     agreements=0)

    for i, (da, db) in enumerate(zip(dec_a, dec_b)):
        if da.state == SelectState.LOCKED:
            a_locked += 1
        if db.state == SelectState.LOCKED:
            b_locked += 1
        if da.state == SelectState.LOCKED and prev_a_state == SelectState.UNLOCKED:
            a_events += 1
        if db.state == SelectState.LOCKED and prev_b_state == SelectState.UNLOCKED:
            b_events += 1

        agree, dtype = targets_agree(da.target, db.target)
        if agree:
            agreements += 1
            if dtype == "both-none":
                stats.type_both_none += 1
        else:
            disagree_indices.append(i)
            if dtype == "A=none  B=target":
                stats.type_a_none_b_tgt += 1
            elif dtype == "A=target  B=none":
                stats.type_a_tgt_b_none += 1
            elif dtype == "different-enemy":
                stats.type_diff_enemy += 1

        prev_a_state = da.state
        prev_b_state = db.state

    stats.a_locked      = a_locked
    stats.b_locked      = b_locked
    stats.a_lock_events = a_events
    stats.b_lock_events = b_events
    stats.agreements    = agreements
    return stats, disagree_indices


# ─────────────────────────────────────────────────────────────────
# Comparison video
# ─────────────────────────────────────────────────────────────────

_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (0, 220, 0)
_BLUE  = (220, 100, 0)
_CYAN  = (200, 200, 0)
_GRAY  = (120, 120, 120)
_WHITE = (255, 255, 255)
_RED   = (0, 80, 255)


def _put(img, text, x, y, color=_WHITE, scale=0.45, thick=1):
    cv2.putText(img, text, (int(x), int(y)), _FONT, scale, color, thick, cv2.LINE_AA)


def annotate_frame(
    frame:  np.ndarray,
    dets:   List[Detection],
    da:     Decision,
    db:     Decision,
    seq_id: int,
    global_stats: dict,
) -> np.ndarray:
    """
    Draw a single comparison frame with both methods' selections overlaid.

    Color convention:
      Gray thin  = all detected enemies
      Green bold = Method A selection only
      Blue bold  = Method B selection only
      Cyan bold  = both methods agree (same enemy)
      Crosshair  = screen centre (white)
    """
    out = frame.copy()
    H, W = out.shape[:2]
    cx, cy = W // 2, H // 2

    # All detections (thin gray)
    for det in dets:
        cv2.rectangle(out,
                      (int(det.x1), int(det.y1)),
                      (int(det.x2), int(det.y2)),
                      _GRAY, 1)

    agree, _ = targets_agree(da.target, db.target)

    if agree and da.target is not None:
        # Both agree — cyan
        x1, y1, x2, y2 = da.target.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), _CYAN, 3)
        _put(out, f"AGREE  A={da.target.score:.2f}", x1, y1 - 8, _CYAN, 0.45)
    else:
        if da.target is not None:
            x1, y1, x2, y2 = da.target.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), _GREEN, 3)
            _put(out, f"A  {da.target.score:.2f}", x1, y1 - 8, _GREEN, 0.45)
        if db.target is not None:
            x1, y1, x2, y2 = db.target.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), _BLUE, 3)
            _put(out, f"B  {db.target.score:.2f}", x1, y2 + 16, _BLUE, 0.45)

    # Screen centre crosshair
    cv2.drawMarker(out, (cx, cy), _WHITE, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

    # AGREE / DISAGREE banner
    banner      = "AGREE" if agree else "DISAGREE"
    banner_col  = _CYAN if agree else _RED
    _put(out, banner, W - 130, 28, banner_col, 0.65, 2)

    # Status bar (bottom)
    bar_h = 52
    cv2.rectangle(out, (0, H - bar_h), (W, H), (0, 0, 0), -1)

    a_state = "A:LOCKED" if da.state == SelectState.LOCKED else "A:UNLOCK"
    b_state = "B:LOCKED" if db.state == SelectState.LOCKED else "B:UNLOCK"
    a_col   = _GREEN if da.state == SelectState.LOCKED else _GRAY
    b_col   = _BLUE  if db.state == SelectState.LOCKED else _GRAY

    _put(out, a_state, 8,       H - 32, a_col, 0.50, 1)
    _put(out, da.reason[:60],   8,       H - 12, a_col, 0.38, 1)
    _put(out, b_state, W // 2,  H - 32, b_col, 0.50, 1)
    _put(out, db.reason[:60],   W // 2,  H - 12, b_col, 0.38, 1)

    # Running totals (top-left)
    tot   = global_stats["total"]
    agr   = global_stats["agree"]
    a_lk  = global_stats["a_locked"]
    b_lk  = global_stats["b_locked"]
    seg   = f"seq{seq_id:02d}  frm{da.frame_idx}"
    cv2.rectangle(out, (0, 0), (320, 60), (0, 0, 0), -1)
    _put(out, seg,                              6, 18, _WHITE, 0.42)
    _put(out, f"agree {agr}/{tot} ({100*agr//max(tot,1)}%)", 6, 36, _CYAN,  0.42)
    _put(out, f"A-locked {a_lk}  B-locked {b_lk}",          6, 54, _WHITE, 0.38)

    return out


def generate_comparison_video(
    video_path:  Path,
    skip:        int,
    dec_a:       List[Decision],
    dec_b:       List[Decision],
    frames_data: List[FrameData],
    output_path: Path,
    src_fps:     float,
    seq_size:    int,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (W, H),
    )

    gstats = {"total": 0, "agree": 0, "a_locked": 0, "b_locked": 0}
    print(f"  Writing comparison video → {output_path.name} …")

    for i, (fd, da, db) in enumerate(zip(frames_data, dec_a, dec_b)):
        ret, frame = cap.read()
        if not ret:
            break

        agree, _ = targets_agree(da.target, db.target)
        gstats["total"] += 1
        if agree:
            gstats["agree"] += 1
        if da.state == SelectState.LOCKED:
            gstats["a_locked"] += 1
        if db.state == SelectState.LOCKED:
            gstats["b_locked"] += 1

        seq_id = i // seq_size
        out    = annotate_frame(frame, fd.dets, da, db, seq_id, gstats)
        writer.write(out)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(dec_a)}")

    cap.release()
    writer.release()
    print(f"  Done.")


# ─────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────

def build_report(
    seq_stats:     List[SeqStats],
    all_dec_a:     List[Decision],
    all_dec_b:     List[Decision],
    frames_data:   List[FrameData],
    W: int, H: int,
) -> str:
    cx, cy = W / 2.0, H / 2.0
    lines  = []
    add    = lines.append

    add("=" * 70)
    add("UC4 TARGET SELECTOR EVALUATION — Method A vs Method B")
    add("=" * 70)
    add("")
    add("Method A : bbox-contains priority + 20-frame temporal window  (current)")
    add("Method B : camera-tracking relative velocity + 30-frame temporal  (proposed)")
    add("")
    add("NOTE: No ground-truth labels.  'Better' = more stable locks,")
    add("      fewer spurious acquisitions, longer coherent lock periods.")
    add("      Visual review of comparison.mp4 is required for a final verdict.")
    add("")

    # ── Overall ──────────────────────────────────────────────────
    tot      = len(all_dec_a)
    a_locked = sum(1 for d in all_dec_a if d.state == SelectState.LOCKED)
    b_locked = sum(1 for d in all_dec_b if d.state == SelectState.LOCKED)
    a_events = sum(1 for i in range(1, tot)
                   if all_dec_a[i].state == SelectState.LOCKED
                   and all_dec_a[i-1].state == SelectState.UNLOCKED)
    b_events = sum(1 for i in range(1, tot)
                   if all_dec_b[i].state == SelectState.LOCKED
                   and all_dec_b[i-1].state == SelectState.UNLOCKED)
    agree    = sum(1 for da, db in zip(all_dec_a, all_dec_b)
                   if targets_agree(da.target, db.target)[0])

    add("OVERALL  ({} frames)".format(tot))
    add("-" * 50)
    add(f"{'':30s}  {'Method A':>12s}  {'Method B':>12s}")
    add(f"{'LOCKED frames':30s}  {a_locked:>10d}   {b_locked:>10d}")
    add(f"{'LOCKED %':30s}  {100*a_locked//max(tot,1):>10d}%  {100*b_locked//max(tot,1):>10d}%")
    add(f"{'Lock acquisitions':30s}  {a_events:>10d}   {b_events:>10d}")
    if a_events > 0:
        add(f"{'Avg lock duration (frames)':30s}  {a_locked//a_events:>10d}   "
            f"{b_locked//max(b_events,1):>10d}")
    add(f"{'Agreement':30s}  {agree}/{tot} ({100*agree//max(tot,1)}%)")
    add("")

    # ── Per-sequence ─────────────────────────────────────────────
    add("PER-SEQUENCE BREAKDOWN")
    add("-" * 70)
    add(f"{'Seq':>4s}  {'Frames':>7s}  {'A-Lk%':>6s}  {'B-Lk%':>6s}  "
        f"{'Agree%':>7s}  {'D:A>B':>6s}  {'D:B>A':>6s}  {'D:Diff':>7s}")
    add("-" * 70)
    for s in seq_stats:
        add(f"{s.name:>4s}  {s.n_frames:>7d}  "
            f"{s.a_locked_pct:>5.0f}%  {s.b_locked_pct:>5.0f}%  "
            f"{s.agree_pct:>6.0f}%  "
            f"{s.type_a_tgt_b_none:>6d}  "
            f"{s.type_a_none_b_tgt:>6d}  "
            f"{s.type_diff_enemy:>7d}")
    add("-" * 70)
    add("D:A>B = A locked, B did not   |  D:B>A = B locked, A did not   |  D:Diff = different enemy")
    add("")

    # ── Disagreement analysis ─────────────────────────────────────
    total_disagree = tot - agree
    add(f"DISAGREEMENT ANALYSIS  ({total_disagree} frames, {100*total_disagree//max(tot,1)}%)")
    add("-" * 70)

    t_a_none_b  = sum(s.type_a_none_b_tgt for s in seq_stats)
    t_a_b_none  = sum(s.type_a_tgt_b_none for s in seq_stats)
    t_diff      = sum(s.type_diff_enemy    for s in seq_stats)

    if total_disagree > 0:
        add(f"  Type 1 — A=target, B=none      : {t_a_b_none:4d} frames  "
            f"({100*t_a_b_none//max(total_disagree,1):3d}%)")
        add(f"    A locked (bbox-contains or proximity) but B's tracking signal")
        add(f"    did not reach threshold.  Suggests A may be locking prematurely")
        add(f"    on enemies that briefly overlap the crosshair region.")
        add("")
        add(f"  Type 2 — A=none, B=target       : {t_a_none_b:4d} frames  "
            f"({100*t_a_none_b//max(total_disagree,1):3d}%)")
        add(f"    B identified a tracking target that A's crosshair/proximity")
        add(f"    test missed.  Suggests B is better at picking up targets that")
        add(f"    are not near the screen centre but are being camera-followed.")
        add("")
        add(f"  Type 3 — different enemy         : {t_diff:4d} frames  "
            f"({100*t_diff//max(total_disagree,1):3d}%)")
        add(f"    Both locked but on different enemies.  These are the critical")
        add(f"    frames — inspect them in comparison.mp4 to judge which is correct.")
        add("")

    # ── Qualitative indicators ────────────────────────────────────
    add("STABILITY INDICATORS  (higher = better, for equal combat exposure)")
    add("-" * 70)

    # Average lock duration
    a_dur = a_locked // max(a_events, 1)
    b_dur = b_locked // max(b_events, 1)
    add(f"  Avg lock duration    A: {a_dur:4d} frames    B: {b_dur:4d} frames")
    winner_dur = "A" if a_dur > b_dur else ("B" if b_dur > a_dur else "TIE")
    add(f"  → Longer avg duration: Method {winner_dur}")
    add("")

    # Acquisition rate (lower = more selective = better for avoiding false locks)
    add(f"  Lock acquisitions    A: {a_events:4d}    B: {b_events:4d}")
    winner_acq = "B" if b_events < a_events else ("A" if a_events < b_events else "TIE")
    add(f"  → Fewer acquisitions (more selective): Method {winner_acq}")
    add("")

    # ── Honest conclusion ─────────────────────────────────────────
    add("HONEST CONCLUSION")
    add("-" * 70)
    add("These metrics are NECESSARY but NOT SUFFICIENT to declare a winner.")
    add("A method with longer locks may simply be locking on the wrong enemy")
    add("and never releasing.  A method with fewer acquisitions may be too")
    add("conservative and miss real engagements.")
    add("")
    add("Required next step: watch comparison.mp4 and find 5-10 frames where")
    add("the methods disagree (DISAGREE banner, orange).  For each one, ask:")
    add("  - Which enemy is DevoManiac clearly shooting at?")
    add("  - Which method selected that enemy?")
    add("")
    add("If Method B wins more than 60% of disagreement inspections →")
    add("replace Method A.  Otherwise keep Method A.")
    add("")
    add("=" * 70)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Compare target selectors on UC4 footage.")
    ap.add_argument("--model",  default="models/training/enemy_detector/weights/best.pt")
    ap.add_argument("--video",  default="dataset/videos/U8tiME2kLok.f399.mp4")
    ap.add_argument("--skip",   type=int, default=10800,
                    help="Start frame (default 10800 = 3 min into video)")
    ap.add_argument("--frames", type=int, default=900,
                    help="Total frames to evaluate (default 900 = 15 s at 60 fps)")
    ap.add_argument("--seqs",   type=int, default=9,
                    help="Number of sequences to divide footage into")
    ap.add_argument("--output", default="eval_results")
    ap.add_argument("--conf",   type=float, default=0.45)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Extracting detections (GPU pass, runs once) …")
    frames_data, W, H, src_fps = extract_detections(
        Path(args.video), Path(args.model),
        args.skip, args.frames, args.conf,
    )

    print("\n[2/5] Replaying Method A (bbox-contains + temporal) …")
    dec_a = replay_method_a(frames_data, W, H)

    print("\n[3/5] Replaying Method B (camera-tracking) …")
    dec_b = replay_method_b(frames_data, W, H)

    print("\n[4/5] Analysing sequences …")
    seq_size  = max(1, args.frames // args.seqs)
    seq_stats = []

    for s in range(args.seqs):
        lo = s * seq_size
        hi = min(lo + seq_size, len(dec_a))
        if lo >= len(dec_a):
            break
        stats, _ = analyze_sequence(
            f"s{s:02d}", dec_a[lo:hi], dec_b[lo:hi]
        )
        seq_stats.append(stats)

    report_text = build_report(seq_stats, dec_a, dec_b, frames_data, W, H)
    report_path = out_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)

    print("\n[5/5] Generating comparison video …")
    generate_comparison_video(
        Path(args.video), args.skip,
        dec_a, dec_b, frames_data,
        out_dir / "comparison.mp4",
        src_fps, seq_size,
    )

    print(f"\nOutputs:")
    print(f"  {out_dir / 'comparison.mp4'}")
    print(f"  {out_dir / 'report.txt'}")
    print("\nNext step: watch comparison.mp4 and inspect orange DISAGREE frames.")
    print("Each DISAGREE frame shows Green=MethodA, Blue=MethodB, Cyan=agree.")


if __name__ == "__main__":
    main()
