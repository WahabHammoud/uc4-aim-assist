"""
Temporal-priority target selector for UC4 aim assist.  Option B+C.

WHY THIS ARCHITECTURE
---------------------
The previous version (proximity-only) measured crosshair-to-enemy-CENTER
distance.  This misclassifies a common case:

    Enemy A: center 80 px from crosshair, bbox 60 px wide
             → nearest bbox edge is 50 px away
             → NOT being aimed at

    Enemy B: center 120 px from crosshair, bbox 160 px wide
             → crosshair is INSIDE the bbox
             → IS being aimed at

The old version locked Enemy A.  The correct answer is Enemy B.

This version fixes that by using two signals:

  1. CROSSHAIR OVERLAP SCORE (per frame, per enemy)
     - Screen centre inside enemy bbox  →  score = 1.0  (ground truth)
     - Screen centre outside bbox       →  score decays linearly with
                                           distance to nearest EDGE point
                                           (not centre), 0.75 → 0.0

  2. TEMPORAL CONSISTENCY (rolling 20-frame mean of overlap scores)
     Separates "consistently targeted" from "briefly crossed centre."
     Enemy who has had crosshair on them for 300 ms scores 0.75.
     Enemy who crossed centre for 1 frame scores 0.05.

ACQUISITION LOGIC (UNLOCKED → LOCKED)
--------------------------------------
  Phase 1 — bbox-contains (immediate, no threshold needed):
    Any enemy whose bbox contains the crosshair right now → lock immediately.
    If multiple: pick the one whose centre is closest to crosshair.

  Phase 2 — temporal threshold (sustained near-crosshair presence):
    If no bbox-contains candidate: find enemy with highest temporal_score.
    Lock if temporal_score > acq_threshold (default 0.25, ~5/20 frames).

  Phase 3 — no candidate → return None (no box drawn).

LOCK MAINTENANCE (LOCKED → LOCKED)
------------------------------------
  Search by: stable ID → IoU → centre proximity (200 px).
  NEVER switch targets while locked.
  Release after max_lost_frames consecutive absent frames.

MILESTONE 2 COMPATIBILITY
--------------------------
  Public API unchanged.  frame= kwarg accepted but ignored.
  TargetInfo.aim_point feeds directly into PID.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.detector import Detection


# ─────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────

class SelectState(Enum):
    UNLOCKED = auto()
    LOCKED   = auto()


@dataclass
class TargetInfo:
    """All information needed downstream about the current locked target."""
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixels)
    center:     Tuple[float, float]
    aim_point:  Tuple[float, float]         # PID controller input
    confidence: float
    track_id:   int
    state:      SelectState
    score:      float = 0.0                 # temporal_score [0-1], shown in overlay


# ─────────────────────────────────────────────────────────────────
# Internal: per-enemy rolling record
# ─────────────────────────────────────────────────────────────────

@dataclass
class _EnemyRecord:
    track_id:       int
    last_cx:        float = 0.0
    last_cy:        float = 0.0
    frames_visible: int   = 0
    frames_absent:  int   = 0
    # Rolling window of per-frame crosshair-overlap scores.
    # Mean of this window = temporal_score.
    score_window: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def temporal_score(self) -> float:
        if not self.score_window:
            return 0.0
        return sum(self.score_window) / len(self.score_window)


# ─────────────────────────────────────────────────────────────────
# Per-frame overlap score  (the key improvement over center-distance)
# ─────────────────────────────────────────────────────────────────

def _overlap_score(det: Detection, cx: float, cy: float, edge_margin: float) -> float:
    """
    How much does the screen crosshair (cx, cy) overlap this detection?

    Inside bbox  →  1.0   (crosshair is literally on the enemy's body)
    At bbox edge →  0.75  (crosshair just outside the body)
    At margin    →  0.0   (crosshair too far away to be aiming here)

    The critical difference from center-distance: we measure from the
    crosshair to the NEAREST POINT ON THE BBOX EDGE, not to the center.
    This correctly handles large bboxes where the crosshair is inside the
    body even though the center is far away.
    """
    inside_x = det.x1 <= cx <= det.x2
    inside_y = det.y1 <= cy <= det.y2

    if inside_x and inside_y:
        return 1.0

    # Nearest point on the rectangle to (cx, cy)
    near_x = max(det.x1, min(cx, det.x2))
    near_y = max(det.y1, min(cy, det.y2))
    edge_dist = math.hypot(cx - near_x, cy - near_y)

    # Linear decay: 0.75 at edge → 0.0 at edge_margin pixels away
    return max(0.0, 1.0 - edge_dist / edge_margin) * 0.75


# ─────────────────────────────────────────────────────────────────
# TargetSelector
# ─────────────────────────────────────────────────────────────────

class TargetSelector:
    """
    Temporal-priority target selector with sticky lock.

    Parameters
    ----------
    max_lost_frames  : consecutive absent frames before lock is released
    temporal_window  : rolling window size for temporal score (frames)
    acq_threshold    : min temporal_score to acquire by proximity (0-1)
    edge_margin_px   : px outside bbox edge that earns partial score
    iou_match_thresh : IoU threshold for frame-to-frame lock tracking
    aim_point_ratio  : fraction of bbox height used as aim point (0.30=chest)
    lock_radius_px   : ignored, kept for backward-compatible call sites
    """

    def __init__(
        self,
        max_lost_frames:  int   = 8,
        temporal_window:  int   = 20,
        acq_threshold:    float = 0.25,
        edge_margin_px:   float = 100.0,
        iou_match_thresh: float = 0.25,
        aim_point_ratio:  float = 0.30,
        lock_radius_px:   float = 150.0,   # backward compat, unused
    ) -> None:
        self._max_lost      = max_lost_frames
        self._win_size      = temporal_window
        self._acq_threshold = acq_threshold
        self._edge_margin   = edge_margin_px
        self._iou_thresh    = iou_match_thresh
        self._aim_ratio     = aim_point_ratio

        self._state:       SelectState                      = SelectState.UNLOCKED
        self._locked_id:   Optional[int]                    = None
        self._locked_box:  Optional[Tuple[int,int,int,int]] = None
        self._lost_count:  int                              = 0

        self._records:  Dict[int, _EnemyRecord] = {}
        self._next_id:  int                     = 2_000_000

    # ── Public API ────────────────────────────────────────────────

    def update(
        self,
        enemies:     List[Detection],
        screen_w:    int,
        screen_h:    int,
        crosshair_x: Optional[float] = None,
        crosshair_y: Optional[float] = None,
        frame:       Optional[np.ndarray] = None,   # ignored, backward compat
    ) -> Optional[TargetInfo]:
        cx = crosshair_x if crosshair_x is not None else screen_w / 2.0
        cy = crosshair_y if crosshair_y is not None else screen_h / 2.0

        # Assign stable IDs and update temporal records for all detections
        assigned = self._update_records(enemies, cx, cy)

        # Age absent records; prune after max_lost_frames
        seen = {tid for _, tid in assigned}
        stale = [
            tid for tid, rec in self._records.items()
            if tid not in seen and rec.frames_absent > self._max_lost
        ]
        for tid in stale:
            del self._records[tid]

        if self._state == SelectState.UNLOCKED:
            return self._try_acquire(assigned, cx, cy)
        return self._maintain_lock(assigned, cx, cy)

    def reset(self) -> None:
        self._state      = SelectState.UNLOCKED
        self._locked_id  = None
        self._locked_box = None
        self._lost_count = 0
        self._records.clear()

    @property
    def state(self) -> SelectState:
        return self._state

    @property
    def locked_id(self) -> Optional[int]:
        return self._locked_id

    # ── Record management ─────────────────────────────────────────

    def _update_records(
        self,
        enemies: List[Detection],
        cx: float,
        cy: float,
    ) -> List[Tuple[Detection, int]]:
        """
        Assign a stable internal ID to each detection.
        ByteTrack IDs used directly when available.
        Otherwise nearest-neighbour within 64 px.

        For every assigned detection: append this frame's overlap score to
        its rolling window and update last-known position.
        """
        assigned: List[Tuple[Detection, int]] = []
        claimed: set = set()

        for det in enemies:
            if det.track_id is not None:
                tid = det.track_id
                if tid not in self._records:
                    self._records[tid] = _EnemyRecord(tid, det.cx, det.cy)
                self._records[tid].frames_absent = 0
                claimed.add(tid)
            else:
                # NN fallback: associate with closest non-absent, unclaimed record
                best_id, best_d = None, 64.0
                for tid, rec in self._records.items():
                    if rec.frames_absent > 0 or tid in claimed:
                        continue
                    d = math.hypot(rec.last_cx - det.cx, rec.last_cy - det.cy)
                    if d < best_d:
                        best_d, best_id = d, tid

                if best_id is not None:
                    self._records[best_id].frames_absent = 0
                    claimed.add(best_id)
                    tid = best_id
                else:
                    tid = self._next_id
                    self._next_id += 1
                    self._records[tid] = _EnemyRecord(tid, det.cx, det.cy)
                    claimed.add(tid)

            rec = self._records[tid]
            rec.frames_visible += 1
            rec.last_cx, rec.last_cy = det.cx, det.cy
            rec.score_window.append(_overlap_score(det, cx, cy, self._edge_margin))
            assigned.append((det, tid))

        # Age absent records (those not in this frame's detections)
        for tid, rec in self._records.items():
            if tid not in claimed:
                rec.frames_absent += 1

        return assigned

    # ── Acquisition (UNLOCKED → LOCKED) ──────────────────────────

    def _try_acquire(
        self,
        assigned: List[Tuple[Detection, int]],
        cx: float,
        cy: float,
    ) -> Optional[TargetInfo]:
        """
        Phase 1 — bbox-contains (immediate, unconditional):
          If any enemy bbox contains the crosshair, lock immediately.
          No temporal threshold required: if the crosshair is on the body,
          that IS the target by definition.
          If multiple bboxes contain the crosshair, pick the one whose centre
          is closest (most centred in the aim).

        Phase 2 — temporal proximity (threshold):
          If no bbox-contains candidate, look for sustained near-crosshair
          presence.  Lock the enemy with the highest temporal_score, provided
          it exceeds acq_threshold (default 0.25 = crosshair within edge_margin
          for ~5 of the last 20 frames).

        Phase 3 — no candidate → return None.
          DevoManiac is not actively aiming at anyone.  No box drawn.
        """
        # ── Phase 1: bbox-contains ────────────────────────────────
        contains = [
            (det, tid) for det, tid in assigned
            if det.x1 <= cx <= det.x2 and det.y1 <= cy <= det.y2
        ]
        if contains:
            best_det, best_tid = min(
                contains,
                key=lambda pair: math.hypot(pair[0].cx - cx, pair[0].cy - cy),
            )
            return self._commit_lock(best_det, best_tid, cx, cy)

        # ── Phase 2: temporal threshold ───────────────────────────
        best_det, best_tid = None, None
        best_ts: float = self._acq_threshold   # gate

        for det, tid in assigned:
            ts = self._records[tid].temporal_score
            if ts > best_ts:
                best_ts, best_det, best_tid = ts, det, tid

        if best_det is not None:
            return self._commit_lock(best_det, best_tid, cx, cy)

        return None

    def _commit_lock(
        self,
        det: Detection,
        tid: int,
        cx: float,
        cy: float,
    ) -> TargetInfo:
        self._locked_id  = tid
        self._locked_box = _bbox(det)
        self._lost_count = 0
        self._state      = SelectState.LOCKED
        return self._build_info(det, tid)

    # ── Lock maintenance (LOCKED → LOCKED) ───────────────────────

    def _maintain_lock(
        self,
        assigned: List[Tuple[Detection, int]],
        cx: float,
        cy: float,
    ) -> Optional[TargetInfo]:
        result = self._find_locked(assigned)

        if result is None:
            self._lost_count += 1
            if self._lost_count >= self._max_lost:
                self.reset()
            return None

        det, tid = result
        self._locked_box = _bbox(det)
        self._locked_id  = tid
        self._lost_count = 0
        return self._build_info(det, tid)

    def _find_locked(
        self, assigned: List[Tuple[Detection, int]]
    ) -> Optional[Tuple[Detection, int]]:
        """
        Three-phase search for the locked target in the current frame.

        Phase 1 — Stable ID match (fast path):
          Works when ByteTrack is active or our NN tracker is consistent.

        Phase 2 — IoU match (handles ID reassignment after occlusion):
          Finds the detection with the highest bbox overlap against the
          last known locked position.

        Phase 3 — Centre proximity fallback (handles fast movers):
          If IoU dropped below threshold (target moved quickly), find the
          detection whose centre is within 200 px of the last known centre.
          200 px is generous but bounded; won't snap to a completely
          different enemy on the other side of the screen.
        """
        if self._locked_id is None or self._locked_box is None:
            return None

        # Phase 1: stable ID
        for det, tid in assigned:
            if tid == self._locked_id:
                return det, tid

        # Phase 2: IoU
        best_iou: float              = self._iou_thresh
        best_det: Optional[Detection] = None
        best_tid: Optional[int]       = None
        for det, tid in assigned:
            iou = _iou(_bbox(det), self._locked_box)
            if iou > best_iou:
                best_iou, best_det, best_tid = iou, det, tid
        if best_det is not None:
            return best_det, best_tid

        # Phase 3: centre proximity
        lx1, ly1, lx2, ly2 = self._locked_box
        lcx = (lx1 + lx2) / 2.0
        lcy = (ly1 + ly2) / 2.0
        best_d: float               = 200.0
        best_det, best_tid = None, None
        for det, tid in assigned:
            d = math.hypot(det.cx - lcx, det.cy - lcy)
            if d < best_d:
                best_d, best_det, best_tid = d, det, tid
        if best_det is not None:
            return best_det, best_tid

        return None

    # ── Info builder ──────────────────────────────────────────────

    def _build_info(self, det: Detection, tid: int) -> TargetInfo:
        bh    = det.y2 - det.y1
        aim_y = det.y1 + bh * self._aim_ratio
        ts    = self._records[tid].temporal_score if tid in self._records else 0.0
        return TargetInfo(
            bbox       = _bbox(det),
            center     = (det.cx, det.cy),
            aim_point  = (det.cx, aim_y),
            confidence = det.confidence,
            track_id   = tid,
            state      = self._state,
            score      = ts,       # temporal score shown in debug overlay
        )


# ─────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────

def _bbox(det: Detection) -> Tuple[int, int, int, int]:
    return (int(det.x1), int(det.y1), int(det.x2), int(det.y2))


def _iou(
    a: Tuple[int, int, int, int],
    b: Optional[Tuple[int, int, int, int]],
) -> float:
    if b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)
