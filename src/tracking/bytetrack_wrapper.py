"""
ByteTrack wrapper using the `supervision` library.

Assigns persistent integer track IDs to enemy detections across frames.
ByteTrack is preferred over DeepSORT because:
  - No re-ID network → lower latency (~0.3 ms vs ~2–5 ms)
  - Handles short disappearances with IoU matching alone
  - Well-tested on crowded scenes with fast motion

Each call to `update()` returns the same Detection list but with
`track_id` fields populated.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)


class ByteTrackWrapper:
    """Thin wrapper around supervision.ByteTrack for enemy detections."""

    def __init__(self, config: dict):
        self._cfg = config
        self._tracker = None
        self._frame_rate = config.get("frame_rate", 60)

    def load(self) -> None:
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self._cfg.get("track_activation_threshold", 0.35),
                lost_track_buffer=self._cfg.get("lost_track_buffer", 45),
                minimum_matching_threshold=self._cfg.get("minimum_matching_threshold", 0.75),
                frame_rate=self._frame_rate,
            )
            log.info("ByteTrack initialised (frame_rate=%d).", self._frame_rate)
        except ImportError:
            log.error(
                "supervision library not installed. "
                "Run: pip install supervision"
            )
            raise

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update the tracker with this frame's detections.

        Returns the same detections with track_id populated.
        Detections without a stable track (just born) receive track_id=-1
        until ByteTrack promotes them.
        """
        if not detections:
            # Still advance internal state so track ages increment
            self._advance_empty()
            return []

        import supervision as sv

        boxes  = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
        confs  = np.array([d.confidence for d in detections], dtype=np.float32)
        clsids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(xyxy=boxes, confidence=confs, class_id=clsids)
        tracked = self._tracker.update_with_detections(sv_dets)

        # Build a map from xyxy → track_id for the returned tracks
        # supervision returns tracked detections in the same order as matched
        # input detections, so we match back by IoU.
        result: List[Detection] = []
        used: set = set()

        track_ids  = tracked.tracker_id if tracked.tracker_id is not None else []
        track_boxes = tracked.xyxy if tracked.xyxy is not None else []

        for det in detections:
            best_iou = 0.0
            best_tid: Optional[int] = None

            for i, (tbox, tid) in enumerate(zip(track_boxes, track_ids)):
                if i in used:
                    continue
                iou = _iou(det.xyxy, tbox)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = int(tid)
                    best_idx = i

            d_copy = Detection(
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                confidence=det.confidence, class_id=det.class_id,
                is_enemy=det.is_enemy, track_id=best_tid,
            )
            if best_tid is not None:
                used.add(best_idx)
            result.append(d_copy)

        return result

    def reset(self) -> None:
        if self._tracker is not None:
            self._tracker.reset()

    def _advance_empty(self) -> None:
        """Advance tracker with no detections so track ages update correctly."""
        import supervision as sv
        empty = sv.Detections.empty()
        self._tracker.update_with_detections(empty)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
