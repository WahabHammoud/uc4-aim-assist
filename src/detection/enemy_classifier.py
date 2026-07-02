"""
Enemy vs. teammate classifier using HSV colour analysis.

Uncharted 4 Multiplayer places a floating coloured marker above each player:
  - RED  marker  → enemy   (must track and aim)
  - BLUE marker  → teammate / friendly sidekick (must IGNORE)

Algorithm for each detected bounding box:
  1. Compute a "marker search region" just above the bounding box top.
  2. Convert that crop to HSV.
  3. Count red pixels and blue pixels using two-range thresholding.
  4. Decide:
       red_count  ≥ min_pixels  → enemy
       blue_count ≥ min_pixels  → teammate  → rejected
       neither                  → apply motion filter (frame-diff)
                                  high motion  → tentative enemy
                                  no motion    → static object → rejected

The previous frame is stored internally so the motion filter can run
without external bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ClassifierConfig:
    red_hsv_lower1: list = None
    red_hsv_upper1: list = None
    red_hsv_lower2: list = None
    red_hsv_upper2: list = None
    blue_hsv_lower: list = None
    blue_hsv_upper: list = None
    marker_search_above_ratio: float = 0.35
    marker_search_width_ratio: float = 0.50
    min_marker_pixels: int = 40
    motion_min_pixels: int = 120

    def __post_init__(self):
        self.red_hsv_lower1 = self.red_hsv_lower1 or [0, 120, 100]
        self.red_hsv_upper1 = self.red_hsv_upper1 or [10, 255, 255]
        self.red_hsv_lower2 = self.red_hsv_lower2 or [165, 120, 100]
        self.red_hsv_upper2 = self.red_hsv_upper2 or [180, 255, 255]
        self.blue_hsv_lower = self.blue_hsv_lower or [95, 120, 80]
        self.blue_hsv_upper = self.blue_hsv_upper or [130, 255, 255]


class EnemyClassifier:
    """Classify each Detection as enemy / teammate / unknown via colour analysis."""

    def __init__(self, config: dict):
        self._cfg = ClassifierConfig(**{
            k: v for k, v in config.items()
            if k in ClassifierConfig.__dataclass_fields__
        })
        self._prev_frame: Optional[np.ndarray] = None

        # Pre-compute HSV arrays once
        self._r1_lo = np.array(self._cfg.red_hsv_lower1, np.uint8)
        self._r1_hi = np.array(self._cfg.red_hsv_upper1, np.uint8)
        self._r2_lo = np.array(self._cfg.red_hsv_lower2, np.uint8)
        self._r2_hi = np.array(self._cfg.red_hsv_upper2, np.uint8)
        self._b_lo  = np.array(self._cfg.blue_hsv_lower, np.uint8)
        self._b_hi  = np.array(self._cfg.blue_hsv_upper, np.uint8)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def classify(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        Annotate each Detection with is_enemy=True/False.

        Returns a new list — the input list is not mutated.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_frame, w_frame = frame.shape[:2]
        out: List[Detection] = []

        for det in detections:
            det_copy = Detection(
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                confidence=det.confidence, class_id=det.class_id,
                is_enemy=False, track_id=det.track_id,
            )
            marker_crop_hsv = self._extract_marker_region(hsv, det, h_frame, w_frame)

            if marker_crop_hsv is not None and marker_crop_hsv.size > 0:
                is_red, is_blue = self._analyse_marker(marker_crop_hsv)
                if is_blue:
                    det_copy.is_enemy = False   # Teammate — drop
                elif is_red:
                    det_copy.is_enemy = True    # Confirmed enemy
                else:
                    # Marker unclear — use motion as tie-breaker
                    det_copy.is_enemy = self._motion_check(frame, det)
            else:
                det_copy.is_enemy = self._motion_check(frame, det)

            out.append(det_copy)

        self._prev_frame = frame.copy()
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_marker_region(
        self,
        hsv_frame: np.ndarray,
        det: Detection,
        h_frame: int,
        w_frame: int,
    ) -> Optional[np.ndarray]:
        """Crop the HSV region just above the bounding box where the marker floats."""
        bbox_h = det.y2 - det.y1
        search_h = int(bbox_h * self._cfg.marker_search_above_ratio)
        if search_h < 4:
            return None

        cx = det.cx
        half_w = int(det.width * self._cfg.marker_search_width_ratio / 2)

        x1 = max(0, int(cx - half_w))
        x2 = min(w_frame - 1, int(cx + half_w))
        y2 = max(0, int(det.y1))
        y1 = max(0, y2 - search_h)

        if x2 <= x1 or y2 <= y1:
            return None

        return hsv_frame[y1:y2, x1:x2]

    def _analyse_marker(self, crop_hsv: np.ndarray) -> Tuple[bool, bool]:
        """Return (is_red, is_blue) based on pixel counts in the HSV crop."""
        red_mask = (
            cv2.inRange(crop_hsv, self._r1_lo, self._r1_hi) |
            cv2.inRange(crop_hsv, self._r2_lo, self._r2_hi)
        )
        blue_mask = cv2.inRange(crop_hsv, self._b_lo, self._b_hi)

        min_px = self._cfg.min_marker_pixels
        red_count  = int(np.count_nonzero(red_mask))
        blue_count = int(np.count_nonzero(blue_mask))

        return red_count >= min_px, blue_count >= min_px

    def _motion_check(self, frame: np.ndarray, det: Detection) -> bool:
        """Return True if the bounding-box region shows significant motion."""
        if self._prev_frame is None:
            return False

        x1 = max(0, int(det.x1))
        y1 = max(0, int(det.y1))
        x2 = min(frame.shape[1], int(det.x2))
        y2 = min(frame.shape[0], int(det.y2))

        if x2 <= x1 or y2 <= y1:
            return False

        curr_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(self._prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(curr_gray, prev_gray)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        changed = int(np.count_nonzero(thresh))

        return changed >= self._cfg.motion_min_pixels
