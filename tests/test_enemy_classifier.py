"""
Unit tests for EnemyClassifier.

Tests cover:
  - Red-marker detection → is_enemy=True
  - Blue-marker detection → is_enemy=False
  - Empty frame (no marker) with no motion → is_enemy=False
  - Motion fallback when no marker → is_enemy=True
  - HUD-edge bbox that clips → graceful (no crash)
  - Blank bbox (zero size) → graceful (no crash)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import Detection
from src.detection.enemy_classifier import EnemyClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "red_hsv_lower1": [0, 120, 100],
    "red_hsv_upper1": [10, 255, 255],
    "red_hsv_lower2": [165, 120, 100],
    "red_hsv_upper2": [180, 255, 255],
    "blue_hsv_lower": [95, 120, 80],
    "blue_hsv_upper": [130, 255, 255],
    "marker_search_above_ratio": 0.35,
    "marker_search_width_ratio": 0.50,
    "min_marker_pixels": 40,
    "motion_min_pixels": 120,
}

H, W = 1080, 1920


def _make_det(x1=860, y1=400, x2=960, y2=700) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.8, class_id=0)


def _frame_with_red_marker(det: Detection) -> np.ndarray:
    """Gray frame with a red HSV patch in the marker search region above det."""
    frame = np.full((H, W, 3), 100, dtype=np.uint8)   # BGR mid-gray
    # Place a solid red patch (BGR = 0, 0, 255 — hue≈0° = red in HSV)
    # Marker region: just above det.y1, horizontally centred
    cx  = int(det.cx)
    hw  = int(det.width * 0.25)
    bh  = det.y2 - det.y1
    sh  = max(4, int(bh * 0.35))
    y2  = max(0, int(det.y1))
    y1  = max(0, y2 - sh)
    x1  = max(0, cx - hw)
    x2  = min(W - 1, cx + hw)
    frame[y1:y2, x1:x2] = (0, 0, 255)    # pure red in BGR (hue=0° in HSV)
    return frame


def _frame_with_blue_marker(det: Detection) -> np.ndarray:
    """Gray frame with a blue HSV patch in the marker region."""
    frame = np.full((H, W, 3), 100, dtype=np.uint8)
    cx  = int(det.cx)
    hw  = int(det.width * 0.25)
    bh  = det.y2 - det.y1
    sh  = max(4, int(bh * 0.35))
    y2  = max(0, int(det.y1))
    y1  = max(0, y2 - sh)
    x1  = max(0, cx - hw)
    x2  = min(W - 1, cx + hw)
    frame[y1:y2, x1:x2] = (255, 0, 0)    # pure blue BGR (hue≈120° = HSV blue)
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnemyClassifier:

    def setup_method(self):
        self.clf = EnemyClassifier(_DEFAULT_CFG)

    # ---- Red marker → is_enemy=True ----

    def test_red_marker_is_enemy(self):
        det = _make_det()
        frame = _frame_with_red_marker(det)
        results = self.clf.classify(frame, [det])
        assert len(results) == 1
        assert results[0].is_enemy is True

    def test_multiple_dets_red_all_enemy(self):
        dets = [_make_det(860, 400, 960, 700), _make_det(500, 300, 600, 600)]
        frames = [_frame_with_red_marker(d) for d in dets]
        # Use the second frame with both markers placed (merge them)
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        for d in dets:
            patch = _frame_with_red_marker(d)
            frame[frame == 100] = patch[frame == 100]
        results = self.clf.classify(frame, dets)
        assert all(r.is_enemy for r in results)

    # ---- Blue marker → is_enemy=False ----

    def test_blue_marker_is_teammate(self):
        det = _make_det()
        frame = _frame_with_blue_marker(det)
        results = self.clf.classify(frame, [det])
        assert len(results) == 1
        assert results[0].is_enemy is False

    # ---- No marker, no motion → is_enemy=False ----

    def test_no_marker_no_motion_not_enemy(self):
        det = _make_det()
        gray = np.full((H, W, 3), 100, dtype=np.uint8)
        # First classify to set _prev_frame; second identical frame = no motion
        self.clf.classify(gray, [det])
        results = self.clf.classify(gray, [det])
        assert results[0].is_enemy is False

    # ---- No marker + motion → is_enemy=True ----

    def test_motion_fallback_is_enemy(self):
        det = _make_det()
        gray1 = np.full((H, W, 3), 100, dtype=np.uint8)
        gray2 = gray1.copy()
        # Add large motion in bbox region of frame 2
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        gray2[y1:y2, x1:x2] = 200   # big brightness change = motion
        self.clf.classify(gray1, [det])   # establishes _prev_frame
        results = self.clf.classify(gray2, [det])
        assert results[0].is_enemy is True

    # ---- Input/output integrity ----

    def test_output_length_matches_input(self):
        det = _make_det()
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        results = self.clf.classify(frame, [det] * 5)
        assert len(results) == 5

    def test_empty_input(self):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        results = self.clf.classify(frame, [])
        assert results == []

    def test_original_not_mutated(self):
        det = _make_det()
        det.is_enemy = False
        frame = _frame_with_red_marker(det)
        self.clf.classify(frame, [det])
        assert det.is_enemy is False    # classifier must return copies

    # ---- Edge cases ----

    def test_bbox_at_top_edge(self):
        """Bbox at y=0 — no room above for marker crop — should not crash."""
        det = _make_det(x1=800, y1=0, x2=900, y2=50)
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        results = self.clf.classify(frame, [det])
        assert len(results) == 1   # did not crash

    def test_zero_area_bbox(self):
        """Degenerate bbox (zero size) — must not crash."""
        det = _make_det(x1=500, y1=500, x2=500, y2=500)
        frame = np.full((H, W, 3), 100, dtype=np.uint8)
        results = self.clf.classify(frame, [det])
        assert len(results) == 1
