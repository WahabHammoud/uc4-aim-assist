"""
Tests for overlay coordinate scaling and window-management invariants.

Covers:
  - Correct linear scaling when frame_size != window_size
  - Identity transform when frame_size == window_size
  - Correct 4K → 1080p down-scale
  - No OpenCV window created by OverlayWindow (duplicate-window guard)
"""

import sys
from pathlib import Path
from typing import Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Pure coordinate-scaling helper — mirrors the logic in overlay_window.tick()
# ---------------------------------------------------------------------------

def _scale_box(
    box: Tuple[int, int, int, int],
    frame_size: Tuple[int, int],
    win_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Apply the same scaling OverlayWindow.tick() uses."""
    x1, y1, x2, y2 = box
    fw, fh = frame_size
    ww, wh = win_size
    if fw > 0 and fh > 0 and (fw != ww or fh != wh):
        x_scale = ww / fw
        y_scale = wh / fh
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
    return x1, y1, x2, y2


class TestOverlayCoordinateScaling:

    def test_identity_when_sizes_match(self):
        """No scaling applied when frame_size == win_size."""
        box = (100, 200, 400, 600)
        result = _scale_box(box, (1920, 1080), (1920, 1080))
        assert result == box

    def test_half_scale(self):
        """Frame 1920×1080 displayed in 960×540 window — coordinates halved."""
        box = (960, 540, 1200, 700)
        x1, y1, x2, y2 = _scale_box(box, (1920, 1080), (960, 540))
        assert x1 == 480
        assert y1 == 270
        assert x2 == 600
        assert y2 == 350

    def test_4k_to_1080p(self):
        """Frame captured at 3840×2160, Chiaki window displayed at 1920×1080."""
        # Box at centre of 4K frame
        box = (1920, 1080, 2040, 1200)
        x1, y1, x2, y2 = _scale_box(box, (3840, 2160), (1920, 1080))
        assert x1 == 960
        assert y1 == 540
        assert x2 == 1020
        assert y2 == 600

    def test_non_uniform_scale(self):
        """Horizontal and vertical scales differ (letterboxing scenario)."""
        # Frame 1920×1080, window 1280×800 (different aspect ratio)
        box = (192, 108, 384, 216)
        x1, y1, x2, y2 = _scale_box(box, (1920, 1080), (1280, 800))
        assert x1 == int(192 * (1280 / 1920))
        assert y1 == int(108 * (800  / 1080))
        assert x2 == int(384 * (1280 / 1920))
        assert y2 == int(216 * (800  / 1080))

    def test_scale_preserves_box_area_proportion(self):
        """After half-scale, the scaled box area should be 1/4 of the original."""
        box = (0, 0, 200, 100)
        sx1, sy1, sx2, sy2 = _scale_box(box, (1920, 1080), (960, 540))
        original_area = (box[2] - box[0]) * (box[3] - box[1])
        scaled_area   = (sx2 - sx1) * (sy2 - sy1)
        assert abs(scaled_area - original_area / 4) <= 2   # ±2 px² for int rounding

    def test_origin_box_stays_at_origin(self):
        """A box starting at (0,0) stays at (0,0) regardless of scale."""
        box = (0, 0, 100, 50)
        sx1, sy1, sx2, sy2 = _scale_box(box, (1920, 1080), (960, 540))
        assert sx1 == 0
        assert sy1 == 0


class TestNoOpenCVWindowFromOverlay:
    """Structural tests that verify OverlayWindow never calls cv2.namedWindow.

    We do this by importing the module and confirming cv2 is not imported at
    module level and that OverlayWindow._run uses only tkinter.
    """

    def test_overlay_module_does_not_import_cv2(self):
        """overlay_window.py must not import cv2 — it is tkinter-only."""
        import importlib, types
        spec = importlib.util.spec_from_file_location(
            "overlay_window",
            str(Path(__file__).resolve().parent.parent
                / "src" / "overlay" / "overlay_window.py"),
        )
        loader_source = spec.loader.get_source("overlay_window")
        assert "import cv2" not in loader_source, (
            "overlay_window.py must not import cv2 — two separate display paths "
            "would produce a duplicate diagnostic window"
        )

    def test_overlay_update_box_does_not_raise_without_window(self):
        """update_box() must be safe to call before the tkinter thread starts."""
        from src.overlay.overlay_window import OverlayWindow
        ov = OverlayWindow(window_title="TestWindow")
        # Should not raise even though _run() has never been called
        ov.update_box((10, 20, 100, 200), visible=True, frame_size=(1920, 1080))
        assert ov._box == (10, 20, 100, 200)
        assert ov._visible is True
