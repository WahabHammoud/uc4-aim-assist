"""
Unit tests for clamp_box (src/utils/geometry.py).

Tests cover:
  - Box entirely within frame: returned unchanged
  - Box extending past right edge: x2 clamped
  - Box extending past bottom edge: y2 clamped
  - Box with negative x1: x1 clamped to 0
  - Box that becomes degenerate after clamping (width < 10): returns None
  - Box entirely outside frame: returns None
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.geometry import clamp_box

W, H = 1920, 1080


class TestClampBox:

    def test_box_within_frame_unchanged(self):
        """Box well inside frame — returned unchanged."""
        result = clamp_box(100, 200, 300, 500, W, H)
        assert result == (100, 200, 300, 500)

    def test_box_right_edge_clamped(self):
        """x2 beyond frame width → clamped to frame_width - 1."""
        result = clamp_box(1800, 200, 2000, 500, W, H)
        assert result is not None
        assert result[2] == W - 1   # x2 = 1919

    def test_box_bottom_edge_clamped(self):
        """y2 beyond frame height → clamped to frame_height - 1."""
        result = clamp_box(100, 900, 300, 1200, W, H)
        assert result is not None
        assert result[3] == H - 1   # y2 = 1079

    def test_negative_x1_clamped_to_zero(self):
        """x1 below 0 → clamped to 0, rest unchanged."""
        result = clamp_box(-50, 200, 200, 500, W, H)
        assert result is not None
        assert result[0] == 0
        assert result[1] == 200
        assert result[2] == 200
        assert result[3] == 500

    def test_degenerate_width_after_clamping_returns_none(self):
        """Box that clips to width < 10 after clamping → None."""
        # x1=1915, x2=1925, after clamp x2=1919 → width=4 < 10
        result = clamp_box(1915, 200, 1925, 500, W, H)
        assert result is None

    def test_box_entirely_outside_frame_returns_none(self):
        """Box entirely to the right of frame → None."""
        result = clamp_box(2000, 200, 2200, 500, W, H)
        assert result is None

    def test_multiple_edges_clamped_simultaneously(self):
        """x1 negative AND x2 past right AND y2 past bottom — all clamped."""
        result = clamp_box(-10, -10, 2000, 2000, W, H)
        assert result == (0, 0, W - 1, H - 1)

    def test_degenerate_height_after_clamping_returns_none(self):
        """Box that clips to height < 10 → None."""
        # y1=1075, y2=1090, after clamp y2=1079 → height=4 < 10
        result = clamp_box(100, 1075, 300, 1090, W, H)
        assert result is None

    def test_exact_frame_boundary_box_valid(self):
        """Box with corners exactly at frame boundaries → valid."""
        result = clamp_box(0, 0, W - 1, H - 1, W, H)
        assert result == (0, 0, W - 1, H - 1)
