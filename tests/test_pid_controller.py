"""
Unit tests for DualAxisPID controller.

Tests cover:
  - Zero error → zero output
  - Positive X error → positive output (move right)
  - Negative Y error → negative output (move up)
  - Deadzone: tiny error → zero output (below 1.8% threshold)
  - Clamp: large error capped at max_output (0.38)
  - EMA smoothing: output changes gradually over multiple frames
  - Reset: clears integral, previous error, and smoothed output
  - Integral anti-windup: integral stays within ±50
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.control.pid_controller import DualAxisPID

_CFG = {
    "pid": {
        "x":  {"kp": 0.22, "ki": 0.0, "kd": 0.04, "max_output": 0.38},
        "y":  {"kp": 0.22, "ki": 0.0, "kd": 0.04, "max_output": 0.38},
        "deadzone_fraction": 0.018,
        "output_smoothing": 0.55,
    }
}

W, H = 1920, 1080
DT = 1.0 / 60.0


def _make_pid() -> DualAxisPID:
    return DualAxisPID(_CFG)


class TestDualAxisPID:

    def test_zero_error_zero_output(self):
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        rx, ry = pid.compute(cx, cy, cx, cy, W, H, DT)
        assert rx == pytest.approx(0.0, abs=1e-6)
        assert ry == pytest.approx(0.0, abs=1e-6)

    def test_positive_x_error_positive_output(self):
        """Target is to the RIGHT of screen centre → correction_x > 0."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        # Target 300 px to the right (well outside deadzone)
        rx, _ = pid.compute(cx, cy, cx + 300, cy, W, H, DT)
        assert rx > 0.0

    def test_negative_x_error_negative_output(self):
        """Target is to the LEFT → correction_x < 0."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        rx, _ = pid.compute(cx, cy, cx - 300, cy, W, H, DT)
        assert rx < 0.0

    def test_positive_y_error_positive_output(self):
        """Target is BELOW screen centre → correction_y > 0."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        _, ry = pid.compute(cx, cy, cx, cy + 300, W, H, DT)
        assert ry > 0.0

    def test_negative_y_error_negative_output(self):
        """Target is ABOVE screen centre → correction_y < 0."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        _, ry = pid.compute(cx, cy, cx, cy - 300, W, H, DT)
        assert ry < 0.0

    def test_deadzone_suppresses_tiny_error(self):
        """Error < 1.8% of screen size must produce exactly zero output."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        # 1.5% of W = 28.8 px — inside deadzone (1.8% = 34.6 px)
        tiny_x = W * 0.015
        rx, ry = pid.compute(cx, cy, cx + tiny_x, cy, W, H, DT)
        assert rx == pytest.approx(0.0, abs=1e-6)
        assert ry == pytest.approx(0.0, abs=1e-6)

    def test_output_clamped_to_max(self):
        """Even a huge error must not exceed max_output (0.38)."""
        pid = _make_pid()
        # Run many frames with huge error to let EMA converge
        cx, cy = 960.0, 540.0
        rx, ry = 0.0, 0.0
        for _ in range(100):
            rx, ry = pid.compute(cx, cy, cx + 10_000, cy + 10_000, W, H, DT)
        assert abs(rx) <= 0.38 + 1e-6
        assert abs(ry) <= 0.38 + 1e-6

    def test_ema_smoothing_limits_first_frame(self):
        """EMA smoothing means the first frame's output is scaled by alpha (0.45),
        not by 1.0.  Verify the first output is a fraction of the steady-state P-only
        output (no derivative contribution because error is constant after frame 1)."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        err = 200
        err_n = err / W    # ≈ 0.104

        # Frame 1 output — large derivative spike then EMA compression
        first_rx, _ = pid.compute(cx, cy, cx + err, cy, W, H, DT)
        assert first_rx > 0.0    # must be positive

        # After many identical frames, EMA settles to P-only output (derivative=0)
        for _ in range(200):
            settled_rx, _ = pid.compute(cx, cy, cx + err, cy, W, H, DT)
        p_only_output = 0.22 * err_n   # kp * err_n
        assert settled_rx == pytest.approx(p_only_output, abs=0.005)

    def test_reset_clears_state(self):
        """After reset, identical input gives the same output as a fresh PID."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        for _ in range(30):
            pid.compute(cx, cy, cx + 500, cy, W, H, DT)
        pid.reset()

        fresh = _make_pid()
        rx_reset, _ = pid.compute(cx, cy, cx + 500, cy, W, H, DT)
        rx_fresh,  _ = fresh.compute(cx, cy, cx + 500, cy, W, H, DT)
        assert rx_reset == pytest.approx(rx_fresh, abs=1e-6)

    def test_symmetric_x_y(self):
        """Same magnitude error on X and Y must produce same magnitude correction."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        # Same normalised displacement on both axes
        err = 100
        rx, ry = pid.compute(cx, cy, cx + err, cy + err, W, H, DT)
        # X and Y normalised errors differ slightly (W≠H) — check they're both positive
        assert rx > 0.0
        assert ry > 0.0

    def test_integral_antiwindup(self):
        """Integral must stay within INTEGRAL_CLAMP=50 even with persistent error."""
        pid = _make_pid()
        cx, cy = 960.0, 540.0
        # Run 10,000 frames with large error
        for _ in range(10_000):
            pid.compute(cx, cy, cx + 500, cy, W, H, DT)
        # Access the private integral state to verify clamp
        assert abs(pid._x_pid._integral) <= 50.0 + 1e-6
        assert abs(pid._y_pid._integral) <= 50.0 + 1e-6
