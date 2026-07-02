"""
Unit tests for KalmanPredictor.

Tests cover:
  - init() sets position correctly
  - update() with consistent position → position converges
  - predict_next() without update moves position by velocity
  - Constant-velocity tracking: position follows moving target
  - reset() clears initialised state
  - predict_next() before init() raises RuntimeError
  - Velocity estimate reasonable for linearly moving target
"""

import sys
from pathlib import Path
import math

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking.kalman_predictor import KalmanPredictor


class TestKalmanPredictor:

    def test_init_sets_position(self):
        kp = KalmanPredictor()
        kp.init(100.0, 200.0)
        assert kp.is_initialised
        px, py = kp.position
        assert px == pytest.approx(100.0, abs=1.0)
        assert py == pytest.approx(200.0, abs=1.0)

    def test_predict_before_init_raises(self):
        kp = KalmanPredictor()
        with pytest.raises(RuntimeError):
            kp.predict_next()

    def test_update_single_observation(self):
        kp = KalmanPredictor()
        kp.update(500.0, 300.0)
        assert kp.is_initialised
        px, py = kp.position
        # After one update position should be near the observed value
        assert abs(px - 500.0) < 50.0
        assert abs(py - 300.0) < 50.0

    def test_predict_next_advances_position(self):
        kp = KalmanPredictor(dt=1.0 / 60.0)
        # Init at known position with implied velocity
        kp.init(960.0, 540.0)
        # Feed several frames at constant velocity (10 px/frame right)
        for i in range(30):
            kp.update(960.0 + i * 10, 540.0)
        px0, py0 = kp.position
        # One prediction should move the position right
        px1, py1 = kp.predict_next()
        assert px1 > px0   # moved right

    def test_constant_velocity_tracking(self):
        """Position estimate should follow a linearly moving target."""
        kp = KalmanPredictor(dt=1.0 / 60.0)
        vx = 5.0   # px/frame

        for i in range(60):
            kp.update(100.0 + i * vx, 400.0)

        px, py = kp.position
        # After 60 frames of linear motion the estimate should be close
        expected_x = 100.0 + 59 * vx
        assert abs(px - expected_x) < 30.0   # ±30 px tolerance
        assert abs(py - 400.0) < 20.0

    def test_velocity_estimate_reasonable(self):
        """Filter should estimate velocity in px/s (state vx satisfies x_new = x + vx*dt).
        For a target moving 8 px/frame at 60 fps: vx_px_per_s = 8 * 60 = 480."""
        dt = 1.0 / 60.0
        kp = KalmanPredictor(dt=dt)
        vx_px_per_frame = 8.0
        vx_px_per_s = vx_px_per_frame / dt   # 480 px/s
        for i in range(60):
            kp.update(100.0 + i * vx_px_per_frame, 300.0)
        vx, vy = kp.velocity
        # The Kalman state stores velocity as px/s (x_new = x + vx*dt)
        assert abs(vx - vx_px_per_s) < 50.0   # within 50 px/s of true velocity
        assert abs(vy) < 50.0

    def test_reset_clears_state(self):
        kp = KalmanPredictor()
        kp.init(500.0, 300.0)
        assert kp.is_initialised
        kp.reset()
        assert not kp.is_initialised
        with pytest.raises(RuntimeError):
            kp.predict_next()

    def test_multiple_predictions_diverge(self):
        """Without corrections, uncertainty grows with each prediction."""
        kp = KalmanPredictor()
        kp.init(960.0, 540.0)
        # Just predict, no updates
        positions = [kp.predict_next() for _ in range(20)]
        # Positions should continue in the last-known direction (zero velocity)
        # All predictions at ~same position (velocity=0 at init)
        xs = [p[0] for p in positions]
        assert max(xs) - min(xs) < 5.0   # nearly constant (no velocity at init)

    def test_update_auto_inits(self):
        """update() before init() should initialise automatically."""
        kp = KalmanPredictor()
        assert not kp.is_initialised
        kp.update(200.0, 300.0)
        assert kp.is_initialised
