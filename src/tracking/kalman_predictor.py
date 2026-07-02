"""
2-D Kalman filter for predicting a target's position when it temporarily
disappears from the detection stream (occlusion, smoke, edge of screen).

State vector: [x, y, vx, vy]
Measurement:  [x, y]
Model: constant-velocity with process noise.

Designed to integrate with the target-lock state machine: the predictor is
initialised when an enemy is confirmed, updated each frame the enemy is
visible, and queried for a predicted position when the enemy is lost.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class KalmanPredictor:
    """Lightweight constant-velocity Kalman filter (4-state, 2-measurement)."""

    def __init__(self, dt: float = 1.0 / 60.0):
        self._dt = dt
        self._initialised = False

        # State vector [x, y, vx, vy]
        self._x = np.zeros((4, 1), dtype=np.float64)
        # Error covariance
        self._P = np.eye(4, dtype=np.float64) * 500.0

        # State transition (constant velocity)
        self._F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Measurement matrix (observe position only)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise — tuned for fast-moving game characters
        q_pos, q_vel = 2.0, 20.0
        self._Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

        # Measurement noise
        r = 15.0
        self._R = np.diag([r, r]).astype(np.float64)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def init(self, x: float, y: float) -> None:
        """Initialise or re-initialise the filter at position (x, y)."""
        self._x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
        self._initialised = True

    def update(self, x: float, y: float) -> None:
        """Feed a new observed position into the filter (predict + correct)."""
        if not self._initialised:
            self.init(x, y)
            return

        # Predict
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # Update (Kalman gain)
        z = np.array([[x], [y]], dtype=np.float64)
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - self._H @ self._x)
        self._P = (np.eye(4) - K @ self._H) @ self._P

    def predict_next(self) -> Tuple[float, float]:
        """
        Predict the next position without incorporating a measurement.

        Call this each frame when the target is not visible.
        """
        if not self._initialised:
            raise RuntimeError("KalmanPredictor not initialised — call init() first.")
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return float(self._x[0, 0]), float(self._x[1, 0])

    @property
    def position(self) -> Tuple[float, float]:
        return float(self._x[0, 0]), float(self._x[1, 0])

    @property
    def velocity(self) -> Tuple[float, float]:
        return float(self._x[2, 0]), float(self._x[3, 0])

    @property
    def is_initialised(self) -> bool:
        return self._initialised

    def reset(self) -> None:
        self._initialised = False
        self._x = np.zeros((4, 1), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 500.0
