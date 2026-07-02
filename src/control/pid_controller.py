"""
Dual-axis PID controller for aim assistance.

Separate PID instances run on X and Y independently so that tuning one
axis does not affect the other.

Design choices:
  - Integral term is disabled (ki=0) by default.  Integral wind-up during
    lock acquisition causes overshot that feels wrong to the user.  The
    derivative term provides enough damping.
  - Output is normalised to [-1, 1] (stick deflection fraction) before
    being scaled by assist_strength in the pipeline.
  - A deadzone suppresses micro-corrections that would make the stick
    vibrate when the aim is already close to the target.
  - Exponential Moving Average (EMA) smoothing rounds off jitter from
    frame-to-frame detection noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PIDParams:
    kp: float = 0.22
    ki: float = 0.00
    kd: float = 0.04
    max_output: float = 0.38


class _SingleAxisPID:
    def __init__(self, params: PIDParams):
        self._kp = params.kp
        self._ki = params.ki
        self._kd = params.kd
        self._max = params.max_output
        self._integral = 0.0
        self._prev_error = 0.0
        self._INTEGRAL_CLAMP = 50.0    # anti-windup hard limit

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output for a signed error (pixels or normalised)."""
        self._integral = max(
            -self._INTEGRAL_CLAMP,
            min(self._INTEGRAL_CLAMP, self._integral + error * dt),
        )
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        raw = self._kp * error + self._ki * self._integral + self._kd * derivative
        return max(-self._max, min(self._max, raw))

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class DualAxisPID:
    """
    X + Y PID controllers with deadzone and EMA output smoothing.

    All outputs are in the range [-1, 1].
    """

    def __init__(self, config: dict):
        pid_cfg = config.get("pid", {})
        self._x_pid = _SingleAxisPID(PIDParams(**pid_cfg.get("x", {})))
        self._y_pid = _SingleAxisPID(PIDParams(**pid_cfg.get("y", {})))

        self._deadzone = config.get("pid", {}).get("deadzone_fraction", 0.018)
        self._smoothing = config.get("pid", {}).get("output_smoothing", 0.55)

        self._smooth_x = 0.0
        self._smooth_y = 0.0

    def compute(
        self,
        aim_x: float,
        aim_y: float,
        target_x: float,
        target_y: float,
        screen_w: int,
        screen_h: int,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Compute (stick_x, stick_y) corrections.

        Parameters
        ----------
        aim_x / aim_y    : current aim point (pixels) = screen centre.
        target_x / target_y : desired aim point (pixels) = enemy chest/neck.
        screen_w / screen_h : frame dimensions (for deadzone normalisation).
        dt               : elapsed time since last frame (seconds).

        Returns
        -------
        (stick_x, stick_y) normalised to [-1, 1].
        """
        err_x = target_x - aim_x          # positive → target is right of centre
        err_y = target_y - aim_y          # positive → target is below centre

        # Normalise errors to screen fraction for deadzone check
        err_x_n = err_x / max(screen_w, 1)
        err_y_n = err_y / max(screen_h, 1)

        # Deadzone: suppress micro-corrections when aim is already close.
        # Must zero err_x_n / err_y_n (the values passed to the PID) — not
        # err_x / err_y which are pixel values that go unused afterwards.
        if abs(err_x_n) < self._deadzone:
            err_x_n = 0.0
        if abs(err_y_n) < self._deadzone:
            err_y_n = 0.0

        # Run PIDs on normalised error so gain units are screen-independent
        raw_x = self._x_pid.compute(err_x_n, dt)
        raw_y = self._y_pid.compute(err_y_n, dt)

        # EMA smoothing
        alpha = 1.0 - self._smoothing   # higher smoothing → smaller alpha
        self._smooth_x = alpha * raw_x + (1.0 - alpha) * self._smooth_x
        self._smooth_y = alpha * raw_y + (1.0 - alpha) * self._smooth_y

        return self._smooth_x, self._smooth_y

    def reset(self) -> None:
        self._x_pid.reset()
        self._y_pid.reset()
        self._smooth_x = 0.0
        self._smooth_y = 0.0
