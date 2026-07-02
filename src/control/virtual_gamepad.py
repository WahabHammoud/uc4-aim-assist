"""
Virtual DS4 / Xbox360 gamepad output via ViGEm Bus Driver.

Architecture:
  Physical DualSense → DualSenseReader  →┐
                                          ├─ VirtualGamepad.send() → Chiaki
  AI aim assist        (PID output)      →┘

The virtual gamepad is what Chiaki sees and forwards to the PS5.
Configure Chiaki to use the virtual controller (look in Chiaki → Settings →
Gamepad and select "vGamepad" or "DS4 emulated").

Prerequisites (Windows):
  1. ViGEm Bus Driver: https://github.com/nefarius/ViGEmBus/releases
  2. pip install vgamepad

The send() method takes the physical controller state and the AI
X/Y correction offsets (normalised -1…1) and blends them so:
  final_rx = clamp(physical_rx + assist_strength * correction_x, -1, 1)
  final_ry = clamp(physical_ry + assist_strength * correction_y, -1, 1)
"""

from __future__ import annotations

from typing import Optional

from src.control.dualsense_reader import ControllerState
from src.utils.logger import get_logger

log = get_logger(__name__)


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class VirtualGamepad:
    """
    Wraps vgamepad to provide a DS4 or Xbox360 virtual controller.

    All inputs are normalised floats:
      sticks:   -1.0 … +1.0
      triggers:  0.0 … 1.0
    """

    def __init__(self, config: dict):
        self._gamepad_type   = config.get("virtual_gamepad_type", "ds4").lower()
        self._assist_strength = config.get("assist_strength", 0.38)  # from root config
        self._gamepad = None
        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self._gamepad_type == "none":
            log.info("Virtual gamepad disabled (type='none') — no ViGEm connection.")
            self._connected = True
            return True

        try:
            import vgamepad as vg
        except ImportError:
            log.error(
                "vgamepad not installed. Run: pip install vgamepad\n"
                "Also requires ViGEm Bus Driver: "
                "https://github.com/nefarius/ViGEmBus/releases"
            )
            return False

        try:
            if self._gamepad_type == "xbox360":
                self._gamepad = vg.VX360Gamepad()
            else:
                self._gamepad = vg.VDS4Gamepad()
            log.info("Virtual %s gamepad created.", self._gamepad_type.upper())
            self._connected = True
            return True
        except Exception as exc:
            log.error("Failed to create virtual gamepad: %s", exc)
            return False

    def disconnect(self) -> None:
        self._gamepad = None
        self._connected = False

    # ------------------------------------------------------------------
    # Main send — call once per frame
    # ------------------------------------------------------------------

    def send(
        self,
        physical: ControllerState,
        correction_x: float = 0.0,
        correction_y: float = 0.0,
    ) -> None:
        """
        Blend physical input with AI correction and send to virtual gamepad.

        correction_x / correction_y are the PID outputs in [-1, 1].
        They are scaled by assist_strength before adding.
        """
        if not self._connected or self._gamepad is None:
            return

        assist_x = self._assist_strength * correction_x
        assist_y = self._assist_strength * correction_y

        final_rx = _clamp(physical.rx + assist_x)
        final_ry = _clamp(physical.ry + assist_y)

        try:
            if self._gamepad_type == "xbox360":
                self._send_xbox(physical, final_rx, final_ry)
            else:
                self._send_ds4(physical, final_rx, final_ry)
        except Exception as exc:
            log.warning("Virtual gamepad send error: %s", exc)

    # ------------------------------------------------------------------
    # Gamepad-specific senders
    # ------------------------------------------------------------------

    def _send_ds4(self, s: ControllerState, rx: float, ry: float) -> None:
        import vgamepad as vg
        g = self._gamepad

        # Sticks
        g.left_joystick_float(x_value_float=s.lx, y_value_float=-s.ly)   # y-inverted
        g.right_joystick_float(x_value_float=rx,   y_value_float=-ry)

        # Triggers
        g.left_trigger_float(value_float=s.l2)
        g.right_trigger_float(value_float=s.r2)

        # Buttons — DS4 button constants
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_CROSS,    s.cross)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE,   s.circle)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_SQUARE,   s.square)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE, s.triangle)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_SHOULDER_LEFT,  s.l1)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_SHOULDER_RIGHT, s.r1)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_LEFT,   s.l2 > 0.1)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_TRIGGER_RIGHT,  s.r2 > 0.1)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT,  s.l3)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_THUMB_RIGHT, s.r3)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_OPTIONS, s.options)
        self._set_button_ds4(g, vg.DS4_BUTTONS.DS4_BUTTON_SHARE,   s.create)

        # D-Pad
        dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NONE
        if s.dpad_up and s.dpad_right:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHEAST
        elif s.dpad_up and s.dpad_left:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTHWEST
        elif s.dpad_down and s.dpad_right:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHEAST
        elif s.dpad_down and s.dpad_left:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTHWEST
        elif s.dpad_up:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_NORTH
        elif s.dpad_down:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_SOUTH
        elif s.dpad_left:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_WEST
        elif s.dpad_right:
            dpad = vg.DS4_DPAD_DIRECTIONS.DS4_BUTTON_DPAD_EAST

        g.directional_pad(direction=dpad)
        g.update()

    def _send_xbox(self, s: ControllerState, rx: float, ry: float) -> None:
        import vgamepad as vg
        g = self._gamepad

        g.left_joystick_float(x_value_float=s.lx, y_value_float=-s.ly)
        g.right_joystick_float(x_value_float=rx,   y_value_float=-ry)
        g.left_trigger_float(value_float=s.l2)
        g.right_trigger_float(value_float=s.r2)

        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_A,           s.cross)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_B,           s.circle)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_X,           s.square)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,           s.triangle)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,  s.l1)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER, s.r1)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,  s.l3)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB, s.r3)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_START,       s.options)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,        s.create)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,     s.dpad_up)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,   s.dpad_down)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,   s.dpad_left)
        self._set_button_xbox(g, vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,  s.dpad_right)
        g.update()

    @staticmethod
    def _set_button_ds4(gamepad, btn, pressed: bool) -> None:
        if pressed:
            gamepad.press_button(button=btn)
        else:
            gamepad.release_button(button=btn)

    @staticmethod
    def _set_button_xbox(gamepad, btn, pressed: bool) -> None:
        if pressed:
            gamepad.press_button(button=btn)
        else:
            gamepad.release_button(button=btn)

    @property
    def is_connected(self) -> bool:
        return self._connected
