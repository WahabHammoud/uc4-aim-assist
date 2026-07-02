"""
Physical DualSense controller HID reader.

Reads the raw USB HID input report from the DualSense connected to the PC
and exposes parsed stick / trigger / button state.

USB input report layout (report ID 0x01, 64 bytes):
  byte  0 : 0x01 (report ID on USB — hidapi strips this)
  byte  1 : left stick X   (0=left, 255=right, 128=centre)
  byte  2 : left stick Y   (0=up,   255=down,  128=centre)
  byte  3 : right stick X
  byte  4 : right stick Y
  byte  5 : L2 analog      (0–255)
  byte  6 : R2 analog      (0–255)
  byte  7 : sequence number
  byte  8 : buttons[0]
              bits 0-3 : D-pad hat (0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW,8=idle)
              bit  4   : Square
              bit  5   : Cross
              bit  6   : Circle
              bit  7   : Triangle
  byte  9 : buttons[1]
              bit  0 : L1
              bit  1 : R1
              bit  2 : L2 digital
              bit  3 : R2 digital
              bit  4 : Create (Share)
              bit  5 : Options
              bit  6 : L3
              bit  7 : R3
  byte 10 : buttons[2]
              bit  0 : PS
              bit  1 : Touchpad click
              bit  2 : Mute
  ...

On Bluetooth the report ID is 0x31 and bytes are shifted; we only support
USB mode here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

_AXIS_CENTRE = 128


def _norm(raw: int) -> float:
    """Convert 0–255 byte to -1.0 … +1.0 float."""
    return (raw - _AXIS_CENTRE) / _AXIS_CENTRE


def _trigger_norm(raw: int) -> float:
    """Convert 0–255 trigger byte to 0.0 … 1.0 float."""
    return raw / 255.0


@dataclass
class ControllerState:
    """Snapshot of DualSense input at a single moment in time."""
    lx: float = 0.0          # Left stick X  (-1 left … +1 right)
    ly: float = 0.0          # Left stick Y  (-1 up   … +1 down)
    rx: float = 0.0          # Right stick X
    ry: float = 0.0          # Right stick Y
    l2: float = 0.0          # L2 trigger    (0 … 1)
    r2: float = 0.0          # R2 trigger    (0 … 1)
    l1: bool = False
    r1: bool = False
    l3: bool = False
    r3: bool = False
    cross: bool = False
    circle: bool = False
    square: bool = False
    triangle: bool = False
    ps_button: bool = False
    options: bool = False
    create: bool = False
    touchpad: bool = False
    dpad_up: bool = False
    dpad_down: bool = False
    dpad_left: bool = False
    dpad_right: bool = False
    connected: bool = False


_DPAD_MAP = {
    0: (True,  False, False, False),   # N
    1: (True,  False, False, True ),   # NE
    2: (False, False, False, True ),   # E
    3: (False, True,  False, True ),   # SE
    4: (False, True,  False, False),   # S
    5: (False, True,  True,  False),   # SW
    6: (False, False, True,  False),   # W
    7: (True,  False, True,  False),   # NW
    8: (False, False, False, False),   # Idle
}


class DualSenseReader:
    """
    Non-blocking DualSense USB HID reader.

    Spawns a background thread that continuously reads the input report and
    updates a shared ControllerState.  The main thread can call get_state()
    at any time to get the latest snapshot without blocking.
    """

    VENDOR_ID  = 0x054C
    PRODUCT_ID = 0x0CE6

    def __init__(self, config: dict):
        self._vendor_id   = config.get("vendor_id",  self.VENDOR_ID)
        self._product_id  = config.get("product_id", self.PRODUCT_ID)
        self._timeout_ms  = config.get("read_timeout_ms", 1)
        self._state       = ControllerState()
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            import hid
        except ImportError:
            log.error("hid library not installed. Run: pip install hidapi")
            return False

        try:
            self._device = hid.device()
            self._device.open(self._vendor_id, self._product_id)
            self._device.set_nonblocking(True)
            log.info(
                "DualSense connected: %s",
                self._device.get_manufacturer_string(),
            )
            with self._lock:
                self._state.connected = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._read_loop, daemon=True, name="DualSenseReader"
            )
            self._thread.start()
            return True
        except Exception as exc:
            log.error("Failed to open DualSense: %s", exc)
            return False

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
        with self._lock:
            self._state.connected = False

    def get_state(self) -> ControllerState:
        with self._lock:
            return ControllerState(**self._state.__dict__)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._state.connected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        log.debug("DualSense read loop started.")
        while not self._stop_event.is_set():
            try:
                data = self._device.read(64, timeout_ms=self._timeout_ms)
                if data:
                    self._parse(data)
            except Exception as exc:
                log.warning("DualSense read error: %s", exc)
                time.sleep(0.001)
        log.debug("DualSense read loop stopped.")

    def _parse(self, data: list) -> None:
        """Parse the 64-byte USB HID input report into ControllerState."""
        # hidapi on Windows returns the report WITHOUT the report ID prefix.
        # So index 0 here = byte 1 in the spec table above.
        if len(data) < 10:
            return

        lx = _norm(data[0])
        ly = _norm(data[1])
        rx = _norm(data[2])
        ry = _norm(data[3])
        l2 = _trigger_norm(data[4])
        r2 = _trigger_norm(data[5])

        b0 = data[7]
        b1 = data[8]
        b2 = data[9]

        dpad_idx  = b0 & 0x0F
        if dpad_idx > 8:
            dpad_idx = 8
        up, dn, lt, rt = _DPAD_MAP[dpad_idx]

        with self._lock:
            s = self._state
            s.lx, s.ly  = lx, ly
            s.rx, s.ry  = rx, ry
            s.l2, s.r2  = l2, r2
            s.square    = bool(b0 & 0x10)
            s.cross     = bool(b0 & 0x20)
            s.circle    = bool(b0 & 0x40)
            s.triangle  = bool(b0 & 0x80)
            s.l1        = bool(b1 & 0x01)
            s.r1        = bool(b1 & 0x02)
            s.create    = bool(b1 & 0x10)
            s.options   = bool(b1 & 0x20)
            s.l3        = bool(b1 & 0x40)
            s.r3        = bool(b1 & 0x80)
            s.ps_button = bool(b2 & 0x01)
            s.touchpad  = bool(b2 & 0x02)
            s.dpad_up   = up
            s.dpad_down = dn
            s.dpad_left = lt
            s.dpad_right = rt
