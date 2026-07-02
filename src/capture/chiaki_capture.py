"""
Chiaki screen capture module.

Finds the Chiaki window on the Windows desktop and captures frames from it
at the configured FPS, feeding them into a thread-safe queue for the
inference pipeline to consume without blocking.
"""

import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import win32gui
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


class ChiakiCapture:
    """
    Continuously captures frames from the Chiaki streaming window.

    Runs a background thread that grabs frames as fast as possible and puts
    them into a bounded queue.  The inference pipeline pops from this queue —
    if it falls behind, older frames are dropped so the pipeline always works
    on the latest frame.
    """

    QUEUE_MAXSIZE = 2   # Never accumulate stale frames

    def __init__(self, config: dict):
        self._cfg = config
        self._queue: Queue = Queue(maxsize=self.QUEUE_MAXSIZE)
        self._stop_event = threading.Event()
        self._region: Optional[CaptureRegion] = None
        self._thread: Optional[threading.Thread] = None
        self._scale = config.get("scale_factor", 1.0)
        self._target_fps = config.get("target_fps", 60)
        self._frame_interval = 1.0 / self._target_fps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Locate the Chiaki window and start the capture thread."""
        self._region = self._find_chiaki_window()
        if self._region is None:
            log.warning("Chiaki window not found — falling back to full-screen capture.")
            self._region = CaptureRegion(
                left=0, top=0,
                width=self._cfg.get("capture_width", 1920),
                height=self._cfg.get("capture_height", 1080),
            )
        log.info(
            "Capture region: %dx%d at (%d, %d)",
            self._region.width, self._region.height,
            self._region.left, self._region.top,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """
        Pop the latest frame from the queue.

        Returns None if no frame is available within *timeout* seconds.
        Always returns the freshest available frame by draining stale ones.
        """
        frame = None
        try:
            while True:
                frame = self._queue.get_nowait()
        except Empty:
            pass
        if frame is not None:
            return frame
        # Nothing in queue — block briefly for the very first frame
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def region(self) -> Optional[CaptureRegion]:
        return self._region

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_chiaki_window(self) -> Optional[CaptureRegion]:
        """Use Win32 API to find the Chiaki window bounding box."""
        if not WIN32_AVAILABLE:
            log.debug("win32gui not available; skipping window detection.")
            return self._region_from_config()

        title_pattern = self._cfg.get("window_title", "Chiaki").lower()
        handles = []

        def _enum(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd).lower()
                if title_pattern in title:
                    handles.append(hwnd)

        win32gui.EnumWindows(_enum, None)
        if not handles:
            return None

        hwnd = handles[0]
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        return CaptureRegion(
            left=left, top=top,
            width=right - left,
            height=bottom - top,
        )

    def _region_from_config(self) -> CaptureRegion:
        return CaptureRegion(
            left=self._cfg.get("window_x") or 0,
            top=self._cfg.get("window_y") or 0,
            width=self._cfg.get("capture_width", 1920),
            height=self._cfg.get("capture_height", 1080),
        )

    def _capture_loop(self) -> None:
        if not MSS_AVAILABLE:
            log.error("mss library not installed — cannot capture screen.")
            return

        with mss.mss() as sct:
            region = self._region
            monitor = {
                "left": region.left,
                "top": region.top,
                "width": region.width,
                "height": region.height,
            }
            log.info("Capture thread started.")
            next_frame_time = time.perf_counter()

            while not self._stop_event.is_set():
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)

                next_frame_time += self._frame_interval

                # Capture
                shot = sct.grab(monitor)
                frame = np.frombuffer(shot.raw, dtype=np.uint8).reshape(
                    shot.height, shot.width, 4
                )
                frame = frame[:, :, :3]  # Drop alpha (BGRA → BGR)

                if self._scale != 1.0:
                    new_w = int(frame.shape[1] * self._scale)
                    new_h = int(frame.shape[0] * self._scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Non-blocking put — drop oldest if full
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                self._queue.put_nowait(frame)

        log.info("Capture thread stopped.")
