import time
import threading
import logging

import cv2

log = logging.getLogger(__name__)


class CaptureCardCapture:
    """
    UVC capture card reader using DirectShow backend (CAP_DSHOW).

    CAP_DSHOW is required on Windows — the default MSMF backend can take
    minutes to open and silently ignores resolution requests for devices
    like the KASTWAVE AvedioLink (MS2130 chip).

    FOURCC must be set BEFORE width/height/FPS or the MS2130 falls back
    to 640×480 regardless of requested resolution.
    """

    def __init__(self, config):
        # -1 = auto-detect: scan devices and pick the first that gives 1920x1080
        self._device_index = config.get("capture_card_index", config.get("device_index", -1))
        self._width  = config.get("capture_card_width",  config.get("width",  1920))
        self._height = config.get("capture_card_height", config.get("height", 1080))
        self._cap    = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._running = False
        self._thread  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        if self._device_index == -1:
            log.info("Auto-detecting capture card device (scanning 0–9 for %dx%d)…", self._width, self._height)
            self._device_index = self._auto_detect_device()
            log.info("Auto-detected capture card on device index %d", self._device_index)
        log.info("Opening capture card device %d via DirectShow…", self._device_index)
        self._cap = self._open_device()

        # Warm-up: confirm a live frame arrives within 5 s before spawning thread
        log.info("Testing capture card — waiting for first frame…")
        t0 = time.time()
        first_frame = None
        while time.time() - t0 < 5.0:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                first_frame = frame
                log.info(
                    "Capture card OK — first frame received in %.1fs",
                    time.time() - t0,
                )
                break
        else:
            raise RuntimeError(
                f"Capture card device {self._device_index} opened but "
                "no frame received in 5 seconds. "
                "Check HDMI cable is connected to PS5 and capture card input."
            )

        # Flush stale frames buffered during device open — reduces lag
        log.info("Flushing capture card buffer (10 frames)…")
        for _ in range(10):
            self._cap.read()

        with self._lock:
            self._frame = first_frame

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("Capture card started on device index %d", self._device_index)

    def get_frame(self, timeout=None):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        log.info("Capture card stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_detect_device(self) -> int:
        """Scan device indices 0–9 and return the first that delivers a 1920×1080 frame."""
        for idx in range(10):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                cap.set(cv2.CAP_PROP_FPS, 60)
                ret, frame = cap.read()
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                cap.release()
                if ret and frame is not None and actual_w == self._width:
                    log.info("Auto-detect: device %d is %dx%d — selected.", idx, self._width, self._height)
                    return idx
                log.debug("Auto-detect: device %d skipped (w=%d, ret=%s)", idx, actual_w, ret)
            except Exception as exc:
                log.debug("Auto-detect: device %d error: %s", idx, exc)
        raise RuntimeError(
            f"Auto-detect failed: no device in range 0–9 delivers {self._width}×{self._height}. "
            "Check HDMI cable and USB 3.0 connection. "
            "Run tools/find_capture_device.py for diagnostics."
        )

    def _open_device(self) -> cv2.VideoCapture:
        """Open device with DirectShow, set FOURCC+resolution, log negotiated values."""
        cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"Capture card device {self._device_index} could not be opened. "
                "Run tools/find_capture_device.py to list available devices."
            )

        # FOURCC must come first — MS2130 silently resets to 640×480 otherwise
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        log.info(
            "Capture card negotiated: %dx%d @ %.0ffps",
            actual_w, actual_h, actual_fps,
        )

        if actual_w != self._width or actual_h != self._height:
            log.warning(
                "Requested %dx%d but got %dx%d",
                self._width, self._height, actual_w, actual_h,
            )
            log.warning("Check USB port is 3.0 and FOURCC is set correctly")

        # MJPG fallback if YUY2 didn't negotiate the target resolution
        if actual_w == 640:
            log.warning("YUY2 mode failed (still 640px wide), trying MJPG fallback…")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, 60)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("After MJPG retry: %dx%d", actual_w, actual_h)

        return cap

    def _reconnect(self) -> bool:
        """Try to reopen the capture device for up to 10 seconds. Returns True on success."""
        t0 = time.time()
        attempt = 0
        while time.time() - t0 < 10.0 and self._running:
            attempt += 1
            try:
                if self._cap:
                    self._cap.release()
                self._cap = self._open_device()
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    log.info("Capture card reconnected (attempt %d).", attempt)
                    with self._lock:
                        self._frame = frame
                    return True
            except Exception as exc:
                log.debug("Reconnect attempt %d failed: %s", attempt, exc)
            time.sleep(1.0)
        log.error(
            "Capture card device %d failed to reconnect after 10 s — signal lost.",
            self._device_index,
        )
        with self._lock:
            self._frame = None
        return False

    def _capture_loop(self):
        consecutive_failures = 0
        fps_count = 0
        fps_t     = time.time()

        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                consecutive_failures = 0
                with self._lock:
                    self._frame = frame

                fps_count += 1
                elapsed = time.time() - fps_t
                if elapsed >= 5.0:
                    log.info(
                        "Capture card FPS: %.1f (target: 60)",
                        fps_count / elapsed,
                    )
                    fps_count = 0
                    fps_t     = time.time()
            else:
                consecutive_failures += 1
                # ~0.5 s of consecutive failures → assume signal lost
                if consecutive_failures >= 30:
                    log.warning(
                        "Capture card signal lost, attempting reconnect…"
                    )
                    if not self._reconnect():
                        return  # give up; get_frame() returns None until restart
                    consecutive_failures = 0
                    fps_count = 0
                    fps_t     = time.time()
