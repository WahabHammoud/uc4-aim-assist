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
        self._device_index = config.get("capture_card_index", config.get("device_index", 0))
        self._width  = config.get("capture_card_width",  config.get("width",  1920))
        self._height = config.get("capture_card_height", config.get("height", 1080))
        self._cap    = None
        self._frame  = None
        self._lock   = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        log.info("Opening capture card device %d via DirectShow…", self._device_index)
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Capture card device {self._device_index} could not be opened. "
                "Run tools/find_capture_device.py to list available devices."
            )

        # FOURCC must come first — MS2130 silently resets to 640x480 otherwise
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
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
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, 60)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            log.info("After MJPG retry: %dx%d", actual_w, actual_h)

        # Warm-up: confirm a live frame arrives within 5 s
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

        with self._lock:
            self._frame = first_frame

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        log.info("Capture card started on device index %d", self._device_index)

    def _capture_loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get_frame(self, timeout=None):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        log.info("Capture card stopped.")
