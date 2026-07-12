import cv2
import threading
import logging

log = logging.getLogger(__name__)


class CaptureCardCapture:
    def __init__(self, config):
        self._device_index = config.get("device_index", 0)
        self._width = config.get("capture_card_width", config.get("width", 1920))
        self._height = config.get("capture_card_height", config.get("height", 1080))
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, 60)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Capture card device {self._device_index} could not be opened. "
                "Run tools/find_capture_device.py to list available devices."
            )
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
