"""
Video file capture module — offline testing without PS5 / Chiaki.

Reads frames from a local video file using cv2.VideoCapture and loops
indefinitely when the end of the file is reached.  Implements the same
interface as ChiakiCapture and CaptureCardCapture so the pipeline needs
no changes to the detection logic.
"""

import threading
import time
from queue import Empty, Queue
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class VideoFileCapture:
    """
    Capture frames from a local video file, looping when it ends.

    Config keys (all under cfg["capture"]):
        video_path          : str   — path to the video file (required)
        target_fps          : int   — playback FPS; defaults to video native FPS
    """

    QUEUE_MAXSIZE = 2

    def __init__(self, config: dict):
        self._cfg        = config
        self._video_path = config["video_path"]
        self._queue: Queue                  = Queue(maxsize=self.QUEUE_MAXSIZE)
        self._stop_event                    = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_size: Optional[Tuple[int, int]] = None

        # target_fps resolved after cap is opened in start()
        self._target_fps: Optional[float] = config.get("target_fps", None)

    # ------------------------------------------------------------------
    # Public API (matches ChiakiCapture / CaptureCardCapture)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the video file and start the capture thread."""
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video file: {self._video_path}\n"
                "Check the path and ensure the codec is supported."
            )
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self._frame_size = (w, h)
        if self._target_fps is None:
            self._target_fps = fps

        log.info(
            "VideoFileCapture: %s  [%dx%d @ %.1f fps]",
            self._video_path, w, h, self._target_fps,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="VideoFileCaptureThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        """Return the freshest available frame, or None on timeout."""
        frame = None
        try:
            while True:
                frame = self._queue.get_nowait()
        except Empty:
            pass
        if frame is not None:
            return frame
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def frame_size(self) -> Optional[Tuple[int, int]]:
        return self._frame_size

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            log.error("VideoFileCapture: failed to open %s in capture thread", self._video_path)
            return

        frame_interval = 1.0 / max(1.0, self._target_fps)
        next_frame_time = time.perf_counter()
        log.info("VideoFileCapture thread started.")

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_interval

            ok, frame = cap.read()
            if not ok or frame is None:
                # End of file — loop back to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    log.error("VideoFileCapture: cannot read from %s", self._video_path)
                    break
                log.debug("VideoFileCapture: looped back to start.")

            # Drop oldest if queue is full so pipeline always gets the freshest frame
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put_nowait(frame)

        cap.release()
        log.info("VideoFileCapture thread stopped.")
