"""
Real-time inference pipeline.

Orchestrates the full frame-to-controller loop:
  1. Capture frame from Chiaki window (threaded, pre-buffered).
  2. Run YOLOv8 / TensorRT detection.
  3. Classify each detection (enemy vs teammate via HSV marker colour).
  4. Filter out HUD elements, static objects, implausible shapes.
  5. Update ByteTrack to assign persistent IDs.
  6. Feed into TargetLock state machine (L2 held → lock on nearest enemy).
  7. Compute PID corrections toward the locked aim point.
  8. Read physical DualSense state.
  9. Send blended state to virtual gamepad → Chiaki sees it.
  10. Profile every section; log summary every N frames.

All heavy computation (steps 2–7) executes on the calling thread so CUDA
context stays consistent.  Screen capture runs on a dedicated thread to
avoid GPU stalls during mss.grab().

The loop runs until stop() is called or a keyboard interrupt is raised.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.capture.chiaki_capture import ChiakiCapture
from src.control.dualsense_reader import ControllerState, DualSenseReader
from src.control.pid_controller import DualAxisPID
from src.control.virtual_gamepad import VirtualGamepad
from src.detection.detector import EnemyDetector
from src.detection.enemy_classifier import EnemyClassifier
from src.detection.object_filter import ObjectFilter
from src.detection.uc4_hud_detector import UC4HUDDetector
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.tracking.target_lock import LockState, TargetLock
from src.utils.logger import get_logger
from src.utils.profiler import FrameProfiler

log = get_logger(__name__)


def _cv2_corners(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    length: int = 20,
    thickness: int = 3,
) -> None:
    """Draw white L-shaped corner markers on an OpenCV frame."""
    white = (255, 255, 255)
    cv2.line(img, (x1, y1), (x1 + length, y1), white, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), white, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), white, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), white, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), white, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), white, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), white, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), white, thickness)


_FEED_WIN = "UC4 Aim Assist — Feed"   # single canonical window name

# ---------------------------------------------------------------------------
# Singleton feed-window thread
#
# cv2.namedWindow() and cv2.imshow() live ONLY inside this thread.  The main
# loop never touches cv2 GUI calls directly, so there is no code path that
# can silently re-create a destroyed window — the original cause of the
# multiple-windows bug on Python 3.9.
# ---------------------------------------------------------------------------

_feed_thread_lock: threading.Lock = threading.Lock()
_feed_thread_instance: Optional["_FeedWindowThread"] = None


class _FeedWindowThread:
    """
    Daemon thread that owns the cv2 feed window for the lifetime of the
    process.  Constructed at most once per pipeline run via
    _get_or_create_feed_thread(); the global singleton prevents a second
    window even if run() is called multiple times.
    """

    QUEUE_MAXSIZE = 2

    def __init__(self, title: str, windowed: bool, stop_callback) -> None:
        self._title         = title
        self._windowed      = windowed
        self._stop_callback = stop_callback   # called when user closes / ESC
        self._queue: Queue  = Queue(maxsize=self.QUEUE_MAXSIZE)
        self._alive         = True
        self._thread        = threading.Thread(
            target=self._run, daemon=True, name="FeedWindow"
        )

    def start(self) -> None:
        if self._thread.is_alive():
            log.warning(
                "_FeedWindowThread.start() called while thread is already alive — ignoring. "
                "This indicates a bug: _get_or_create_feed_thread() should have returned the "
                "existing instance instead of constructing a new one."
            )
            return
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def push_frame(self, frame: np.ndarray) -> None:
        """Non-blocking push — drops oldest frame when queue is full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        try:
            self._queue.put_nowait(frame)
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the thread to exit; it destroys the cv2 window itself."""
        self._alive = False

    def join(self, timeout: float = 2.0) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        # Destroy any leftover windows from a previous process that exited without
        # proper cleanup (e.g. killed via taskkill or an unhandled crash).
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # ONE namedWindow call, ever — no imshow anywhere outside this thread.
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        if self._windowed:
            cv2.resizeWindow(self._title, 960, 540)
            log.info("Feed window opened (960×540 windowed) — press ESC to quit.")
        else:
            cv2.setWindowProperty(
                self._title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            log.info("Feed window opened (fullscreen) — press ESC to quit.")

        last_frame: Optional[np.ndarray] = None

        while self._alive:
            try:
                last_frame = self._queue.get(timeout=0.02)
            except Empty:
                pass   # queue empty — redisplay last frame

            if last_frame is not None:
                cv2.imshow(self._title, last_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                log.info("Feed window: ESC pressed — stopping.")
                self._stop_callback()
                break

            try:
                if cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Feed window closed by user — stopping.")
                    self._stop_callback()
                    break
            except Exception:
                self._stop_callback()
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        log.info("Feed window thread exited.")


def _get_or_create_feed_thread(windowed: bool, stop_callback) -> "_FeedWindowThread":
    """Return the process-wide feed thread, creating and starting it once.

    Two independent guards prevent a second window:
    1. Singleton reference — instance exists and thread is alive.
    2. cv2.getWindowCount() — hard OS-level check: if any cv2 window is
       open (including one from a previous instance whose thread reference
       we may have lost), refuse to open another.
    """
    global _feed_thread_instance
    with _feed_thread_lock:
        # Guard 1: singleton reference
        if _feed_thread_instance is not None and _feed_thread_instance.is_alive():
            return _feed_thread_instance

        # Guard 2: absolute cv2 window count — belt-and-suspenders for Python 3.9
        try:
            existing = cv2.getWindowCount()
        except Exception:
            existing = 0
        if existing > 0:
            log.warning(
                "Feed window guard: cv2.getWindowCount()=%d — window already open, "
                "not creating another. Returning existing instance if available.",
                existing,
            )
            if _feed_thread_instance is not None:
                return _feed_thread_instance
            # Thread reference lost but window still open — wait for it to close
            # rather than spawn a duplicate.  Caller will retry next frame.
            return _feed_thread_instance  # type: ignore[return-value]

        _feed_thread_instance = _FeedWindowThread(
            title=_FEED_WIN,
            windowed=windowed,
            stop_callback=stop_callback,
        )
        _feed_thread_instance.start()
        return _feed_thread_instance


class InferencePipeline:
    """
    Top-level controller for the aim assist system.

    Instantiate, call start() to bring up all subsystems, then run() to
    enter the main loop.  Call stop() from another thread or signal handler
    to shut down cleanly.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        config: Optional[dict] = None,
    ):
        if config is not None:
            self._cfg = config
            log.info("Configuration provided directly (in-memory).")
        else:
            with open(config_path, "r") as f:
                self._cfg = yaml.safe_load(f)
            log.info("Configuration loaded from %s", config_path)

        # Subsystem references
        self._capture:      Optional[ChiakiCapture]    = None
        self._detector:     Optional[EnemyDetector]    = None
        self._classifier:   Optional[EnemyClassifier]  = None
        self._filter:       Optional[ObjectFilter]     = None
        self._hud_detector: Optional[UC4HUDDetector]   = None
        self._tracker:      Optional[ByteTrackWrapper] = None
        self._lock_sm:      Optional[TargetLock]       = None
        self._pid:          Optional[DualAxisPID]      = None
        self._ds_reader:    Optional[DualSenseReader]  = None
        self._vgamepad:     Optional[VirtualGamepad]   = None
        self._profiler    = FrameProfiler(
            log_interval_frames=self._cfg.get("performance", {})
                                         .get("profiler_log_interval_frames", 300)
        )
        self._running  = False
        self._frame_w  = self._cfg["capture"]["capture_width"]
        self._frame_h  = self._cfg["capture"]["capture_height"]
        self._screen_cx = self._frame_w / 2.0
        self._screen_cy = self._frame_h / 2.0

        # Threaded inference state (populated in start() when enabled)
        self._infer_lock   = threading.Lock()
        self._infer_event  = threading.Event()
        self._infer_frame: Optional[np.ndarray] = None
        self._infer_result: Optional[Tuple] = None   # (classified, enemies, tracked)
        self._infer_thread: Optional[threading.Thread] = None

        # Feed window is managed by the process-wide _FeedWindowThread singleton;
        # no per-instance state is needed here.

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise and warm up all subsystems."""
        log.info("Starting UC4 Aim Assist pipeline…")

        # 1. Screen capture
        capture_cfg = self._cfg["capture"]
        mode = capture_cfg.get("mode", "chiaki")
        if mode == "capture_card":
            from src.capture.capture_card import CaptureCardCapture
            self._capture = CaptureCardCapture(capture_cfg)
            log.info(
                "Capture mode: capture_card (device %d)",
                capture_cfg.get("capture_card_index", 0),
            )
        elif mode == "video_file":
            from src.capture.video_file_capture import VideoFileCapture
            self._capture = VideoFileCapture(capture_cfg)
            log.info("Capture mode: video_file (%s)", capture_cfg.get("video_path", ""))
        else:
            self._capture = ChiakiCapture(capture_cfg)
            log.info("Capture mode: chiaki (screen capture)")
        self._capture.start()
        log.info("Waiting for first frame…")
        frame = None
        for _ in range(60):
            frame = self._capture.get_frame(timeout=0.1)
            if frame is not None:
                break
        if frame is None:
            mode = self._cfg["capture"].get("mode", "chiaki")
            if mode == "capture_card":
                raise RuntimeError(
                    "No frame received from capture card after 6 s. "
                    "Run tools/find_capture_device.py to check device index."
                )
            raise RuntimeError(
                "No frame received from Chiaki capture after 6 s. "
                "Is Chiaki open and streaming?"
            )
        # Update actual frame dimensions if scale != 1.0
        self._frame_h, self._frame_w = frame.shape[:2]
        self._screen_cx = self._frame_w / 2.0
        self._screen_cy = self._frame_h / 2.0
        log.info("First frame received: %dx%d", self._frame_w, self._frame_h)

        # 2. Detection
        self._detector = EnemyDetector(self._cfg["detection"])
        self._detector.load()
        self._detector.warmup(n_iters=self._cfg.get("capture", {}).get("warmup_frames", 30))

        # 3. Classification + Filtering
        self._classifier    = EnemyClassifier(self._cfg["enemy_classification"])
        self._filter        = ObjectFilter(self._cfg["object_filter"])
        self._hud_detector  = UC4HUDDetector(self._cfg.get("hud_detector", {}))
        if self._cfg.get("hud_detector", {}).get("enabled", True):
            log.info("UC4 HUD detector enabled — supplementing YOLO with red name-tag detection.")

        # 4. Tracker
        self._tracker = ByteTrackWrapper(self._cfg["tracking"])
        self._tracker.load()

        # 5. Target lock state machine
        self._lock_sm = TargetLock(
            config=self._cfg["target_lock"],
            frame_width=self._frame_w,
            frame_height=self._frame_h,
            aim_point_ratio=self._cfg["roi"]["aim_point_ratio"],
            aim_point_x_ratio=self._cfg["roi"].get("aim_point_x_ratio", 0.50),
        )

        # 6. PID
        self._pid = DualAxisPID(self._cfg)

        # 7. Physical controller
        self._ds_reader = DualSenseReader(self._cfg["controller"])
        if not self._ds_reader.connect():
            log.warning(
                "Running in AUTO mode — box will appear automatically on detected enemies. "
                "Connect DualSense for manual L2/R2 control."
            )
        else:
            log.info(
                "DualSense connected — L2 gating active. Box appears only when L2 is pressed."
            )

        # 8. Virtual gamepad
        self._vgamepad = VirtualGamepad({
            **self._cfg["controller"],
            "assist_strength": self._cfg["pid"].get("assist_strength", 0.38),
        })
        if not self._vgamepad.connect():
            log.error(
                "Virtual gamepad failed. Install ViGEm Bus Driver and vgamepad."
            )

        log.info("All subsystems ready. Entering main loop…")

    def stop(self) -> None:
        self._running = False

    def run(self, show_debug: bool = False, overlay=None, show_feed: bool = False, windowed: bool = False, demo_mode: bool = False) -> None:
        """
        Main loop. Runs until stop() is called.

        Parameters
        ----------
        show_debug : bool
            If True, save every 10th frame as a JPEG to
            ~/Desktop/debug_frames/ with the locked box drawn on it.
            No popup window is created.
        overlay : OverlayWindow | None
            If provided, update_box() is called after every frame so the
            transparent overlay reflects the current lock state in real time.
        show_feed : bool
            If True, open a fullscreen cv2 window showing the capture card
            feed with the red box drawn directly on the frame.  Press ESC
            to quit.  Intended for use with --capture-card.
        """
        self._running = True
        prev_time = time.perf_counter()

        # Start inference worker NOW — self._running must be True before the
        # thread enters its while loop, otherwise it exits immediately.
        perf_cfg  = self._cfg.get("performance", {})
        _threaded = perf_cfg.get("threaded_inference", False)
        if _threaded:
            self._infer_thread = threading.Thread(
                target=self._inference_worker, daemon=True, name="InferenceWorker"
            )
            self._infer_thread.start()
            log.info("Threaded inference enabled — YOLO runs in background thread.")

        # Window creation is deferred to the first frame so that the capture
        # subsystem is confirmed alive before any GUI surfaces.  The
        # _feed_window_created flag prevents a second cv2.namedWindow() call
        # if run() is ever re-entered or the loop restarts.

        _debug_dir: Optional[Path] = None
        _debug_frame_count = 0
        _debug_save_count  = 0
        if show_debug:
            _debug_dir = Path.home() / "Desktop" / "debug_frames"
            _debug_dir.mkdir(parents=True, exist_ok=True)
            log.info("Debug mode: saving every 10th frame to %s", _debug_dir)

        _skip        = max(0, int(perf_cfg.get("inference_skip_frames", 0)))
        _frame_n     = 0
        _last_classified: List   = []
        _last_enemies:    List   = []
        _last_tracked:    List   = []
        _feed_thread: Optional[_FeedWindowThread] = None

        try:
            while self._running:
                self._profiler.begin_frame()

                # ---- 1. Capture ----
                with self._profiler.section("capture"):
                    frame = self._capture.get_frame(timeout=0.02)
                    if frame is None:
                        continue

                dt = time.perf_counter() - prev_time
                prev_time = time.perf_counter()
                dt = max(dt, 1e-4)

                # ---- 2–5. Detection / classification / filter / tracking ----
                _run_infer = (_frame_n % max(1, _skip + 1) == 0)
                _frame_n  += 1

                if _threaded:
                    # Submit frame to background worker (non-blocking)
                    if _run_infer:
                        with self._infer_lock:
                            self._infer_frame = frame
                        self._infer_event.set()
                    # Read latest result; fall back to last known on first frames
                    with self._infer_lock:
                        _result = self._infer_result
                    if _result is not None:
                        classified, enemies, tracked_enemies = _result
                        _last_classified, _last_enemies, _last_tracked = classified, enemies, tracked_enemies
                    else:
                        classified, enemies, tracked_enemies = _last_classified, _last_enemies, _last_tracked
                elif _run_infer:
                    with self._profiler.section("detection"):
                        raw_dets = self._detector.detect(frame)
                    with self._profiler.section("classification"):
                        classified = self._classifier.classify(frame, raw_dets)
                        hud_dets = self._hud_detector.detect(frame, raw_dets) if self._hud_detector else []
                        if hud_dets:
                            classified = list(classified) + hud_dets
                    with self._profiler.section("filter"):
                        enemies = self._filter.filter(classified, self._frame_w, self._frame_h)
                    with self._profiler.section("tracking"):
                        tracked_enemies = self._tracker.update(enemies)
                    _last_classified, _last_enemies, _last_tracked = classified, enemies, tracked_enemies
                else:
                    classified, enemies, tracked_enemies = _last_classified, _last_enemies, _last_tracked

                # ---- 6. Read physical controller ----
                with self._profiler.section("controller_read"):
                    if self._ds_reader and self._ds_reader.is_connected:
                        ctrl_state = self._ds_reader.get_state()
                    else:
                        ctrl_state = ControllerState(connected=False)

                l2_held = (
                    ctrl_state.l2 >= self._cfg["controller"]["l2_activation_threshold"]
                    if ctrl_state.connected
                    else True    # no controller → always active for testing
                )
                r2_held = (
                    ctrl_state.r2 >= self._cfg["controller"].get("r2_activation_threshold", 0.30)
                    if ctrl_state.connected
                    else True    # no controller → treat as always firing for testing
                )
                if demo_mode:
                    l2_held = True
                    r2_held = True

                # ---- 7. Target lock ----
                with self._profiler.section("target_lock"):
                    aim_point, lock_state = self._lock_sm.update(
                        tracked_enemies, l2_held=l2_held, r2_held=r2_held
                    )

                # ---- 8. PID correction ----
                correction_x = 0.0
                correction_y = 0.0
                if aim_point is not None and lock_state != LockState.NO_BOX:
                    with self._profiler.section("pid"):
                        correction_x, correction_y = self._pid.compute(
                            aim_x=self._screen_cx,
                            aim_y=self._screen_cy,
                            target_x=aim_point[0],
                            target_y=aim_point[1],
                            screen_w=self._frame_w,
                            screen_h=self._frame_h,
                            dt=dt,
                        )
                else:
                    self._pid.reset()

                # ---- 9. Send to virtual gamepad ----
                with self._profiler.section("gamepad_send"):
                    if self._vgamepad and self._vgamepad.is_connected:
                        self._vgamepad.send(ctrl_state, correction_x, correction_y)

                # ---- 10. Overlay update ----
                if overlay is not None:
                    overlay.update_box(
                        self._lock_sm.locked_box if self._lock_sm else None,
                        lock_state == LockState.ENGAGED,
                        frame_size=(self._frame_w, self._frame_h),
                    )

                # ---- 11. Feed window (singleton thread — one window, ever) ----
                if show_feed:
                    if _feed_thread is None:
                        _feed_thread = _get_or_create_feed_thread(windowed, self.stop)
                    if not _feed_thread.is_alive():
                        break   # user closed / ESC — thread signalled stop
                    feed_frame = self._draw_feed(frame, lock_state)
                    _feed_thread.push_frame(feed_frame)

                # ---- 12. Debug frames (saved to disk, no popup window) ----
                if show_debug:
                    _debug_frame_count += 1
                    if _debug_frame_count % 10 == 0:
                        with self._profiler.section("debug_overlay"):
                            _debug_save_count += 1
                            debug_frame = self._draw_debug(
                                frame, classified, tracked_enemies,
                                aim_point, lock_state, correction_x, correction_y,
                            )
                            cv2.imwrite(
                                str(_debug_dir / f"frame_{_debug_save_count:03d}.jpg"),
                                debug_frame,
                            )

                self._profiler.end_frame()

                if self._profiler.should_log():
                    log.info(self._profiler.report())

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — shutting down.")
        finally:
            self._shutdown(show_debug, show_feed)

    # ------------------------------------------------------------------
    # Background inference worker (threaded_inference mode)
    # ------------------------------------------------------------------

    def _inference_worker(self) -> None:
        """
        Background thread: picks up the latest frame, runs the full
        detection→classify→filter→track pipeline, stores result for the
        main loop to read.  Runs continuously until self._running is False.
        """
        log.info("InferenceWorker thread started.")
        while self._running:
            triggered = self._infer_event.wait(timeout=0.5)
            if not triggered:
                continue
            self._infer_event.clear()

            with self._infer_lock:
                frame = self._infer_frame

            if frame is None:
                continue

            try:
                raw_dets   = self._detector.detect(frame)
                classified = self._classifier.classify(frame, raw_dets)
                hud_dets   = self._hud_detector.detect(frame, raw_dets) if self._hud_detector else []
                if hud_dets:
                    classified = list(classified) + hud_dets
                enemies    = self._filter.filter(classified, self._frame_w, self._frame_h)
                tracked    = self._tracker.update(enemies)
                with self._infer_lock:
                    self._infer_result = (classified, enemies, tracked)
            except Exception as exc:
                log.warning("InferenceWorker error: %s", exc)

        log.info("InferenceWorker thread stopped.")

    # ------------------------------------------------------------------
    # Feed and debug overlays
    # ------------------------------------------------------------------

    def _draw_feed(self, frame: np.ndarray, lock_state: LockState) -> np.ndarray:
        """Minimal overlay for --show-feed: red box + white L-corner markers when ENGAGED."""
        out = frame.copy()
        locked_box = self._lock_sm.locked_box if self._lock_sm else None
        if lock_state == LockState.ENGAGED and locked_box is not None:
            x1, y1, x2, y2 = locked_box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            _cv2_corners(out, x1, y1, x2, y2)
        return out

    def _draw_debug(
        self,
        frame: np.ndarray,
        classified,
        tracked_enemies,
        aim_point,
        lock_state: LockState,
        corr_x: float,
        corr_y: float,
    ) -> np.ndarray:
        overlay = frame.copy()
        H, W = overlay.shape[:2]

        # Single red box — only when actively ENGAGED (not HOLDING or NO_BOX).
        locked_box = self._lock_sm.locked_box if self._lock_sm else None
        if lock_state == LockState.ENGAGED and locked_box is not None:
            cv2.rectangle(overlay, (locked_box[0], locked_box[1]), (locked_box[2], locked_box[3]), (0, 0, 255), 2)

        # Aim point crosshair on the locked target
        if aim_point is not None:
            ax, ay = int(aim_point[0]), int(aim_point[1])
            cv2.drawMarker(overlay, (ax, ay), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # Screen-centre crosshair (always visible)
        cv2.drawMarker(overlay,
                       (int(self._screen_cx), int(self._screen_cy)),
                       (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)

        # Status bar
        state_str = lock_state.name
        fps_str   = f"FPS:{self._profiler.fps():.0f}"
        corr_str  = f"corr=({corr_x:+.3f}, {corr_y:+.3f})"
        hud = f"{state_str}  {fps_str}  {corr_str}"
        cv2.rectangle(overlay, (0, H - 28), (W, H), (0, 0, 0), -1)
        cv2.putText(overlay, hud, (8, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self, show_debug: bool, show_feed: bool = False) -> None:
        global _feed_thread_instance
        log.info("Shutting down pipeline…")
        with _feed_thread_lock:
            if _feed_thread_instance is not None:
                _feed_thread_instance.stop()
                _feed_thread_instance.join(timeout=2.0)
                _feed_thread_instance = None
        if self._capture:
            self._capture.stop()
        if self._ds_reader:
            self._ds_reader.disconnect()
        if self._vgamepad:
            self._vgamepad.disconnect()
        log.info("Pipeline stopped cleanly.")
