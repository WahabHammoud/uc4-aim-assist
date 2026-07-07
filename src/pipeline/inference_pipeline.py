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

import time
from pathlib import Path
from typing import Optional

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
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.tracking.target_lock import LockState, TargetLock
from src.utils.logger import get_logger
from src.utils.profiler import FrameProfiler

log = get_logger(__name__)


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
        self._capture:    Optional[ChiakiCapture]    = None
        self._detector:   Optional[EnemyDetector]    = None
        self._classifier: Optional[EnemyClassifier]  = None
        self._filter:     Optional[ObjectFilter]     = None
        self._tracker:    Optional[ByteTrackWrapper] = None
        self._lock_sm:    Optional[TargetLock]       = None
        self._pid:        Optional[DualAxisPID]      = None
        self._ds_reader:  Optional[DualSenseReader]  = None
        self._vgamepad:   Optional[VirtualGamepad]   = None
        self._profiler    = FrameProfiler(
            log_interval_frames=self._cfg.get("performance", {})
                                         .get("profiler_log_interval_frames", 300)
        )
        self._running  = False
        self._frame_w  = self._cfg["capture"]["capture_width"]
        self._frame_h  = self._cfg["capture"]["capture_height"]
        self._screen_cx = self._frame_w / 2.0
        self._screen_cy = self._frame_h / 2.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise and warm up all subsystems."""
        log.info("Starting UC4 Aim Assist pipeline…")

        # 1. Screen capture
        self._capture = ChiakiCapture(self._cfg["capture"])
        self._capture.start()
        log.info("Waiting for first frame from Chiaki…")
        frame = None
        for _ in range(60):
            frame = self._capture.get_frame(timeout=0.1)
            if frame is not None:
                break
        if frame is None:
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
        self._classifier = EnemyClassifier(self._cfg["enemy_classification"])
        self._filter     = ObjectFilter(self._cfg["object_filter"])

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

    def run(self, show_debug: bool = False) -> None:
        """
        Main loop. Runs until stop() is called.

        Parameters
        ----------
        show_debug : bool
            If True, save every 10th frame as a JPEG to
            ~/Desktop/debug_frames/ with the locked box drawn on it.
            No popup window is created.
        """
        self._running = True
        prev_time = time.perf_counter()

        _debug_dir: Optional[Path] = None
        _debug_frame_count = 0
        _debug_save_count  = 0
        if show_debug:
            _debug_dir = Path.home() / "Desktop" / "debug_frames"
            _debug_dir.mkdir(parents=True, exist_ok=True)
            log.info("Debug mode: saving every 10th frame to %s", _debug_dir)

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

                # ---- 2. Detection ----
                with self._profiler.section("detection"):
                    raw_dets = self._detector.detect(frame)

                # ---- 3. Enemy classification (HSV marker) ----
                with self._profiler.section("classification"):
                    classified = self._classifier.classify(frame, raw_dets)

                # ---- 4. Geometric filter ----
                with self._profiler.section("filter"):
                    enemies = self._filter.filter(classified, self._frame_w, self._frame_h)

                # ---- 5. ByteTrack ----
                with self._profiler.section("tracking"):
                    tracked_enemies = self._tracker.update(enemies)

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

                # ---- 10. Debug frames (saved to disk, no popup window) ----
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
            self._shutdown(show_debug)

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

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

    def _shutdown(self, show_debug: bool) -> None:
        log.info("Shutting down pipeline…")
        if self._capture:
            self._capture.stop()
        if self._ds_reader:
            self._ds_reader.disconnect()
        if self._vgamepad:
            self._vgamepad.disconnect()
        log.info("Pipeline stopped cleanly.")
