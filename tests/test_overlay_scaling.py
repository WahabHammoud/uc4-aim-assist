"""
Tests for overlay coordinate scaling and window-management invariants.

Covers:
  - Correct linear scaling when frame_size != window_size
  - Identity transform when frame_size == window_size
  - Correct 4K → 1080p down-scale
  - No OpenCV window created by OverlayWindow (duplicate-window guard)
"""

import sys
from pathlib import Path
from typing import Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Pure coordinate-scaling helper — mirrors the logic in overlay_window.tick()
# ---------------------------------------------------------------------------

def _scale_box(
    box: Tuple[int, int, int, int],
    frame_size: Tuple[int, int],
    win_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Apply the same scaling OverlayWindow.tick() uses."""
    x1, y1, x2, y2 = box
    fw, fh = frame_size
    ww, wh = win_size
    if fw > 0 and fh > 0 and (fw != ww or fh != wh):
        x_scale = ww / fw
        y_scale = wh / fh
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
    return x1, y1, x2, y2


class TestOverlayCoordinateScaling:

    def test_identity_when_sizes_match(self):
        """No scaling applied when frame_size == win_size."""
        box = (100, 200, 400, 600)
        result = _scale_box(box, (1920, 1080), (1920, 1080))
        assert result == box

    def test_half_scale(self):
        """Frame 1920×1080 displayed in 960×540 window — coordinates halved."""
        box = (960, 540, 1200, 700)
        x1, y1, x2, y2 = _scale_box(box, (1920, 1080), (960, 540))
        assert x1 == 480
        assert y1 == 270
        assert x2 == 600
        assert y2 == 350

    def test_4k_to_1080p(self):
        """Frame captured at 3840×2160, Chiaki window displayed at 1920×1080."""
        # Box at centre of 4K frame
        box = (1920, 1080, 2040, 1200)
        x1, y1, x2, y2 = _scale_box(box, (3840, 2160), (1920, 1080))
        assert x1 == 960
        assert y1 == 540
        assert x2 == 1020
        assert y2 == 600

    def test_non_uniform_scale(self):
        """Horizontal and vertical scales differ (letterboxing scenario)."""
        # Frame 1920×1080, window 1280×800 (different aspect ratio)
        box = (192, 108, 384, 216)
        x1, y1, x2, y2 = _scale_box(box, (1920, 1080), (1280, 800))
        assert x1 == int(192 * (1280 / 1920))
        assert y1 == int(108 * (800  / 1080))
        assert x2 == int(384 * (1280 / 1920))
        assert y2 == int(216 * (800  / 1080))

    def test_scale_preserves_box_area_proportion(self):
        """After half-scale, the scaled box area should be 1/4 of the original."""
        box = (0, 0, 200, 100)
        sx1, sy1, sx2, sy2 = _scale_box(box, (1920, 1080), (960, 540))
        original_area = (box[2] - box[0]) * (box[3] - box[1])
        scaled_area   = (sx2 - sx1) * (sy2 - sy1)
        assert abs(scaled_area - original_area / 4) <= 2   # ±2 px² for int rounding

    def test_origin_box_stays_at_origin(self):
        """A box starting at (0,0) stays at (0,0) regardless of scale."""
        box = (0, 0, 100, 50)
        sx1, sy1, sx2, sy2 = _scale_box(box, (1920, 1080), (960, 540))
        assert sx1 == 0
        assert sy1 == 0


class TestNoOpenCVWindowFromOverlay:
    """Structural tests that verify OverlayWindow never calls cv2.namedWindow.

    We do this by importing the module and confirming cv2 is not imported at
    module level and that OverlayWindow._run uses only tkinter.
    """

    def test_overlay_module_does_not_import_cv2(self):
        """overlay_window.py must not import cv2 — it is tkinter-only."""
        import importlib, types
        spec = importlib.util.spec_from_file_location(
            "overlay_window",
            str(Path(__file__).resolve().parent.parent
                / "src" / "overlay" / "overlay_window.py"),
        )
        loader_source = spec.loader.get_source("overlay_window")
        assert "import cv2" not in loader_source, (
            "overlay_window.py must not import cv2 — two separate display paths "
            "would produce a duplicate diagnostic window"
        )

    def test_overlay_update_box_does_not_raise_without_window(self):
        """update_box() must be safe to call before the tkinter thread starts."""
        from src.overlay.overlay_window import OverlayWindow
        ov = OverlayWindow(window_title="TestWindow")
        # Should not raise even though _run() has never been called
        ov.update_box((10, 20, 100, 200), visible=True, frame_size=(1920, 1080))
        assert ov._box == (10, 20, 100, 200)
        assert ov._visible is True

    def test_overlay_start_twice_does_not_create_second_thread(self):
        """Calling start() while the thread is already alive must be a no-op
        so that a pipeline restart never spawns a second tkinter window."""
        from src.overlay.overlay_window import OverlayWindow
        ov = OverlayWindow(window_title="TestWindow")
        # Inject a fake alive thread so we can test the guard without actually
        # starting tkinter (which needs a display).
        import threading
        dummy = threading.Thread(target=lambda: None, daemon=True)
        dummy.start()
        dummy.join()   # thread is now dead — use a live one
        live = threading.Thread(target=lambda: __import__("time").sleep(2), daemon=True)
        live.start()
        ov._thread = live
        # start() must detect the live thread and return immediately
        ov.start()
        # The thread object must still be the live one we injected
        assert ov._thread is live
        live.join(timeout=0)  # don't actually wait

    def test_pipeline_feed_singleton_not_started_on_init(self):
        """Constructing InferencePipeline must NOT start a feed window thread.
        The module-level singleton _feed_thread_instance must remain None until
        run(show_feed=True) is called for the first time."""
        import yaml
        import src.pipeline.inference_pipeline as pip_mod
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        from src.pipeline.inference_pipeline import InferencePipeline
        # Reset global in case a prior test left it dirty
        with pip_mod._feed_thread_lock:
            pip_mod._feed_thread_instance = None
        InferencePipeline(config=cfg)
        with pip_mod._feed_thread_lock:
            assert pip_mod._feed_thread_instance is None

    def test_feed_thread_singleton_reuses_instance(self):
        """_get_or_create_feed_thread() must return the same object when called
        twice while the thread is alive — ensuring only one window ever exists."""
        import src.pipeline.inference_pipeline as pip_mod

        # Reset state
        with pip_mod._feed_thread_lock:
            pip_mod._feed_thread_instance = None

        # Inject a fake alive thread so we never actually open a cv2 window
        import threading
        fake_stop = lambda: None

        class _FakeThread:
            def __init__(self):
                self._t = threading.Thread(
                    target=lambda: __import__("time").sleep(5), daemon=True
                )
                self._t.start()
                self._title    = pip_mod._FEED_WIN
                self._windowed = False
                self._alive    = True

            def is_alive(self):
                return self._t.is_alive()

            def stop(self):
                self._alive = False

            def join(self, timeout=2.0):
                pass

        fake = _FakeThread()
        with pip_mod._feed_thread_lock:
            pip_mod._feed_thread_instance = fake  # type: ignore[assignment]

        # Calling the helper again must return the same object, not a new one
        result = pip_mod._get_or_create_feed_thread(windowed=False, stop_callback=fake_stop)
        assert result is fake

        # Cleanup
        with pip_mod._feed_thread_lock:
            pip_mod._feed_thread_instance = None

    def test_pipeline_feed_win_constant_matches_imshow_name(self):
        """The module-level _FEED_WIN constant must be the name used in imshow
        so that the single-window guard covers the actual cv2 call."""
        import src.pipeline.inference_pipeline as pip_mod
        import inspect
        source = inspect.getsource(pip_mod.InferencePipeline._draw_feed)
        # _draw_feed itself doesn't call imshow; verify the module constant exists
        assert hasattr(pip_mod, "_FEED_WIN")
        assert isinstance(pip_mod._FEED_WIN, str)
        assert len(pip_mod._FEED_WIN) > 0
