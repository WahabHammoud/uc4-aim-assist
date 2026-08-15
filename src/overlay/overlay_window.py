"""
Transparent always-on-top overlay that draws the aim-assist red box
directly over the Chiaki stream window.

Design
------
The overlay runs in a dedicated daemon thread.  tkinter is created and
owned exclusively by that thread (all calls stay on it), so there are no
cross-thread GUI calls.  The pipeline pushes state via update_box() which
writes to a lock-protected slot; the overlay reads that slot in its own
50 ms tick.

Coordinate scaling
------------------
Detection runs on the captured frame (which may be 1920x1080 or any other
resolution).  The Chiaki window displayed on screen may be a different size
(windowed mode, DPI scaling, etc.).  update_box() therefore accepts the
frame dimensions alongside the bounding box; tick() scales the coordinates
to match the actual Chiaki window size before drawing.

Requires: pip install pygetwindow
"""

from __future__ import annotations

import time
import threading
import tkinter as tk
from typing import Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

# Colour used as the transparent key.  Must not appear in the red box outline.
_TRANSPARENT = "black"
_BOX_COLOUR  = "red"
_BOX_WIDTH   = 2
_CORNER_LEN  = 20   # arm length of each L-corner marker (px)
_CORNER_W    = 3    # line thickness of corner markers
_TICK_MS     = 50    # overlay refresh interval (ms) — 20 fps is plenty


def _draw_corners(
    canvas: tk.Canvas,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    """Draw white L-shaped corner markers at each corner of the box."""
    c, w, n = "white", _CORNER_W, _CORNER_LEN
    canvas.create_line(x1, y1, x1 + n, y1, fill=c, width=w)
    canvas.create_line(x1, y1, x1, y1 + n, fill=c, width=w)
    canvas.create_line(x2, y1, x2 - n, y1, fill=c, width=w)
    canvas.create_line(x2, y1, x2, y1 + n, fill=c, width=w)
    canvas.create_line(x1, y2, x1 + n, y2, fill=c, width=w)
    canvas.create_line(x1, y2, x1, y2 - n, fill=c, width=w)
    canvas.create_line(x2, y2, x2 - n, y2, fill=c, width=w)
    canvas.create_line(x2, y2, x2, y2 - n, fill=c, width=w)


class OverlayWindow:
    """
    Transparent tkinter overlay positioned over the Chiaki window.

    Usage
    -----
    overlay = OverlayWindow(window_title="Chiaki")
    overlay.start()                                          # spawns overlay thread
    overlay.update_box((x1,y1,x2,y2), True, (1920, 1080))  # call from pipeline each frame
    overlay.stop()                                           # on shutdown
    """

    def __init__(self, window_title: str = "Chiaki"):
        self._window_title = window_title
        self._lock       = threading.Lock()
        self._box:        Optional[Tuple[int, int, int, int]] = None
        self._visible:    bool = False
        self._frame_size: Tuple[int, int] = (1920, 1080)  # source frame dimensions
        self._running:    bool = False
        self._thread:     Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API — safe to call from any thread
    # ------------------------------------------------------------------

    def update_box(
        self,
        box:        Optional[Tuple[int, int, int, int]],
        visible:    bool,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        """
        Push the latest lock state.  Called from the pipeline thread.

        frame_size is the (width, height) of the frame the bounding box
        coordinates are relative to.  The overlay scales them to match the
        actual Chiaki window size on screen before drawing.
        """
        with self._lock:
            self._box        = box
            self._visible    = visible
            self._frame_size = frame_size

    def start(self) -> None:
        """Spawn the overlay thread.  Returns immediately."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="OverlayWindow"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the overlay thread to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Internals — all run on the overlay thread
    # ------------------------------------------------------------------

    def _find_chiaki(self) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, width, height) of the Chiaki window, or None."""
        try:
            import pygetwindow as gw

            # Try exact known titles first (fastest path)
            for title in ("chiaki-ng", "Chiaki-ng", "chiaki", "Chiaki", "PS4", "PS5"):
                wins = gw.getWindowsWithTitle(title)
                if wins:
                    w = wins[0]
                    return w.left, w.top, w.width, w.height

            # Fallback: scan every open window for 'chiaki' substring
            for w in gw.getAllWindows():
                if "chiaki" in w.title.lower():
                    log.info("Overlay: found window via fallback scan: '%s'", w.title)
                    return w.left, w.top, w.width, w.height

            return None
        except Exception as exc:
            log.warning("pygetwindow: %s", exc)
            return None

    def _run(self) -> None:
        # Retry for up to 10 s — Chiaki may still be loading when the pipeline starts
        geo = None
        t0 = time.time()
        while time.time() - t0 < 10.0 and self._running:
            geo = self._find_chiaki()
            if geo is not None:
                break
            time.sleep(0.5)

        if geo is None:
            log.error(
                "Overlay: Chiaki window not found after 10 s — overlay disabled. "
                "Open Chiaki first, or use --show-feed for capture card mode.",
            )
            return

        left, top, width, height = geo
        log.info(
            "Overlay: attached to Chiaki at (%d, %d)  size %dx%d",
            left, top, width, height,
        )

        root = tk.Tk()
        root.overrideredirect(True)                     # no title bar / border
        root.attributes("-topmost", True)               # always on top
        root.attributes("-transparentcolor", _TRANSPARENT)
        root.configure(bg=_TRANSPARENT)
        root.geometry(f"{width}x{height}+{left}+{top}")

        canvas = tk.Canvas(root, bg=_TRANSPARENT, highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Track current on-screen window size so scaling stays correct even
        # after Chiaki is resized or moved.
        win_size = [width, height]  # mutable list updated in tick()

        def tick() -> None:
            if not self._running:
                root.quit()
                return

            # Reposition overlay if the Chiaki window moved / resized
            cur_geo = self._find_chiaki()
            if cur_geo:
                l, t, w, h = cur_geo
                root.geometry(f"{w}x{h}+{l}+{t}")
                win_size[0], win_size[1] = w, h

            canvas.delete("all")

            with self._lock:
                box        = self._box
                visible    = self._visible
                frame_size = self._frame_size

            if visible and box is not None:
                x1, y1, x2, y2 = box

                # Scale from detection-frame coordinates to on-screen window coords.
                fw, fh = frame_size
                ww, wh = win_size
                if fw > 0 and fh > 0 and (fw != ww or fh != wh):
                    x_scale = ww / fw
                    y_scale = wh / fh
                    x1 = int(x1 * x_scale)
                    y1 = int(y1 * y_scale)
                    x2 = int(x2 * x_scale)
                    y2 = int(y2 * y_scale)

                canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=_BOX_COLOUR,
                    width=_BOX_WIDTH,
                )
                _draw_corners(canvas, x1, y1, x2, y2)

            root.after(_TICK_MS, tick)

        root.after(_TICK_MS, tick)
        root.mainloop()
        log.info("Overlay window closed.")
