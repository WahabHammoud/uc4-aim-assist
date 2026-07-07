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

Requires: pip install pygetwindow
"""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Optional, Tuple

from src.utils.logger import get_logger

log = get_logger(__name__)

# Colour used as the transparent key.  Must not appear in the red box outline.
_TRANSPARENT = "black"
_BOX_COLOUR  = "red"
_BOX_WIDTH   = 3
_TICK_MS     = 50    # overlay refresh interval (ms) — 20 fps is plenty


class OverlayWindow:
    """
    Transparent tkinter overlay positioned over the Chiaki window.

    Usage
    -----
    overlay = OverlayWindow(window_title="Chiaki")
    overlay.start()                          # spawns overlay thread
    overlay.update_box((x1,y1,x2,y2), True) # call from pipeline each frame
    overlay.stop()                           # on shutdown
    """

    def __init__(self, window_title: str = "Chiaki"):
        self._window_title = window_title
        self._lock    = threading.Lock()
        self._box:     Optional[Tuple[int, int, int, int]] = None
        self._visible: bool = False
        self._running: bool = False
        self._thread:  Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API — safe to call from any thread
    # ------------------------------------------------------------------

    def update_box(
        self,
        box:     Optional[Tuple[int, int, int, int]],
        visible: bool,
    ) -> None:
        """Push the latest lock state.  Called from the pipeline thread."""
        with self._lock:
            self._box     = box
            self._visible = visible

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
            wins = gw.getWindowsWithTitle(self._window_title)
            if not wins:
                return None
            w = wins[0]
            return w.left, w.top, w.width, w.height
        except Exception as exc:
            log.warning("pygetwindow: %s", exc)
            return None

    def _run(self) -> None:
        geo = self._find_chiaki()
        if geo is None:
            log.error(
                "Overlay: Chiaki window '%s' not found — "
                "open Chiaki before starting the overlay.",
                self._window_title,
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

        def tick() -> None:
            if not self._running:
                root.quit()
                return

            # Reposition overlay if the Chiaki window moved / resized
            geo = self._find_chiaki()
            if geo:
                l, t, w, h = geo
                root.geometry(f"{w}x{h}+{l}+{t}")

            canvas.delete("all")

            with self._lock:
                box     = self._box
                visible = self._visible

            if visible and box is not None:
                x1, y1, x2, y2 = box
                canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=_BOX_COLOUR,
                    width=_BOX_WIDTH,
                )

            root.after(_TICK_MS, tick)

        root.after(_TICK_MS, tick)
        root.mainloop()
        log.info("Overlay window closed.")
