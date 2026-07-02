"""Geometry utilities for the aim assist pipeline."""
from __future__ import annotations

from typing import Optional, Tuple


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Clamp a bounding box to valid frame dimensions.

    Returns the clamped (x1, y1, x2, y2) as integers, or None if the box
    becomes degenerate (width < 10 or height < 10) after clamping.
    """
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width - 1, x2)
    y2 = min(frame_height - 1, y2)
    w = x2 - x1
    h = y2 - y1
    if w < 10 or h < 10:
        return None
    return int(x1), int(y1), int(x2), int(y2)
