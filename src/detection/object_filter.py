"""
Geometric and spatial filter that rejects false-positive detections.

Rejected candidates:
  • Detections inside HUD exclusion zones (minimap, health bar, ammo counter).
  • Bounding boxes that are too small (noise) or too large (background geometry).
  • Bounding boxes with an aspect ratio that can't belong to a standing / crouching
    / jumping person.
  • Any detection classified as is_enemy=False by EnemyClassifier.
  • (Fix A) Detections whose bbox CENTRE falls in the lower-centre self-player
    exclusion zone (configurable SELF_EXCL_* fractions).
  • (Fix B) Detections whose bbox BOTTOM edge reaches or exceeds
    self_excl_bottom_max fraction of frame height (partially off-screen body).

Spectator-mode exception (Fix A/B):
  If every candidate that passed the HUD/area/aspect checks would be rejected
  by Fix A or Fix B, none of them are rejected.  This handles kill-cams and
  spectator views where there is no local player character on screen.

Returns only confirmed enemies after all filters pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)

# Type alias
Zone = Tuple[float, float, float, float]   # (x1_n, y1_n, x2_n, y2_n) normalised

# Named constants (Fix A / Fix B) — mirror the config.yaml defaults.
SELF_EXCL_X_MIN: float = 0.35
SELF_EXCL_X_MAX: float = 0.65
SELF_EXCL_Y_MIN: float = 0.75
SELF_EXCL_Y_MAX: float = 1.00
SELF_EXCL_BOTTOM_MAX: float = 0.95


@dataclass
class FilterConfig:
    hud_exclusion_zones: List[Zone] = field(default_factory=list)
    min_bbox_area_px: float = 400
    max_bbox_area_px: float = 250_000
    max_bbox_area_fraction: float = 0.0    # if > 0, overrides max_bbox_area_px (fraction of frame)
    min_aspect_ratio: float = 0.40         # height / width
    max_aspect_ratio: float = 5.00         # height / width
    # Confidence gate — applied inside ObjectFilter after area/aspect checks
    conf_threshold: float = 0.0            # 0 = disabled (detector handles confidence)
    large_box_area_fraction: float = 0.15  # boxes larger than this fraction get reduced threshold
    large_box_conf_threshold: float = 0.40 # reduced threshold for close-range (large) boxes
    # Fix A — self-player centre exclusion zone (normalised fractions)
    self_excl_x_min: float = SELF_EXCL_X_MIN
    self_excl_x_max: float = SELF_EXCL_X_MAX
    self_excl_y_min: float = SELF_EXCL_Y_MIN
    self_excl_y_max: float = SELF_EXCL_Y_MAX
    # Fix B — self-player bottom-edge gate
    self_excl_bottom_max: float = SELF_EXCL_BOTTOM_MAX
    # Low-res overrides (frame_height < 720): player appears higher in frame
    self_excl_x_min_lowres: float = 0.35
    self_excl_x_max_lowres: float = 0.65
    self_excl_y_min_lowres: float = 0.60
    self_excl_y_max_lowres: float = 1.00


class ObjectFilter:
    """
    Pure geometric filter applied after EnemyClassifier.

    call: filter(detections, frame_width, frame_height)
    returns: only enemy detections that pass all geometric checks.
    """

    def __init__(self, config: dict):
        self._cfg = FilterConfig(**{
            k: v for k, v in config.items()
            if k in FilterConfig.__dataclass_fields__
        })

    def filter(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:
        """Return the subset of detections that are confirmed enemies, geometrically valid."""
        frame_area = frame_width * frame_height
        passed: List[Detection] = []
        for det in detections:
            if not det.is_enemy:
                continue
            if self._in_hud_zone(det, frame_width, frame_height):
                log.debug("Rejected (HUD zone): %.0f,%.0f", det.cx, det.cy)
                continue
            if not self._area_valid(det, frame_area):
                log.debug("Rejected (area=%.0f): %.0f,%.0f", det.area, det.cx, det.cy)
                continue
            if not self._aspect_valid(det):
                log.debug("Rejected (aspect=%.2f h/w): %.0f,%.0f",
                          det.height / max(det.width, 1), det.cx, det.cy)
                continue
            if not self._conf_valid(det, frame_area):
                log.debug("Rejected (conf=%.2f): %.0f,%.0f", det.confidence, det.cx, det.cy)
                continue
            log.info(
                "DETECTION_PASSED: accepted detection at (%.0f%%, %.0f%%) conf=%.2f",
                det.cx / frame_width * 100.0,
                det.cy / frame_height * 100.0,
                det.confidence,
            )
            passed.append(det)

        return self._apply_self_exclusion(passed, frame_width, frame_height)

    # ------------------------------------------------------------------
    # Self-player exclusion (Fix A + Fix B)
    # ------------------------------------------------------------------

    def _apply_self_exclusion(
        self,
        candidates: List[Detection],
        fw: int,
        fh: int,
    ) -> List[Detection]:
        """
        Reject detections that are likely the local player's own character model.

        Spectator-mode escape hatch: if every candidate would be rejected,
        return all candidates unchanged (no local player on screen).
        """
        if not candidates:
            return candidates

        kept: List[Detection] = []
        excluded: List[Detection] = []

        for det in candidates:
            if self._in_self_zone(det, fw, fh) or self._bottom_too_low(det, fh):
                excluded.append(det)
            else:
                kept.append(det)

        # Spectator / kill-cam mode: if MULTIPLE candidates all fall in the
        # exclusion zone, the local player is probably not on screen — keep all.
        # A single detection in the zone is more likely the player's own body
        # than a spectator-view enemy, so single-detection case still rejects.
        if not kept and len(excluded) > 1:
            log.debug(
                "SELF_EXCLUSION: spectator/killcam — all %d candidates in zone, keeping all",
                len(excluded),
            )
            return candidates

        for det in excluded:
            cx_pct = det.cx / fw * 100.0
            cy_pct = det.cy / fh * 100.0
            log.info(
                "SELF_EXCLUSION: rejected detection at (%.0f%%, %.0f%%)",
                cx_pct, cy_pct,
            )

        return kept

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_hud_zone(self, det: Detection, fw: int, fh: int) -> bool:
        """True if the detection centre falls inside any HUD exclusion zone."""
        cx_n = det.cx / fw
        cy_n = det.cy / fh
        for x1_n, y1_n, x2_n, y2_n in self._cfg.hud_exclusion_zones:
            if x1_n <= cx_n <= x2_n and y1_n <= cy_n <= y2_n:
                return True
        return False

    def _in_self_zone(self, det: Detection, fw: int, fh: int) -> bool:
        """Fix A: True if bbox centre is inside the lower-centre self-exclusion zone."""
        cx_n = det.cx / fw
        cy_n = det.cy / fh
        if fh < 720:
            return (self._cfg.self_excl_x_min_lowres <= cx_n <= self._cfg.self_excl_x_max_lowres and
                    self._cfg.self_excl_y_min_lowres <= cy_n <= self._cfg.self_excl_y_max_lowres)
        return (self._cfg.self_excl_x_min <= cx_n <= self._cfg.self_excl_x_max and
                self._cfg.self_excl_y_min <= cy_n <= self._cfg.self_excl_y_max)

    def _bottom_too_low(self, det: Detection, fh: int) -> bool:
        """Fix B: True if bbox bottom edge reaches or exceeds the self-exclusion threshold."""
        return (det.y2 / fh) >= self._cfg.self_excl_bottom_max

    def _area_valid(self, det: Detection, frame_area: int) -> bool:
        max_area = (self._cfg.max_bbox_area_fraction * frame_area
                    if self._cfg.max_bbox_area_fraction > 0
                    else self._cfg.max_bbox_area_px)
        return self._cfg.min_bbox_area_px <= det.area <= max_area

    def _aspect_valid(self, det: Detection) -> bool:
        ratio = det.height / max(det.width, 1.0)   # height / width
        return self._cfg.min_aspect_ratio <= ratio <= self._cfg.max_aspect_ratio

    def _conf_valid(self, det: Detection, frame_area: int) -> bool:
        if self._cfg.conf_threshold <= 0:
            return True   # disabled — detector already applied its own threshold
        area_fraction = det.area / max(frame_area, 1)
        threshold = (self._cfg.large_box_conf_threshold
                     if area_fraction > self._cfg.large_box_area_fraction
                     else self._cfg.conf_threshold)
        return det.confidence >= threshold
