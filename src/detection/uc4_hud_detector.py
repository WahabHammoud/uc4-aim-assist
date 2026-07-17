"""
UC4 enemy HUD detector — red/orange name-tag colour thresholding.

Uncharted 4 renders a red/orange name tag and health bar above every enemy's
head, even at ranges where the body silhouette is too small for YOLO to pick
up.  This detector finds those tags and estimates a body bounding box below
each one, producing Detection objects that are pre-classified as enemies and
fed into the shared ObjectFilter + ByteTrack pipeline.

Design contract
---------------
- Returns List[Detection] with is_enemy=True, confidence=0.70, class_id=0.
- Does NOT replace YOLO — output is merged into the classified list AFTER
  EnemyClassifier so that it goes through ObjectFilter (self-exclusion zone,
  HUD zone rejection, area/aspect checks) before reaching ByteTrack.
- All coordinates are clamped to frame boundaries before returning.
- Two false-positive guards beyond the shape gate (added after live testing):
    1. HUD exclusion zones: body-box centre must not fall in fixed screen
       regions known to contain red UI elements (minimap, health, ammo, timer).
    2. Person proximity: at least one raw YOLO detection must exist within
       person_proximity_px of the tag centre (below it), confirming something
       person-shaped is there even if its confidence is sub-threshold.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)

# Body-box geometry relative to the detected name-tag bounding rect.
# The name tag is roughly 1/3 the width of the character body and sits just
# above the head.  These multipliers produce a plausible full-body estimate:
#   width  = 3 x tag_width  (1 tag-width margin on each side)
#   height = 8 x tag_height (body extends downward from tag top)
_BODY_HEIGHT_MULT = 8

# Fixed confidence score for HUD-derived detections (between YOLO thresholds).
# 0.70 is intentionally above conf_threshold (0.25) so they never get dropped
# by the filter's confidence gate, but below high_conf_fast_track (0.75) so
# they go through the normal stability gate.
_HUD_CONFIDENCE = 0.70

# Fixed HUD screen regions (normalised fractions) known to contain red UI
# elements that are NOT enemy name tags.  The body-box centre must fall
# outside all of these for the detection to be kept.
_HUD_EXCLUSION_ZONES: List[Tuple[float, float, float, float]] = [
    (0.00, 0.00, 0.25, 0.25),   # top-left: minimap
    (0.00, 0.00, 0.15, 1.00),   # left edge: health / score
    (0.00, 0.85, 1.00, 1.00),   # bottom bar: ammo / weapons
    (0.35, 0.00, 0.65, 0.15),   # top centre: score / timer
]


class UC4HUDDetector:
    """
    Finds UC4 enemy name tags via HSV red thresholding and returns body-box
    Detections.  Instantiate once; call detect(frame, yolo_detections) per tick.
    """

    def __init__(self, config: dict):
        self._enabled  = config.get("enabled", True)
        self._min_area = config.get("min_area", 20)
        self._max_area = config.get("max_area", 2000)
        # UC4 name tags are wide horizontal strips (e.g. 60x10px at 1080p).
        # Require tag width >= min_tag_aspect * tag height to filter square/tall
        # UI elements like health bars, ammo counters, and dialog boxes.
        self._min_tag_aspect    = config.get("min_tag_aspect", 2.0)
        # Hard height cap: name tags are thin — anything taller is not a tag.
        self._max_tag_height_px = config.get("max_tag_height_px", 25)
        # Fixed screen region filtering
        self._use_hud_zones      = config.get("hud_exclusion_zones", True)
        # Person proximity: keep HUD detection only when a raw YOLO detection
        # exists within this many pixels below the tag centre.
        self._person_proximity_px = float(config.get("person_proximity_px", 200))

        # Red occupies two HSV lobes (0-10 deg and 170-180 deg)
        self._lower_red1 = np.array([0,   150, 150], dtype=np.uint8)
        self._upper_red1 = np.array([10,  255, 255], dtype=np.uint8)
        self._lower_red2 = np.array([170, 150, 150], dtype=np.uint8)
        self._upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        yolo_detections: Optional[List[Detection]] = None,
    ) -> List[Detection]:
        """
        Run colour detection on *frame* and return a list of body-box Detections.

        Parameters
        ----------
        frame           : BGR frame from the capture source.
        yolo_detections : Raw YOLO detections (before confidence filtering) used
                          for the person-proximity check.  Pass None to skip the
                          check (e.g. during testing without a live YOLO model).

        Returns an empty list when disabled or when no name tags survive all
        filters.  All returned Detections have is_enemy=True.
        """
        if not self._enabled:
            return []

        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self._lower_red1, self._upper_red1)
        mask2 = cv2.inRange(hsv, self._lower_red2, self._upper_red2)
        mask  = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Pass 1 — collect candidates that pass the colour / shape gate
        # Each entry: (tag_cx, tag_cy, body_Detection)
        candidates: List[Tuple[float, float, Detection]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self._min_area < area < self._max_area):
                continue

            tx, ty, tw, th = cv2.boundingRect(cnt)

            # Shape gate: name tags are wide horizontal strips.
            # Thick/tall or square contours are health bars, text, or UI noise.
            if th > self._max_tag_height_px:
                continue
            if th > 0 and tw / th < self._min_tag_aspect:
                continue

            # Estimate body box below the tag.
            half_extra = tw
            body_x1 = max(0,  tx - half_extra)
            body_y1 = ty
            body_x2 = min(fw, tx + tw + half_extra)
            body_y2 = min(fh, ty + _BODY_HEIGHT_MULT * th)

            if body_x2 <= body_x1 or body_y2 <= body_y1:
                continue

            det = Detection(
                x1=float(body_x1),
                y1=float(body_y1),
                x2=float(body_x2),
                y2=float(body_y2),
                confidence=_HUD_CONFIDENCE,
                class_id=0,
                is_enemy=True,
            )
            tag_cx = tx + tw / 2.0
            tag_cy = ty + th / 2.0
            candidates.append((tag_cx, tag_cy, det))
            log.debug(
                "HUD tag (%d,%d) %dx%d -> body (%.0f,%.0f)-(%.0f,%.0f)",
                tx, ty, tw, th, body_x1, body_y1, body_x2, body_y2,
            )

        # Pass 2 — apply false-positive guards
        detections: List[Detection] = []
        for tag_cx, tag_cy, det in candidates:
            # Guard 1: body-box centre must not fall in a fixed HUD region
            if self._use_hud_zones and self._in_hud_zone(det.cx, det.cy, fw, fh):
                log.debug(
                    "HUD_DETECTOR: rejected tag at (%.0f,%.0f) — HUD zone",
                    tag_cx, tag_cy,
                )
                continue

            # Guard 2: a raw YOLO detection must be nearby (below) the tag
            if yolo_detections is not None:
                if not self._has_nearby_person(tag_cx, tag_cy, yolo_detections):
                    log.debug(
                        "HUD_DETECTOR: rejected tag at (%.0f,%.0f) — no YOLO person within %.0fpx",
                        tag_cx, tag_cy, self._person_proximity_px,
                    )
                    continue

            detections.append(det)

        if detections:
            log.info(
                "HUD_DETECTOR: %d/%d tag(s) passed all filters -> %d body box(es)",
                len(detections), len(candidates), len(detections),
            )

        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _in_hud_zone(self, cx: float, cy: float, fw: int, fh: int) -> bool:
        """True if the normalised centre (cx/fw, cy/fh) falls in any exclusion zone."""
        cx_n = cx / fw
        cy_n = cy / fh
        for x1n, y1n, x2n, y2n in _HUD_EXCLUSION_ZONES:
            if x1n <= cx_n <= x2n and y1n <= cy_n <= y2n:
                return True
        return False

    def _has_nearby_person(
        self,
        tag_cx: float,
        tag_cy: float,
        yolo_detections: List[Detection],
    ) -> bool:
        """
        True if any YOLO detection centre is within person_proximity_px of the
        tag centre AND is located below it (cy > tag_cy).
        """
        for det in yolo_detections:
            if det.cy < tag_cy:
                continue  # skip detections above the tag
            dist = math.hypot(det.cx - tag_cx, det.cy - tag_cy)
            if dist <= self._person_proximity_px:
                return True
        return False
