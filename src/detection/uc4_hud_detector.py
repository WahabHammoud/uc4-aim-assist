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
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.detection.detector import Detection
from src.utils.logger import get_logger

log = get_logger(__name__)

# Body-box geometry relative to the detected name-tag bounding rect.
# The name tag is roughly 1/3 the width of the character body and sits just
# above the head.  These multipliers produce a plausible full-body estimate:
#   width  = 3 × tag_width  (1 width left, tag width, 1 width right)
#   height = 8 × tag_height (body extends downward from tag top)
_BODY_WIDTH_MULT  = 3   # total body width  = MULT × tag width
_BODY_HEIGHT_MULT = 8   # body height below tag top = MULT × tag height

# Fixed confidence score for HUD-derived detections (between YOLO thresholds).
# 0.70 is intentionally above conf_threshold (0.25) so they never get dropped
# by the filter's confidence gate, but below high_conf_fast_track (0.75) so
# they go through the normal stability gate.
_HUD_CONFIDENCE = 0.70


class UC4HUDDetector:
    """
    Finds UC4 enemy name tags via HSV red thresholding and returns body-box
    Detections.  Instantiate once; call detect(frame) every inference tick.
    """

    def __init__(self, config: dict):
        self._enabled  = config.get("enabled", True)
        self._min_area = config.get("min_area", 20)
        self._max_area = config.get("max_area", 2000)
        # UC4 name tags are wide horizontal strips (e.g. 60x10px at 1080p).
        # Require tag width >= min_tag_aspect * tag height to filter out square/
        # tall UI elements like health bars, ammo counters, and dialog boxes.
        self._min_tag_aspect    = config.get("min_tag_aspect", 2.0)
        # Hard height cap: name tags are thin — anything taller is not a tag.
        self._max_tag_height_px = config.get("max_tag_height_px", 25)

        # Red occupies two HSV lobes (0-10 deg and 170-180 deg)
        self._lower_red1 = np.array([0,   150, 150], dtype=np.uint8)
        self._upper_red1 = np.array([10,  255, 255], dtype=np.uint8)
        self._lower_red2 = np.array([170, 150, 150], dtype=np.uint8)
        self._upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run colour detection on *frame* and return a list of body-box Detections.

        Returns an empty list when disabled or when no name tags are found.
        All returned Detections have is_enemy=True and are clamped to the frame.
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

        detections: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self._min_area < area < self._max_area):
                continue

            tx, ty, tw, th = cv2.boundingRect(cnt)

            # Shape gate: name tags are wide horizontal strips.
            # Thick/tall or square contours are UI elements, not name tags.
            if th > self._max_tag_height_px:
                continue
            if th > 0 and tw / th < self._min_tag_aspect:
                continue

            # Estimate body position below the name tag.
            # Tag centre is assumed to sit above the character's head, so the
            # body box starts at the tag's top edge and extends downward.
            half_extra = tw  # one tag-width of margin on each side
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
                class_id=0,      # person class — same as YOLO
                is_enemy=True,   # red name tag = enemy by definition
            )
            detections.append(det)
            log.debug(
                "HUD tag (%d,%d) %dx%d -> body (%.0f,%.0f)-(%.0f,%.0f)",
                tx, ty, tw, th,
                body_x1, body_y1, body_x2, body_y2,
            )

        if detections:
            log.info("HUD_DETECTOR: %d name tag(s) -> %d body box(es)", len(detections), len(detections))

        return detections
