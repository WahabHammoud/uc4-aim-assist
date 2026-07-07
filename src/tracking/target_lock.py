"""
Target-lock state machine — engagement-based single-box selection.

Behaviour
---------
A single red box appears ONLY when the player is demonstrably engaging an
enemy.  The engagement gate requires:
  1. L2 held  (player is in ADS mode)
  2. R2 held  (player is actively firing)
     — live:    real R2 trigger value from DualSenseReader
     — demo:    caller passes r2_held=True when inferred from video geometry
  3. At least ONE enemy is within strict_radius_px of screen centre.
     When multiple candidates are present the highest-priority one is selected
     by score_candidate() (50% crosshair proximity, 30% confidence, 20% age).
  4. The selected enemy's track has been visible for at least min_stable_frames
     (new/unstable detection → no box)

Once engaged, the box is held for engagement_hold_frames after the gate
closes, covering brief occlusions, reloads, and detection gaps.

States
------
  NO_BOX   → no engagement: show nothing.
  ENGAGED  → single red box on the locked enemy (gate open and conditions met).
  HOLDING  → single red box maintained while hold countdown runs; gate was
             closed but enemy is still visible and hold has not expired.

Transitions
-----------
  NO_BOX + gate_open + stable_best_candidate → ENGAGED  (lock acquired)
  ENGAGED + conditions still met             → ENGAGED  (maintains lock)
  ENGAGED + conditions fail + enemy visible  → HOLDING  (hold countdown)
  HOLDING + conditions met again             → ENGAGED  (re-engages)
  HOLDING + hold countdown expires           → NO_BOX
  HOLDING + locked enemy lost                → NO_BOX
  any + L2 released                          → NO_BOX

Kalman predictor
----------------
A Kalman filter tracks the locked enemy's aim point for the PID controller.
It provides a smooth estimated aim point even during the hold phase when the
enemy briefly steps out of detection.  The Kalman state is internal and does
NOT affect the displayed box (which is always drawn at the actual bbox).
"""
from __future__ import annotations

import math
from collections import defaultdict
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.detection.detector import Detection
from src.tracking.kalman_predictor import KalmanPredictor
from src.utils.geometry import clamp_box
from src.utils.logger import get_logger

log = get_logger(__name__)


class LockState(Enum):
    NO_BOX   = auto()   # Nothing shown
    ENGAGED  = auto()   # Single red box, actively tracking
    HOLDING  = auto()   # Single red box, hold countdown


class TargetLock:
    """
    Engagement-based target selector.

    Parameters
    ----------
    config : dict
        target_lock sub-section from config.yaml.
    frame_width / frame_height : int
        Capture resolution (used to compute screen centre).
    aim_point_ratio : float
        Fraction of bbox height for the aim-point (0.30 = upper chest).
    """

    def __init__(
        self,
        config: dict,
        frame_width: int,
        frame_height: int,
        aim_point_ratio: float = 0.30,
        aim_point_x_ratio: float = 0.50,
    ):
        _radius_std    = float(config.get("strict_radius_px",         380))
        _radius_lowres = float(config.get("strict_radius_px_lowres", 125))
        self._radius   = _radius_lowres if frame_height < 720 else _radius_std
        self._min_stable    = int(  config.get("min_stable_frames",          8))
        self._hold          = int(  config.get("engagement_hold_frames",     45))
        self._max_predict   = int(  config.get("kalman_max_predict_frames",  50))
        self._reacq_dist    = float(config.get("reacquire_distance_px",      80))
        # Fix C — belt-and-suspenders: track must be at least this old to hold.
        # Set to 0 to disable.  Production default: 3.
        self._min_hold_age  = int(  config.get("min_hold_track_age",          0))

        # Prompt 4: high-confidence detections skip stability gate (age ≥ 1 is enough)
        self._high_conf_threshold  = float(config.get("high_conf_fast_track",  0.75))
        # Prompt 6: crowded scene — lower min_stable to 1 when many enemies visible
        self._crowded_threshold    = int(  config.get("crowded_min_detections",   2))

        # Prompt 7: target priority scoring
        self._crosshair_y_ratio     = float(config.get("crosshair_y_ratio",         0.45))
        self._switch_min_score_drop = float(config.get("switch_min_score_drop",     0.20))
        self._switch_min_score_gap  = float(config.get("switch_min_score_gap",      0.40))
        self._crosshair_x = frame_width  * 0.5
        self._crosshair_y = frame_height * self._crosshair_y_ratio
        self._max_score_dist = math.hypot(frame_width, frame_height)

        # Lock persistence and hysteresis (Prompt 3)
        self._lock_release_frames  = int(  config.get("lock_release_frames",    12))
        self._min_reacq_confidence = float(config.get("min_reacq_confidence",  0.65))
        self._max_reacq_distance   = float(config.get("max_reacq_distance",    200))
        self._lock_expiry_frames   = int(  config.get("lock_expiry_frames",      90))

        self._aim_ratio   = aim_point_ratio
        self._aim_x_ratio = aim_point_x_ratio

        self._fw = frame_width
        self._fh = frame_height
        self._cx = frame_width  / 2.0
        self._cy = frame_height / 2.0

        self._state      = LockState.NO_BOX
        self._locked_id: Optional[int] = None
        self._locked_box: Optional[Tuple[int, int, int, int]] = None
        self._hold_cnt   = 0
        self._predict_frames = 0

        # Lock persistence state (Prompt 3)
        self._release_countdown  = 0
        self._expiry_counter     = 0
        self._smooth_w: Optional[float] = None   # populated in Prompt 5
        self._smooth_h: Optional[float] = None   # populated in Prompt 5
        self._last_known_cx: Optional[float] = None
        self._last_known_cy: Optional[float] = None

        # Track-age counter: how many frames each track_id has been seen
        self._track_age: Dict[int, int] = defaultdict(int)

        # Kalman predictor for smooth PID aim-point
        self._kalman = KalmanPredictor(dt=1.0 / 60.0)

        self._last_aim_x = self._cx
        self._last_aim_y = self._cy

    # ── Main update ──────────────────────────────────────────────────────────

    def update(
        self,
        enemies: List[Detection],
        l2_held: bool,
        r2_held: Optional[bool] = True,
    ) -> Tuple[Optional[Tuple[float, float]], LockState]:
        """
        Process one frame.

        Parameters
        ----------
        enemies   : classified enemy detections with ByteTrack track_ids.
        l2_held   : True while L2 trigger is pressed beyond threshold.
                    — live: from DualSenseReader controller state
                    — demo: pass True always (demo assumes ADS mode)
        r2_held   : Engagement gate.
                    — live: from DualSenseReader.r2 >= r2_activation_threshold
                    — demo: None → infer internally from video geometry:
                      gate opens when exactly one stable enemy is in the strict
                      zone (single unambiguous target, precision over recall).

        Returns
        -------
        aim_point : (x, y) pixel position for the PID controller, or None.
        state     : current LockState for display (NO_BOX / ENGAGED / HOLDING).
        """
        # CHANGE 4: Lock expiry — count NO_BOX frames; clear position memory when stale
        if self._state == LockState.NO_BOX:
            self._expiry_counter += 1
            if self._expiry_counter >= self._lock_expiry_frames:
                self._last_known_cx = None
                self._last_known_cy = None
                self._smooth_w      = None
                self._smooth_h      = None
                self._expiry_counter = 0
                log.debug("Lock expired — position memory cleared")
        else:
            self._expiry_counter = 0

        # Update track-age counters for all visible enemies
        for d in enemies:
            if d.track_id is not None:
                self._track_age[d.track_id] += 1

        # Hard gate: must be in ADS mode
        if not l2_held:
            self._release("L2 released")
            return None, self._state

        # Find candidates in the strict engagement zone
        candidates = [e for e in enemies if self._in_zone(e)]

        # Prompt 6: crowded scene — temporarily reduce stability requirement
        effective_min_stable = 1 if len(enemies) > self._crowded_threshold else None

        # CHANGE 1+2: ENGAGED dropout — locked track vanished, stay ENGAGED via Kalman
        if self._state == LockState.ENGAGED and self._locked_id is not None:
            _locked_visible = next(
                (e for e in enemies if e.track_id == self._locked_id), None
            )
            if _locked_visible is None:
                self._release_countdown += 1
                if self._release_countdown < self._lock_release_frames:
                    px, py = self._kalman.predict_next()
                    half_w = self._smooth_w / 2.0 if self._smooth_w is not None else 60
                    half_h = self._smooth_h / 2.0 if self._smooth_h is not None else 120
                    pred_box = clamp_box(
                        int(px - half_w), int(py - half_h),
                        int(px + half_w), int(py + half_h),
                        self._fw, self._fh,
                    )
                    if pred_box is not None:
                        self._locked_box = pred_box
                    self._last_aim_x, self._last_aim_y = px, py
                    log.debug(
                        "Dropout %d/%d — Kalman hold on track #%d",
                        self._release_countdown, self._lock_release_frames,
                        self._locked_id,
                    )
                    return (px, py), self._state
                else:
                    log.info(
                        "Dropout exceeded %d frames on track #%d — releasing",
                        self._lock_release_frames, self._locked_id,
                    )
                    self._release("dropout_exceeded")
                    return None, self._state
            else:
                self._release_countdown = 0

        # Prompt 7: score every in-zone candidate and pick the highest-priority one.
        # Ambiguity between multiple candidates is resolved by score instead of
        # being rejected outright.
        best_candidate: Optional[Detection] = None
        if candidates:
            best_candidate = max(candidates, key=self._score_candidate)

            # Switch hysteresis — don't abandon an existing lock for a new
            # candidate unless the current target's score has collapsed AND
            # the new candidate is decisively better.
            if (self._locked_id is not None
                    and best_candidate.track_id != self._locked_id):
                current_cand = next(
                    (c for c in candidates if c.track_id == self._locked_id), None
                )
                if current_cand is not None:
                    current_score = self._score_candidate(current_cand)
                    new_score     = self._score_candidate(best_candidate)
                    if not (current_score < self._switch_min_score_drop
                            and new_score - current_score > self._switch_min_score_gap):
                        best_candidate = current_cand

        # ── Determine whether engagement gate is open ──────────────────────
        if r2_held is None:
            # Demo mode: infer engagement from video geometry.
            # Gate opens when the top-priority candidate in the zone is stable.
            gate_open = (best_candidate is not None
                         and self._is_stable(best_candidate, effective_min_stable))
        else:
            gate_open = r2_held

        # ── Engagement logic ───────────────────────────────────────────────
        if gate_open and best_candidate is not None:
            cand = best_candidate
            if self._is_stable(cand, effective_min_stable):
                # Place 1: clamp box to frame before use (guards PID from OOB coords)
                clamped = clamp_box(
                    int(cand.x1), int(cand.y1), int(cand.x2), int(cand.y2),
                    self._fw, self._fh,
                )
                if clamped is None:
                    log.debug("Degenerate box for track #%d after clamping", cand.track_id)
                    return self._try_hold(enemies, "degenerate_box")

                # CHANGE 3: stricter re-acquisition gate when coming from NO_BOX
                if self._state == LockState.NO_BOX and self._last_known_cx is not None:
                    if cand.confidence < self._min_reacq_confidence:
                        log.debug(
                            "Re-acq blocked: conf=%.2f < %.2f for track #%d",
                            cand.confidence, self._min_reacq_confidence, cand.track_id,
                        )
                        return None, self._state
                    reacq_dist = math.hypot(
                        cand.cx - self._last_known_cx,
                        cand.cy - self._last_known_cy,
                    )
                    if reacq_dist > self._max_reacq_distance:
                        log.debug(
                            "Re-acq blocked: dist=%.0fpx > %.0fpx for track #%d",
                            reacq_dist, self._max_reacq_distance, cand.track_id,
                        )
                        return None, self._state

                # Engage or maintain engagement
                if self._locked_id != cand.track_id:
                    log.info(
                        "Engaged enemy #%d  dist=%.0fpx  age=%d frames",
                        cand.track_id,
                        self._dist(cand),
                        self._track_age.get(cand.track_id, 0),
                    )
                self._locked_id  = cand.track_id
                self._locked_box = clamped
                self._state      = LockState.ENGAGED
                self._hold_cnt   = self._hold
                self._predict_frames = 0
                self._release_countdown = 0
                self._expiry_counter    = 0

                cx1, cy1, cx2, cy2 = clamped
                # STEP 3: EMA-smooth box dimensions (alpha=0.25 → slow to change)
                raw_w = float(cx2 - cx1)
                raw_h = float(cy2 - cy1)
                if self._smooth_w is None:
                    self._smooth_w = raw_w
                    self._smooth_h = raw_h
                else:
                    self._smooth_w = 0.25 * raw_w + 0.75 * self._smooth_w
                    self._smooth_h = 0.25 * raw_h + 0.75 * self._smooth_h

                ax = cx1 + self._aim_x_ratio * (cx2 - cx1)
                ay = cy1 + self._aim_ratio   * (cy2 - cy1)
                self._kalman.update(ax, ay)
                self._last_aim_x, self._last_aim_y = ax, ay
                # Update confirmed last-known position every real (non-predicted) frame
                self._last_known_cx = (cx1 + cx2) / 2.0
                self._last_known_cy = (cy1 + cy2) / 2.0
                return (ax, ay), self._state

            else:
                # Candidate exists but hasn't been tracked long enough
                log.debug(
                    "Enemy #%d in zone but unstable (age=%d/%d)",
                    cand.track_id,
                    self._track_age.get(cand.track_id, 0),
                    self._min_stable,
                )
                return self._try_hold(enemies, "unstable")

        elif len(candidates) == 0:
            return self._try_hold(enemies, "no_candidate")

        else:
            # gate_open is False (R2 not pressed)
            return self._try_hold(enemies, "r2_off")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _try_hold(
        self,
        enemies: List[Detection],
        reason: str,
    ) -> Tuple[Optional[Tuple[float, float]], LockState]:
        """
        Maintain a hold-box on the locked enemy while the hold countdown runs.
        Falls through to NO_BOX if countdown expired or enemy is gone.
        """
        if self._hold_cnt > 0 and self._locked_id is not None:
            locked = next((e for e in enemies if e.track_id == self._locked_id), None)
            if locked is not None:
                # Fix C: belt-and-suspenders age gate on HOLDING.
                # Prompt 4: high-confidence detections bypass this gate after 1 frame
                # (already passed self-exclusion in ObjectFilter upstream).
                age = self._track_age.get(locked.track_id, 0)
                high_conf_override = (
                    locked.confidence >= self._high_conf_threshold and age >= 1
                )
                if self._min_hold_age > 0 and age < self._min_hold_age and not high_conf_override:
                    log.info(
                        "SELF_EXCLUSION(C): HOLDING rejected — track #%d age %d < %d",
                        locked.track_id, age, self._min_hold_age,
                    )
                else:
                    # Place 1: clamp box to frame before use
                    clamped = clamp_box(
                        int(locked.x1), int(locked.y1),
                        int(locked.x2), int(locked.y2),
                        self._fw, self._fh,
                    )
                    if clamped is None:
                        # Degenerate — fall through to Kalman prediction
                        pass
                    else:
                        self._locked_box = clamped
                        cx1, cy1, cx2, cy2 = clamped
                        self._state    = LockState.HOLDING
                        self._hold_cnt -= 1
                        ax = (cx1 + cx2) / 2.0
                        ay = cy1 + self._aim_ratio * (cy2 - cy1)
                        self._kalman.update(ax, ay)
                        self._last_aim_x, self._last_aim_y = ax, ay
                        log.debug("HOLDING  countdown=%d  reason=%s", self._hold_cnt, reason)
                        return (ax, ay), self._state

            # Enemy not visible — try Kalman prediction to maintain hold
            if self._predict_frames < self._max_predict:
                self._state = LockState.HOLDING
                self._hold_cnt -= 1
                self._predict_frames += 1
                px, py = self._kalman.predict_next()
                self._last_aim_x, self._last_aim_y = px, py
                return (px, py), self._state

        self._release(reason)
        return None, self._state

    def _in_zone(self, det: Detection) -> bool:
        """True if the detection centre is within the strict engagement radius."""
        return self._dist(det) <= self._radius

    def _dist(self, det: Detection) -> float:
        return math.hypot(det.cx - self._cx, det.cy - self._cy)

    def _score_candidate(self, det: Detection) -> float:
        """
        Prompt 7: priority score combining crosshair proximity, confidence,
        and track age.  Higher is better.
        """
        dist_to_crosshair = math.hypot(det.cx - self._crosshair_x, det.cy - self._crosshair_y)
        proximity_score = max(0.0, 1.0 - dist_to_crosshair / self._max_score_dist)
        age = self._track_age.get(det.track_id, 0)
        age_score = min(age / 20.0, 1.0)
        return 0.50 * proximity_score + 0.30 * det.confidence + 0.20 * age_score

    def _is_stable(self, det: Detection, min_stable: Optional[int] = None) -> bool:
        threshold = min_stable if min_stable is not None else self._min_stable
        age = self._track_age.get(det.track_id, 0)
        return age >= threshold

    def _release(self, reason: str = "") -> None:
        if self._state != LockState.NO_BOX:
            log.info("Lock released (was on #%s)  reason=%s", self._locked_id, reason)
        self._state             = LockState.NO_BOX
        self._locked_id         = None
        self._locked_box        = None
        self._hold_cnt          = 0
        self._predict_frames    = 0
        self._release_countdown = 0
        self._smooth_w          = None
        self._smooth_h          = None
        self._kalman.reset()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def locked_id(self) -> Optional[int]:
        return self._locked_id

    @property
    def locked_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Clamped (x1, y1, x2, y2) of the currently engaged enemy, or None."""
        return self._locked_box
