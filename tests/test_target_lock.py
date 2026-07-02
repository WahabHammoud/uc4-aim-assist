"""
Unit tests for TargetLock engagement-based state machine.

Tests cover:
  - L2 gate: L2 not held → NO_BOX always
  - R2 gate: R2 not held, no prior lock → NO_BOX
  - Demo mode: r2_held=None → gate inferred from video geometry
  - Acquisition: single stable enemy in strict zone + gates open → ENGAGED
  - Strict zone: enemy outside radius → NO_BOX
  - Ambiguity rejection: 2+ enemies in zone → NO_BOX
  - Non-ambiguous: one enemy in zone, one outside → ENGAGED
  - Stability gate: track age < min_stable → NO_BOX, then ENGAGED once threshold met
  - Aim point ratio: applied correctly to bbox height
  - HOLDING: gate closes (R2 off) but enemy still visible → box held
  - HOLDING timeout: expires after engagement_hold_frames + max_predict_frames → NO_BOX
  - no track_id: enemies without an ID are not eligible for engagement
  - Properties: locked_id and state match expectations
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import Detection
from src.tracking.target_lock import LockState, TargetLock

W, H = 1920, 1080
_SCREEN_CX = W / 2.0   # 960.0
_SCREEN_CY = H / 2.0   # 540.0

# Default test config — min_stable=1 so one update() call is enough to engage
_CFG = {
    "strict_radius_px": 380,
    "min_stable_frames": 1,
    "engagement_hold_frames": 5,
    "kalman_max_predict_frames": 3,
    "reacquire_distance_px": 80,
}

# Config for HOLDING timeout test — short hold and short dropout window
_TIMEOUT_CFG = {
    **_CFG,
    "engagement_hold_frames": 3,
    "kalman_max_predict_frames": 10,
    "lock_release_frames": 3,   # dropout window matches hold, keeps test fast
}

# Config for lock-persistence tests (Prompt 3)
_HYSTERESIS_CFG = {
    **_CFG,
    "lock_release_frames": 12,
    "min_reacq_confidence": 0.65,
    "max_reacq_distance": 200,
    "lock_expiry_frames": 90,
}

# Config for stability gate test — require 3 frames before engaging
_STRICT_CFG = {
    **_CFG,
    "min_stable_frames": 3,
}


def _make_lock(cfg=None) -> TargetLock:
    return TargetLock(config=cfg or _CFG, frame_width=W, frame_height=H, aim_point_ratio=0.30)


def _enemy(
    cx: float, cy: float, tid, bw: int = 100, bh: int = 200, conf: float = 0.9
) -> Detection:
    """Create a Detection centred at (cx, cy) with given size and track_id."""
    x1, y1 = cx - bw // 2, cy - bh // 2
    x2, y2 = cx + bw // 2, cy + bh // 2
    d = Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=0)
    d.is_enemy = True
    d.track_id = tid
    return d


class TestTargetLock:

    # ── L2 gate ───────────────────────────────────────────────────────────

    def test_l2_not_held_always_no_box(self):
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        aim, state = lock.update([e], l2_held=False, r2_held=True)
        assert state == LockState.NO_BOX
        assert aim is None

    def test_l2_releases_active_engagement(self):
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED

        aim, state = lock.update([e], l2_held=False, r2_held=True)
        assert state == LockState.NO_BOX
        assert aim is None
        assert lock.locked_id is None

    # ── R2 gate ───────────────────────────────────────────────────────────

    def test_r2_not_held_no_engagement(self):
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        aim, state = lock.update([e], l2_held=True, r2_held=False)
        assert state == LockState.NO_BOX
        assert aim is None

    # ── Acquisition ───────────────────────────────────────────────────────

    def test_no_enemies_stays_no_box(self):
        lock = _make_lock()
        aim, state = lock.update([], l2_held=True, r2_held=True)
        assert state == LockState.NO_BOX
        assert aim is None

    def test_single_enemy_in_zone_engages(self):
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=7)
        aim, state = lock.update([e], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED
        assert lock.locked_id == 7
        assert aim is not None

    def test_enemy_outside_radius_no_box(self):
        lock = _make_lock()
        # 450px > strict_radius_px=380
        e = _enemy(_SCREEN_CX + 450, _SCREEN_CY, tid=1)
        aim, state = lock.update([e], l2_held=True, r2_held=True)
        assert state == LockState.NO_BOX
        assert aim is None

    # ── Ambiguity rejection ───────────────────────────────────────────────

    def test_ambiguity_two_enemies_in_zone_resolved_by_score(self):
        """Prompt 7: scoring resolves multi-candidate ambiguity instead of
        rejecting to NO_BOX. Equal-score candidates → highest-priority
        (first-found) candidate engages."""
        lock = _make_lock()
        e1 = _enemy(_SCREEN_CX + 100, _SCREEN_CY, tid=1)
        e2 = _enemy(_SCREEN_CX - 100, _SCREEN_CY, tid=2)
        aim, state = lock.update([e1, e2], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED
        assert lock.locked_id == 1

    def test_second_enemy_outside_zone_not_ambiguous(self):
        """Two detected enemies but only one inside strict zone → not ambiguous → ENGAGED."""
        lock = _make_lock()
        near = _enemy(_SCREEN_CX + 100, _SCREEN_CY, tid=1)
        far  = _enemy(_SCREEN_CX + 500, _SCREEN_CY, tid=2)   # 500px > 380px radius
        _, state = lock.update([near, far], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED
        assert lock.locked_id == 1

    # ── Stability gate ────────────────────────────────────────────────────

    def test_stability_gate_blocks_then_allows(self):
        """min_stable_frames=3: enemy must be seen for 3 frames before engaging."""
        lock = _make_lock(_STRICT_CFG)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)

        _, s1 = lock.update([e], l2_held=True, r2_held=True)
        assert s1 == LockState.NO_BOX   # frame 1: age=1, need 3

        _, s2 = lock.update([e], l2_held=True, r2_held=True)
        assert s2 == LockState.NO_BOX   # frame 2: age=2, need 3

        _, s3 = lock.update([e], l2_held=True, r2_held=True)
        assert s3 == LockState.ENGAGED  # frame 3: age=3, threshold met

    # ── Aim point ─────────────────────────────────────────────────────────

    def test_aim_point_ratio_applied(self):
        """aim_point_ratio=0.30 → aim at 30% down from top of bbox."""
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1, bw=100, bh=300)
        aim, state = lock.update([e], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED
        assert aim is not None
        _, ay = aim
        expected_ay = e.y1 + 0.30 * e.height
        assert abs(ay - expected_ay) < 1.0

    # ── HOLDING ───────────────────────────────────────────────────────────

    def test_holding_when_r2_released(self):
        """After ENGAGED, dropping R2 while enemy is still visible → HOLDING."""
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED

        aim, state = lock.update([e], l2_held=True, r2_held=False)
        assert state == LockState.HOLDING
        assert aim is not None

    def test_holding_expires_no_box(self):
        """Dropout window exhausted → NO_BOX (lock_release_frames=3 in _TIMEOUT_CFG)."""
        lock = _make_lock(_TIMEOUT_CFG)   # hold=3, max_predict=10, lock_release_frames=3
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED

        # 3rd dropout frame hits release_countdown >= lock_release_frames → NO_BOX
        for _ in range(3):
            lock.update([], l2_held=True, r2_held=True)
        assert lock.state == LockState.NO_BOX

    # ── No track_id ───────────────────────────────────────────────────────

    def test_no_track_id_not_eligible(self):
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=None)
        _, state = lock.update([e], l2_held=True, r2_held=True)
        assert state == LockState.NO_BOX

    # ── Demo mode ─────────────────────────────────────────────────────────

    def test_demo_mode_single_stable_engages(self):
        """r2_held=None + single stable enemy in zone → gate inferred open → ENGAGED."""
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        _, state = lock.update([e], l2_held=True, r2_held=None)
        assert state == LockState.ENGAGED

    def test_demo_mode_ambiguous_resolved_by_score(self):
        """r2_held=None + two equal-score candidates → scoring picks one → ENGAGED."""
        lock = _make_lock()
        e1 = _enemy(_SCREEN_CX + 100, _SCREEN_CY, tid=1)
        e2 = _enemy(_SCREEN_CX - 100, _SCREEN_CY, tid=2)
        _, state = lock.update([e1, e2], l2_held=True, r2_held=None)
        assert state == LockState.ENGAGED
        assert lock.locked_id == 1

    # ── Lock persistence and hysteresis (Prompt 3) ─────────────────────────

    def test_t1_dropout_under_threshold_stays_engaged(self):
        """T1: locked track absent for 5 frames → ENGAGED (5 < lock_release_frames=12)."""
        lock = _make_lock(_HYSTERESIS_CFG)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED

        for _ in range(5):
            _, state = lock.update([], l2_held=True, r2_held=True)
            assert state == LockState.ENGAGED

    def test_t2_dropout_at_threshold_releases(self):
        """T2: locked track absent for 12 frames → NO_BOX."""
        lock = _make_lock(_HYSTERESIS_CFG)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED

        # Frame 12 is when release_countdown hits lock_release_frames=12 → NO_BOX
        for _ in range(12):
            lock.update([], l2_held=True, r2_held=True)
        assert lock.state == LockState.NO_BOX

    def _setup_no_box_with_last_known(self, cfg=None):
        """Helper: engage, dropout to NO_BOX, return lock with last_known_cx set."""
        lock = _make_lock(cfg or _HYSTERESIS_CFG)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        # Exhaust dropout window → NO_BOX
        for _ in range(12):
            lock.update([], l2_held=True, r2_held=True)
        assert lock.state == LockState.NO_BOX
        assert lock._last_known_cx is not None
        return lock

    def test_t3_reacq_low_confidence_blocked(self):
        """T3: conf=0.60 < min_reacq_confidence=0.65 within 150px → no re-acquire."""
        lock = self._setup_no_box_with_last_known()
        # New candidate near last known position but low confidence
        new_e = _enemy(_SCREEN_CX + 50, _SCREEN_CY, tid=2, conf=0.60)
        # Build up track age so stability gate passes
        for _ in range(2):
            lock.update([new_e], l2_held=True, r2_held=True)
        _, state = lock.update([new_e], l2_held=True, r2_held=True)
        assert state == LockState.NO_BOX

    def test_t4_reacq_high_confidence_within_distance_engages(self):
        """T4: conf=0.70 >= 0.65 within 150px → re-acquires."""
        lock = self._setup_no_box_with_last_known()
        # New candidate near last known position with sufficient confidence
        new_e = _enemy(_SCREEN_CX + 50, _SCREEN_CY, tid=2, conf=0.70)
        # Build up track age so stability gate passes
        for _ in range(2):
            lock.update([new_e], l2_held=True, r2_held=True)
        _, state = lock.update([new_e], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED

    def test_t5_expiry_clears_last_known_position(self):
        """T5: 90 consecutive NO_BOX frames → _last_known_cx/cy reset to None."""
        lock = self._setup_no_box_with_last_known()
        assert lock._last_known_cx is not None

        for _ in range(90):
            lock.update([], l2_held=True, r2_held=True)

        assert lock._last_known_cx is None
        assert lock._last_known_cy is None

    def test_t6_dropout_aim_point_not_none(self):
        """T6: during dropout window, aim_point is not None (Kalman provides it)."""
        lock = _make_lock(_HYSTERESIS_CFG)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED

        for _ in range(5):
            aim_pt, state = lock.update([], l2_held=True, r2_held=True)
            assert state == LockState.ENGAGED
            assert aim_pt is not None

    # ── Prompt 4: close-range / high-confidence fast-track ─────────────────

    def test_high_conf_bypasses_hold_age_gate(self):
        """High-confidence (≥0.75) track bypasses min_hold_track_age Fix-C gate."""
        cfg = {**_CFG, "min_hold_track_age": 5, "high_conf_fast_track": 0.75}
        lock = _make_lock(cfg)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1, conf=0.80)   # age will be 1 on first pass
        # Engage
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED
        # Drop R2 to force HOLDING path — track still visible, age=2, min_hold_track_age=5
        # Without high-conf override this would fall through to NO_BOX (age 2 < 5).
        _, state = lock.update([e], l2_held=True, r2_held=False)
        assert state == LockState.HOLDING

    def test_normal_conf_respects_hold_age_gate(self):
        """conf=0.50 < high_conf_threshold=0.75 → Fix-C gate enforced.
        With kalman_max_predict_frames=0 (no fallback), this releases to NO_BOX."""
        cfg = {
            **_CFG,
            "min_hold_track_age": 5,
            "high_conf_fast_track": 0.75,
            "kalman_max_predict_frames": 0,  # no Kalman fallback to expose Fix-C behaviour
        }
        lock = _make_lock(cfg)
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1, conf=0.50)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock.state == LockState.ENGAGED
        _, state = lock.update([e], l2_held=True, r2_held=False)
        assert state == LockState.NO_BOX   # Fix-C age gate rejects, Kalman exhausted

    # ── Prompt 5: EMA box-dimension smoothing ──────────────────────────────

    def test_ema_smoothing_initialises_on_first_frame(self):
        """smooth_w/h must be set (not None) after first ENGAGED frame."""
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1, bw=100, bh=200)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock._smooth_w is not None
        assert lock._smooth_h is not None

    def test_ema_smoothing_blends_on_subsequent_frames(self):
        """smooth_w converges toward raw_w via alpha=0.25 blend."""
        lock = _make_lock()
        e1 = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1, bw=100, bh=200)
        lock.update([e1], l2_held=True, r2_held=True)
        prev_w = lock._smooth_w
        # Update with larger box
        e2 = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1, bw=200, bh=200)
        lock.update([e2], l2_held=True, r2_held=True)
        expected = 0.25 * 200.0 + 0.75 * prev_w
        assert abs(lock._smooth_w - expected) < 0.01

    def test_ema_reset_on_release(self):
        """smooth_w/h must be None after lock is released."""
        lock = _make_lock()
        e = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        lock.update([e], l2_held=True, r2_held=True)
        assert lock._smooth_w is not None
        lock.update([], l2_held=False, r2_held=True)   # L2 off → release
        assert lock._smooth_w is None
        assert lock._smooth_h is None

    # ── Prompt 6: crowded scene override ───────────────────────────────────

    def test_crowded_scene_lowers_stability_threshold(self):
        """With >2 enemies visible, stability gate drops to 1 frame (crowded scene)."""
        cfg = {**_STRICT_CFG, "crowded_min_detections": 2}  # strict needs 3 frames normally
        lock = _make_lock(cfg)
        in_zone  = _enemy(_SCREEN_CX, _SCREEN_CY,        tid=1)
        outside1 = _enemy(_SCREEN_CX + 500, _SCREEN_CY,  tid=2)
        outside2 = _enemy(_SCREEN_CX - 500, _SCREEN_CY,  tid=3)
        # 3 total enemies (>crowded_min_detections=2) → effective_min_stable=1
        _, state = lock.update([in_zone, outside1, outside2], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED   # age=1 suffices in crowded scene

    def test_non_crowded_scene_uses_normal_stability(self):
        """With ≤2 enemies visible, standard min_stable_frames=3 applies."""
        cfg = {**_STRICT_CFG, "crowded_min_detections": 2}
        lock = _make_lock(cfg)
        in_zone = _enemy(_SCREEN_CX, _SCREEN_CY, tid=1)
        outside = _enemy(_SCREEN_CX + 500, _SCREEN_CY, tid=2)
        # 2 total enemies (not > crowded_min_detections) → normal threshold=3
        _, state = lock.update([in_zone, outside], l2_held=True, r2_held=True)
        assert state == LockState.NO_BOX   # age=1 insufficient, needs 3

    # ── Prompt 7: target priority scoring ──────────────────────────────────

    def test_closer_to_crosshair_selected_when_scores_differ(self):
        """Enemy closer to scoring-crosshair (y=H*0.45) wins over one farther away."""
        lock = _make_lock()
        crosshair_y = H * 0.45  # 486.0
        # e_near: exactly at crosshair
        e_near = _enemy(_SCREEN_CX, crosshair_y, tid=1, conf=0.80)
        # e_far: farther from crosshair
        e_far  = _enemy(_SCREEN_CX, crosshair_y + 200, tid=2, conf=0.80)
        _, state = lock.update([e_near, e_far], l2_held=True, r2_held=True)
        assert state == LockState.ENGAGED
        assert lock.locked_id == 1

    def test_switch_hysteresis_prevents_target_switch(self):
        """Even if a new candidate scores higher, hysteresis prevents switch when
        current lock's score has not dropped below switch_min_score_drop=0.20."""
        cfg = {**_CFG, "switch_min_score_drop": 0.20, "switch_min_score_gap": 0.40}
        lock = _make_lock(cfg)
        # Establish lock on e1 at screen centre (perfect crosshair proximity)
        e1 = _enemy(_SCREEN_CX, H * 0.45, tid=1, conf=0.90)
        lock.update([e1], l2_held=True, r2_held=True)
        assert lock.locked_id == 1

        # Now present e1 slightly off-centre AND a new e2 nearer to crosshair.
        # e1 will remain above score 0.20 threshold so hysteresis should block switch.
        e1_shifted = _enemy(_SCREEN_CX + 50, H * 0.45, tid=1, conf=0.80)
        e2_new     = _enemy(_SCREEN_CX,      H * 0.45, tid=2, conf=0.90)
        lock.update([e1_shifted, e2_new], l2_held=True, r2_held=True)
        assert lock.locked_id == 1   # should NOT have switched
