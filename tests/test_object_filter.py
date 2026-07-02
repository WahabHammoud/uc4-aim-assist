"""
Unit tests for ObjectFilter.

Tests cover:
  - Valid enemy passes all checks
  - is_enemy=False always rejected (regardless of geometry)
  - HUD zone rejection (top bar, bottom bar, minimap)
  - Too-small area rejection
  - Too-large area rejection
  - Aspect ratio below minimum rejected
  - Aspect ratio above maximum rejected
  - Multiple detections: mix of pass/fail
  - Fix A: self-player centre exclusion zone rejects lower-centre body
  - Fix A: detection outside zone passes
  - Fix A: spectator-mode escape hatch (all candidates in zone → keep all)
  - Fix B: bbox bottom edge at/above 95% rejected
  - Fix B: bbox bottom edge below 95% passes
  - Fix A+B: combined — enemy at edge of zone not rejected by zone but by bottom gate
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.detector import Detection
from src.detection.object_filter import ObjectFilter

_DEFAULT_CFG = {
    "hud_exclusion_zones": [
        [0.00, 0.00, 1.00, 0.08],
        [0.00, 0.88, 1.00, 1.00],
        [0.00, 0.00, 0.15, 0.25],
    ],
    "min_bbox_area_px": 400,
    "max_bbox_area_px": 250_000,
    "min_aspect_ratio": 0.40,   # height / width
    "max_aspect_ratio": 5.00,   # height / width
}

W, H = 1920, 1080


def _det(x1, y1, x2, y2, is_enemy=True, conf=0.9) -> Detection:
    d = Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=0)
    d.is_enemy = is_enemy
    return d


class TestObjectFilter:

    def setup_method(self):
        self.f = ObjectFilter(_DEFAULT_CFG)

    # ---- Pass cases ----

    def test_valid_enemy_passes(self):
        # Mid-screen, reasonable person bbox
        d = _det(800, 300, 900, 700)   # 100×400 px, area=40000, ratio=0.25
        assert self.f.filter([d], W, H) == [d]

    def test_minimal_valid_area(self):
        # Exactly at minimum: 20×20 = 400 px²
        d = _det(500, 500, 520, 520)
        assert self.f.filter([d], W, H) == [d]

    # ---- is_enemy=False always rejected ----

    def test_not_enemy_rejected_regardless_of_geometry(self):
        d = _det(800, 300, 900, 700, is_enemy=False)
        assert self.f.filter([d], W, H) == []

    # ---- HUD zone rejection ----

    def test_top_hud_rejected(self):
        # cy = 50/1080 ≈ 4.6% → inside top bar [0%→8%]
        d = _det(100, 20, 200, 80)    # cy ≈ 50
        assert self.f.filter([d], W, H) == []

    def test_bottom_hud_rejected(self):
        # cy = 1000/1080 ≈ 92.6% → inside bottom bar [88%→100%]
        d = _det(100, 970, 200, 1030)  # cy ≈ 1000
        assert self.f.filter([d], W, H) == []

    def test_minimap_rejected(self):
        # cx=100/1920≈5%, cy=150/1080≈14% → inside minimap [0%→15%,0%→25%]
        d = _det(80, 100, 120, 200)   # cx=100, cy=150
        assert self.f.filter([d], W, H) == []

    def test_just_outside_top_hud_passes(self):
        # cy ≈ 14% → just outside top bar [0%→8%]; thin/tall bbox for valid aspect ratio
        # x1=500,y1=90,x2=550,y2=270 → cy=(90+270)/2=180 (16.7%), aspect=50/180=0.28
        d = _det(500, 90, 550, 270)
        assert self.f.filter([d], W, H) == [d]

    # ---- Area checks ----

    def test_too_small_area_rejected(self):
        d = _det(500, 500, 519, 520)  # 19×20=380 < 400
        assert self.f.filter([d], W, H) == []

    def test_too_large_area_rejected(self):
        # 500×510 = 255,000 > 250,000
        d = _det(100, 100, 600, 610)
        assert self.f.filter([d], W, H) == []

    # ---- Aspect ratio checks ----

    def test_too_wide_aspect_rejected(self):
        # width=300, height=100 → ratio=3.0 > 2.0
        d = _det(300, 400, 600, 500)
        assert self.f.filter([d], W, H) == []

    def test_too_tall_aspect_rejected(self):
        # width=10, height=500 → ratio=0.02 < 0.15
        d = _det(500, 200, 510, 700)
        assert self.f.filter([d], W, H) == []

    def test_valid_aspect_passes(self):
        # width=50, height=200 → ratio=0.25 ∈ [0.15, 2.0]
        d = _det(500, 300, 550, 500)
        assert self.f.filter([d], W, H) == [d]

    # ---- Multiple detections ----

    def test_mixed_detections(self):
        valid = _det(800, 300, 900, 700)
        hud   = _det(100, 20, 200, 80)      # top HUD
        small = _det(500, 500, 510, 510)    # area=100 < 400
        non_e = _det(800, 300, 900, 700, is_enemy=False)
        result = self.f.filter([valid, hud, small, non_e], W, H)
        assert result == [valid]

    def test_empty_input(self):
        assert self.f.filter([], W, H) == []

    # ── Fix A: self-player centre exclusion zone ──────────────────────────

    def test_self_zone_centre_rejected(self):
        """Detection centred at 50%, 75% (lower-centre) → rejected by Fix A."""
        # cx=960 (50%), cy=810 (75%)
        d = _det(910, 710, 1010, 910)   # cx=960, cy=810 — 50% x, 75% y
        assert self.f.filter([d], W, H) == []

    def test_self_zone_left_edge_rejected(self):
        """Detection centred at 35% x (exactly on border), 80% y → rejected by Fix A."""
        # cx=672 (35%), cy=864 (80%) — x exactly on lower boundary, y well inside zone
        d = _det(622, 764, 722, 964)    # cx=672, cy=864
        assert self.f.filter([d], W, H) == []

    def test_self_zone_enemy_outside_zone_passes(self):
        """Detection centred at 20% x, 70% y — outside x-range → passes Fix A."""
        # cx=384 (20%), cy=756 (70%) — x < 35%, not in self zone
        d = _det(334, 656, 434, 856)    # cx=384, cy=756
        assert self.f.filter([d], W, H) == [d]

    def test_self_zone_enemy_above_y_threshold_passes(self):
        """Detection centred at 50% x but only 40% y — above 60% threshold → passes."""
        # cx=960 (50%), cy=432 (40%)
        d = _det(910, 332, 1010, 532)   # cx=960, cy=432
        assert self.f.filter([d], W, H) == [d]

    def test_self_zone_spectator_mode_escape_hatch(self):
        """If every candidate falls in the self-zone, return all (spectator mode)."""
        # Two detections both in lower-centre: cx~50%, cy~75%
        d1 = _det(860, 700, 960, 900)   # cx=910 (47%), cy=800 (74%)
        d2 = _det(960, 700, 1060, 900)  # cx=1010 (53%), cy=800 (74%)
        result = self.f.filter([d1, d2], W, H)
        assert result == [d1, d2]

    # ── Fix B: bottom-edge gate ───────────────────────────────────────────

    def test_bottom_edge_at_threshold_rejected(self):
        """bbox y2 = 95% of frame height → rejected by Fix B."""
        # y2 = 0.95 * 1080 = 1026; cy must be above HUD zone (< 88%)
        # y1 = 826, y2 = 1026 → cy = 926/1080 ≈ 85.7% — just under HUD (88%)
        # cx centred at 20% to be outside self-zone x-range
        d = _det(300, 826, 400, 1026)   # y2=1026 → y2/H=0.950 >= 0.95
        assert self.f.filter([d], W, H) == []

    def test_bottom_edge_above_threshold_passes(self):
        """bbox y2 = 90% of frame height → passes Fix B (below 95% threshold)."""
        # y2 = 0.90 * 1080 = 972; cy must be above HUD (< 88%: cy < 950)
        # y1 = 772, y2 = 972 → cy = 872/1080 ≈ 80.7%
        d = _det(300, 772, 400, 972)    # y2=972 → y2/H=0.900 < 0.95
        assert self.f.filter([d], W, H) == [d]

    # ── Prompt 4: close-range detection fixes ──────────────────────────────

    def test_close_range_wide_box_within_new_aspect_passes(self):
        """width=500,height=200 → h/w=0.40 (new min, inclusive) passes, but the
        old width/height computation (ratio=2.5) would have exceeded the old
        max_aspect_ratio=2.00 and been rejected."""
        d = _det(700, 400, 1200, 600)   # 500x200, cx=950(49.5%), cy=500(46.3%)
        assert self.f.filter([d], W, H) == [d]

    def test_close_range_large_area_capped_by_fraction(self):
        """max_bbox_area_fraction=0.45 → box covering 50% of frame rejected."""
        cfg = {**_DEFAULT_CFG, "max_bbox_area_fraction": 0.45}
        f = ObjectFilter(cfg)
        # 1296x800 = 1,036,800px = exactly 50% of 1920x1080, h/w=0.617 (valid aspect),
        # cy=46.3% (outside self-zone y-range, outside HUD/bottom-edge gate)
        d = _det(312, 100, 1608, 900)
        assert f.filter([d], W, H) == []

    def test_close_range_large_area_within_fraction_passes(self):
        """max_bbox_area_fraction=0.45 → box covering ~40% of frame passes."""
        cfg = {**_DEFAULT_CFG, "max_bbox_area_fraction": 0.45}
        f = ObjectFilter(cfg)
        # 1036x800 = 828,800px ≈ 40% of frame, h/w=0.772 (valid), cy=46.3% (safe zone)
        d = _det(442, 100, 1478, 900)
        assert f.filter([d], W, H) == [d]

    def test_size_adjusted_confidence_large_box_low_conf_passes(self):
        """conf_threshold=0.55 standard, but large_box (>15% area) only needs 0.40."""
        cfg = {
            **_DEFAULT_CFG,
            "conf_threshold": 0.55,
            "large_box_area_fraction": 0.15,
            "large_box_conf_threshold": 0.40,
            "max_bbox_area_fraction": 0.30,
        }
        f = ObjectFilter(cfg)
        # 593x700 ≈ 415,100px ≈ 20% of frame (>15% large-box threshold), h/w=1.18 (valid),
        # cy=41.7% (safe zone), conf=0.45 ≥ 0.40 (large-box threshold)
        d = _det(664, 100, 1257, 800, conf=0.45)
        assert f.filter([d], W, H) == [d]

    def test_size_adjusted_confidence_small_box_low_conf_rejected(self):
        """Standard-size box below conf_threshold=0.55 is rejected (no size break)."""
        cfg = {**_DEFAULT_CFG, "conf_threshold": 0.55}
        f = ObjectFilter(cfg)
        d = _det(800, 300, 900, 700, conf=0.45)   # 100x400, small fraction of frame
        assert f.filter([d], W, H) == []

    def test_bottom_edge_spectator_escape_hatch(self):
        """All candidates have low bottom edge → spectator mode, keep all."""
        # Both detections have y2 >= 95%, cx outside self-zone x-range
        d1 = _det(200, 800, 300, 1026)  # y2=1026/1080=0.950
        d2 = _det(350, 800, 450, 1030)  # y2=1030/1080=0.954; cx=400 (20%), NOT in x zone
        # Wait — d2: cx=400/1920=20.8%, outside x self-zone (35-65%) — so fix A won't catch it.
        # Only fix B catches it. Both caught by fix B → spectator escape triggers.
        result = self.f.filter([d1, d2], W, H)
        assert result == [d1, d2]
