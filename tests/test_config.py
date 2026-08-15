"""
Tests for config_local.yaml override precedence and the _deep_merge helper.

These tests exercise the merge logic directly (without touching the filesystem)
so they run fast and don't require a real config_local.yaml to be present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _deep_merge


class TestDeepMerge:

    def test_top_level_key_override(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99})
        assert base == {"a": 1, "b": 99}

    def test_top_level_new_key_added(self):
        base = {"a": 1}
        _deep_merge(base, {"z": 42})
        assert base["z"] == 42
        assert base["a"] == 1

    def test_nested_dict_merged_not_replaced(self):
        """Merging into a nested dict must not wipe out sibling keys."""
        base = {
            "target_lock": {
                "strict_radius_px": 1200,
                "use_fixed_box_size": False,
                "center_ema_alpha": 0.0,
            }
        }
        overrides = {
            "target_lock": {
                "use_fixed_box_size": True,
                "center_ema_alpha": 0.10,
            }
        }
        _deep_merge(base, overrides)
        # Overridden keys
        assert base["target_lock"]["use_fixed_box_size"] is True
        assert base["target_lock"]["center_ema_alpha"] == 0.10
        # Sibling key not present in overrides must survive
        assert base["target_lock"]["strict_radius_px"] == 1200

    def test_deep_nested_three_levels(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        _deep_merge(base, {"a": {"b": {"c": 99}}})
        assert base["a"]["b"]["c"] == 99
        assert base["a"]["b"]["d"] == 2   # sibling preserved

    def test_list_value_replaced_not_merged(self):
        """Lists are not recursively merged — the override replaces the whole list."""
        base = {"weights": [1, 2, 3]}
        _deep_merge(base, {"weights": [4, 5]})
        assert base["weights"] == [4, 5]

    def test_non_dict_value_replacing_dict_allowed(self):
        """If an override replaces a dict with a scalar, accept it (unusual but valid)."""
        base = {"section": {"key": "val"}}
        _deep_merge(base, {"section": "disabled"})
        assert base["section"] == "disabled"

    def test_empty_overrides_leaves_base_unchanged(self):
        base = {"x": 1, "y": {"z": 2}}
        original = {"x": 1, "y": {"z": 2}}
        _deep_merge(base, {})
        assert base == original

    def test_config_local_visualization_pattern(self):
        """Simulate the real use-case: config_local.yaml with only viz overrides."""
        base = {
            "target_lock": {
                "strict_radius_px": 1200,
                "min_stable_frames": 1,
                "use_fixed_box_size": False,
                "fixed_box_width_ratio": 0.12,
                "fixed_box_height_ratio": 0.28,
                "center_ema_alpha": 0.0,
            },
            "capture": {
                "mode": "chiaki",
                "capture_card_index": -1,
            },
        }
        local = {
            "target_lock": {
                "use_fixed_box_size": True,
                "fixed_box_width_ratio": 0.13,
                "fixed_box_height_ratio": 0.30,
                "center_ema_alpha": 0.10,
            },
            "capture": {
                "mode": "capture_card",
                "capture_card_index": 0,
            },
        }
        _deep_merge(base, local)

        assert base["target_lock"]["use_fixed_box_size"] is True
        assert base["target_lock"]["fixed_box_width_ratio"] == 0.13
        assert base["target_lock"]["fixed_box_height_ratio"] == 0.30
        assert base["target_lock"]["center_ema_alpha"] == 0.10
        assert base["target_lock"]["strict_radius_px"] == 1200  # untouched
        assert base["capture"]["mode"] == "capture_card"
        assert base["capture"]["capture_card_index"] == 0
