"""
Unit tests for FrameProfiler.

Tests cover:
  - begin_frame / end_frame round-trip
  - section() context manager records times
  - fps() returns sensible value
  - p95() on real section data
  - report() includes all sections and total
  - total_avg_ms property (previously broken — was computing p95('_total') which returned 0.0)
  - should_log() triggers at correct intervals
  - empty profiler returns zero / no crash
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.profiler import FrameProfiler


class TestFrameProfiler:

    def test_frame_count_increments(self):
        p = FrameProfiler()
        assert p.frame_count == 0
        p.begin_frame()
        p.end_frame()
        assert p.frame_count == 1
        p.begin_frame()
        p.end_frame()
        assert p.frame_count == 2

    def test_section_records_time(self):
        p = FrameProfiler()
        p.begin_frame()
        with p.section("test"):
            time.sleep(0.01)   # 10 ms
        p.end_frame()
        assert p.avg("test") >= 8.0   # should be ≥8 ms (loose for CI)

    def test_fps_positive_after_frames(self):
        p = FrameProfiler()
        for _ in range(5):
            p.begin_frame()
            time.sleep(0.001)
            p.end_frame()
        assert p.fps() > 0.0

    def test_total_avg_ms_property(self):
        """This was broken: p95('_total') was always 0.0 because it looked in the
        wrong dict. Now total_avg_ms computes directly from _frame_total deque."""
        p = FrameProfiler()
        p.begin_frame()
        time.sleep(0.01)
        p.end_frame()
        assert p.total_avg_ms >= 8.0

    def test_total_avg_ms_zero_when_empty(self):
        p = FrameProfiler()
        assert p.total_avg_ms == pytest.approx(0.0, abs=1e-9)

    def test_p95_section(self):
        p = FrameProfiler(history=100)
        for i in range(20):
            p.begin_frame()
            with p.section("work"):
                time.sleep(0.001)
            p.end_frame()
        p95 = p.p95("work")
        assert p95 >= p.avg("work")   # p95 ≥ mean

    def test_report_contains_total(self):
        p = FrameProfiler()
        p.begin_frame()
        with p.section("capture"):
            pass
        p.end_frame()
        report = p.report()
        assert "total" in report
        assert "capture" in report

    def test_report_total_p95_not_zero(self):
        """Previously report() showed p95=0.00ms for total (bug). Must now be >0."""
        p = FrameProfiler()
        for _ in range(5):
            p.begin_frame()
            time.sleep(0.002)
            p.end_frame()
        report = p.report()
        lines = [l for l in report.splitlines() if "total" in l]
        assert lines, "report must have a 'total' line"
        # Parse p95 value: "  total   avg=X.XXms  p95=Y.YYms"
        import re
        m = re.search(r"p95=(\d+\.\d+)ms", lines[0])
        assert m, f"p95 not found in: {lines[0]}"
        p95_val = float(m.group(1))
        assert p95_val > 0.0, f"p95 was {p95_val} — was 0.0 due to old bug"

    def test_should_log_at_interval(self):
        p = FrameProfiler(log_interval_frames=5)
        for i in range(1, 6):
            p.begin_frame()
            p.end_frame()
        assert p.should_log()   # frame_count=5 → 5%5==0 → True

    def test_should_not_log_before_interval(self):
        p = FrameProfiler(log_interval_frames=10)
        for _ in range(7):
            p.begin_frame()
            p.end_frame()
        assert not p.should_log()

    def test_multiple_sections(self):
        p = FrameProfiler()
        p.begin_frame()
        with p.section("a"):
            pass
        with p.section("b"):
            pass
        with p.section("c"):
            pass
        p.end_frame()
        assert p.avg("a") >= 0.0
        assert p.avg("b") >= 0.0
        assert p.avg("c") >= 0.0

    def test_fps_empty(self):
        p = FrameProfiler()
        assert p.fps() == pytest.approx(0.0, abs=1e-9)

    def test_avg_unknown_section(self):
        p = FrameProfiler()
        assert p.avg("nonexistent") == pytest.approx(0.0, abs=1e-9)

    def test_p95_unknown_section(self):
        p = FrameProfiler()
        assert p.p95("nonexistent") == pytest.approx(0.0, abs=1e-9)
