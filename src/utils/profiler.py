"""
Frame-level performance profiler.

Usage:
    profiler = FrameProfiler()
    with profiler.section("capture"):
        frame = capture()
    with profiler.section("inference"):
        results = model(frame)
    profiler.end_frame()
    if profiler.should_log():
        print(profiler.report())
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict, Generator


class FrameProfiler:
    def __init__(self, log_interval_frames: int = 300, history: int = 300):
        self._log_interval = log_interval_frames
        self._history = history
        self._frame_count = 0
        self._section_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history))
        self._frame_starts: Dict[str, float] = {}
        self._frame_total: deque = deque(maxlen=history)
        self._frame_start: float = 0.0

    def begin_frame(self) -> None:
        self._frame_start = time.perf_counter()

    @contextmanager
    def section(self, name: str) -> Generator:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._section_times[name].append(elapsed_ms)

    def end_frame(self) -> None:
        total_ms = (time.perf_counter() - self._frame_start) * 1000.0
        self._frame_total.append(total_ms)
        self._frame_count += 1

    def should_log(self) -> bool:
        return self._frame_count % self._log_interval == 0

    def avg(self, name: str) -> float:
        times = self._section_times.get(name)
        if not times:
            return 0.0
        return sum(times) / len(times)

    def p95(self, name: str) -> float:
        times = self._section_times.get(name)
        if not times:
            return 0.0
        sorted_times = sorted(times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def fps(self) -> float:
        if not self._frame_total:
            return 0.0
        avg_ms = sum(self._frame_total) / len(self._frame_total)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def report(self) -> str:
        lines = [f"[Profiler] Frame #{self._frame_count}  FPS={self.fps():.1f}"]
        total_vals = list(self._frame_total)
        total_avg = sum(total_vals) / len(total_vals) if total_vals else 0.0
        total_p95 = sorted(total_vals)[int(len(total_vals) * 0.95)] if len(total_vals) > 1 else total_avg
        lines.append(f"  {'total':<18} avg={total_avg:.2f}ms  p95={total_p95:.2f}ms")
        for name, times in sorted(self._section_times.items()):
            avg = sum(times) / len(times) if times else 0.0
            p95 = sorted(list(times))[int(len(times) * 0.95)] if len(times) > 1 else avg
            lines.append(f"  {name:<18} avg={avg:.2f}ms  p95={p95:.2f}ms")
        return "\n".join(lines)

    @property
    def total_avg_ms(self) -> float:
        """Average total frame time in milliseconds."""
        if not self._frame_total:
            return 0.0
        return sum(self._frame_total) / len(self._frame_total)

    @property
    def frame_count(self) -> int:
        return self._frame_count
