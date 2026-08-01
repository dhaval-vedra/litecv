"""
litecv._profiler — Performance profiler for edge device pipelines

Measures FPS, per-step latency, memory usage.
Zero dependencies (stdlib only).

Usage:
    from litecv.profiler import Profiler, fps_counter

    # Simple FPS counter
    fps = fps_counter()
    for frame in cam:
        process(frame)
        print(f"FPS: {fps():.1f}")

    # Detailed profiler
    prof = Profiler()
    for frame in cam:
        with prof.step("capture"):
            img = cam.capture()
        with prof.step("clahe"):
            img = img.clahe()
        with prof.step("thermal"):
            result = img.thermal()
    prof.report()
"""
from __future__ import annotations

import time
import contextlib
import statistics
from collections import defaultdict, deque
from typing import Callable, Optional, Dict, List


class Profiler:
    """
    Profile individual steps in a processing pipeline.

    Usage:
        prof = Profiler(window=100)
        with prof.step("blur"):
            img = img.blur(3)
        with prof.step("clahe"):
            img = img.clahe()
        prof.report()     # prints summary table
        stats = prof.stats()  # returns dict
    """

    def __init__(self, window: int = 100):
        self._window = window
        self._times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._counts: Dict[str, int] = defaultdict(int)
        self._global_start = time.perf_counter()
        self._frame_count = 0

    @contextlib.contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self._times[name].append(elapsed)
            self._counts[name] += 1

    def frame(self):
        """Call once per frame to track overall FPS."""
        self._frame_count += 1

    @property
    def fps(self) -> float:
        elapsed = time.perf_counter() - self._global_start
        return self._frame_count / elapsed if elapsed > 0 else 0.0

    def stats(self) -> Dict[str, dict]:
        result = {}
        for name, times in self._times.items():
            lst = list(times)
            if not lst: continue
            result[name] = {
                "count":   self._counts[name],
                "mean_ms": statistics.mean(lst) * 1000,
                "min_ms":  min(lst) * 1000,
                "max_ms":  max(lst) * 1000,
                "std_ms":  statistics.stdev(lst) * 1000 if len(lst) > 1 else 0.0,
                "fps":     1.0 / statistics.mean(lst) if statistics.mean(lst) > 0 else 0,
            }
        return result

    def report(self, sort_by: str = "mean_ms") -> str:
        s = self.stats()
        if not s:
            print("[litecv Profiler] No data yet.")
            return ""
        rows = sorted(s.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)
        header  = f"{'Step':<20} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'FPS':>8} {'Count':>8}"
        divider = "-" * len(header)
        lines = ["\n[litecv Profiler]", divider, header, divider]
        for name, stat in rows:
            lines.append(
                f"{name:<20} {stat['mean_ms']:>10.2f} {stat['min_ms']:>10.2f} "
                f"{stat['max_ms']:>10.2f} {stat['fps']:>8.1f} {stat['count']:>8d}"
            )
        lines.append(divider)
        lines.append(f"Overall FPS: {self.fps:.1f} | Frames: {self._frame_count}")
        output = "\n".join(lines)
        print(output)
        return output

    def reset(self):
        self._times.clear()
        self._counts.clear()
        self._frame_count = 0
        self._global_start = time.perf_counter()


def fps_counter(window: int = 30) -> Callable[[], float]:
    """
    Returns a function that tracks FPS over a rolling window.

    Usage:
        fps = fps_counter(window=30)
        for frame in cam:
            process(frame)
            print(f"FPS: {fps():.1f}")
    """
    times: deque = deque(maxlen=window)

    def tick() -> float:
        now = time.perf_counter()
        times.append(now)
        if len(times) < 2:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    return tick


class MemoryMonitor:
    """
    Monitor process memory usage (Linux /proc/self/status).

    Usage:
        mem = MemoryMonitor()
        print(f"RSS: {mem.rss_mb:.1f} MB")
        print(f"VmPeak: {mem.peak_mb:.1f} MB")
    """

    def _read_status(self) -> Dict[str, int]:
        try:
            with open("/proc/self/status") as f:
                lines = f.readlines()
            result = {}
            for line in lines:
                if ":" in line:
                    key, val = line.split(":", 1)
                    val = val.strip()
                    if val.endswith(" kB"):
                        result[key.strip()] = int(val[:-3])
            return result
        except FileNotFoundError:
            return {}

    @property
    def rss_mb(self) -> float:
        return self._read_status().get("VmRSS", 0) / 1024.0

    @property
    def peak_mb(self) -> float:
        return self._read_status().get("VmPeak", 0) / 1024.0

    @property
    def virt_mb(self) -> float:
        return self._read_status().get("VmSize", 0) / 1024.0

    def report(self) -> str:
        rss = self.rss_mb
        peak = self.peak_mb
        line = f"[litecv Memory] RSS: {rss:.1f} MB | Peak: {peak:.1f} MB"
        print(line)
        return line


class LatencyTracker:
    """
    Track end-to-end latency from capture to output.

    Usage:
        tracker = LatencyTracker()
        stamp = tracker.start()
        # ... process ...
        latency_ms = tracker.end(stamp)
    """

    def __init__(self):
        self._history: deque = deque(maxlen=100)

    def start(self) -> float:
        return time.perf_counter()

    def end(self, start: float) -> float:
        ms = (time.perf_counter() - start) * 1000
        self._history.append(ms)
        return ms

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self._history) if self._history else 0.0

    @property
    def p95_ms(self) -> float:
        if not self._history:
            return 0.0
        sorted_h = sorted(self._history)
        idx = int(len(sorted_h) * 0.95)
        return sorted_h[min(idx, len(sorted_h)-1)]
