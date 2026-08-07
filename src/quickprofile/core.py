"""
Lightweight time + memory profiling for any block of code.

Uses process RSS (via psutil) rather than tracemalloc for memory. This
matters for any code doing native/C-level allocation — numpy arrays,
PDF/image rendering (PyMuPDF, Pillow), subprocess buffers, etc. —
tracemalloc only tracks Python-level allocations and reports near-zero for
exactly the calls most likely to spike memory. RSS is measured directly
against the OS instead, and a background sampler thread polls it during the
block to catch transient peaks that a single before/after measurement could
miss if memory was freed before the block returned.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileResult:
    """Timing and RSS memory measurements for one profiled block."""

    label: str
    duration_seconds: float
    rss_start_mb: float
    rss_end_mb: float
    rss_peak_mb: float

    @property
    def rss_delta_mb(self) -> float:
        return self.rss_end_mb - self.rss_start_mb

    def __str__(self) -> str:
        return (
            f"[{self.label}] {self.duration_seconds:.2f}s | "
            f"RSS {self.rss_start_mb:.1f} -> {self.rss_end_mb:.1f} MB "
            f"(peak {self.rss_peak_mb:.1f} MB, delta {self.rss_delta_mb:+.1f} MB)"
        )


def _current_rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


class _RSSSampler:
    """Background thread polling RSS at a fixed interval to catch transient peaks."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, _current_rss_mb())
            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.peak_mb = _current_rss_mb()
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        return self.peak_mb


class _ResultHolder:
    """Mutable box so the context manager can hand back a result populated only on exit."""

    def __init__(self) -> None:
        self.result: ProfileResult | None = None


@contextmanager
def profile(label: str = "block"):
    """
    Measure wall time and process RSS around a block of code.

        with profile("pdf_conversion") as holder:
            images = convert_pdf_to_images(pdf_bytes)
        print(holder.result)  # populated on exit, not before
    """
    sampler = _RSSSampler()
    holder = _ResultHolder()

    rss_start = _current_rss_mb()
    start_time = time.perf_counter()
    sampler.start()
    try:
        yield holder
    finally:
        duration = time.perf_counter() - start_time
        sampled_peak = sampler.stop()
        rss_end = _current_rss_mb()
        holder.result = ProfileResult(
            label=label,
            duration_seconds=duration,
            rss_start_mb=rss_start,
            rss_end_mb=rss_end,
            rss_peak_mb=max(sampled_peak, rss_start, rss_end),
        )


def profiled(label: str | None = None, recorder: "ProfileRecorder | None" = None):
    """
    Decorator form of profile(), for wrapping a whole function rather than a block.

        @profiled()
        def load_dataset(path): ...

        load_dataset("data.csv")
        print(load_dataset.last_result)  # ProfileResult from the most recent call

    Pass recorder=some_recorder to also accumulate every call into it (e.g.
    profiling every call in a loop over many files) rather than only keeping
    the most recent result.
    """

    def decorator(func):
        import functools

        call_label = label or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profile(call_label) as holder:
                result = func(*args, **kwargs)
            wrapper.last_result = holder.result
            if recorder is not None:
                recorder.results.append(holder.result)
            return result

        wrapper.last_result = None
        return wrapper

    return decorator


@dataclass
class ProfileRecorder:
    """Accumulates ProfileResult objects across multiple profiled blocks, for a batch summary."""

    results: list[ProfileResult] = field(default_factory=list)

    @contextmanager
    def track(self, label: str):
        """Same usage as profile(), but appends the result to self.results instead of handing it back."""
        with profile(label) as holder:
            yield
        self.results.append(holder.result)

    def summary(self) -> str:
        if not self.results:
            return "No profiled operations recorded."

        lines = [
            "| Label | Duration (s) | RSS delta (MB) | RSS peak (MB) |",
            "|---|---|---|---|",
        ]
        for r in self.results:
            lines.append(f"| {r.label} | {r.duration_seconds:.2f} | {r.rss_delta_mb:+.1f} | {r.rss_peak_mb:.1f} |")

        total_duration = sum(r.duration_seconds for r in self.results)
        max_peak = max(r.rss_peak_mb for r in self.results)
        lines.append("")
        lines.append(f"Total duration: {total_duration:.2f}s | Max RSS peak observed: {max_peak:.1f} MB")
        return "\n".join(lines)
