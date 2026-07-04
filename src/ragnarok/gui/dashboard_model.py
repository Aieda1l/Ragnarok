"""Pure telemetry-history + sparkline geometry for the Dashboard (spec §10.3).

ZERO Qt: a bounded ring buffer of FPS / loop-latency samples and a function that
maps a series into polyline points inside a rect. The QPainter render lives in
dashboard_panel; all the math is here so it is unit-testable without a display.
"""
from __future__ import annotations

from collections import deque


class TelemetryHistory:
    KEYS = ("fps", "p50", "p99")

    def __init__(self, maxlen: int = 240) -> None:
        self._series = {k: deque(maxlen=maxlen) for k in self.KEYS}

    def push(self, *, fps: float, p50: float, p99: float) -> None:
        self._series["fps"].append(float(fps))
        self._series["p50"].append(float(p50))
        self._series["p99"].append(float(p99))

    def push_snapshot(self, snap) -> None:
        self.push(fps=snap.fps, p50=snap.loop_ms_p50, p99=snap.loop_ms_p99)

    def series(self, key: str) -> tuple[float, ...]:
        return tuple(self._series[key])

    def stats(self) -> dict[str, float]:
        return {k: (self._series[k][-1] if self._series[k] else 0.0)
                for k in self.KEYS}

    def __len__(self) -> int:
        return len(self._series["fps"])
