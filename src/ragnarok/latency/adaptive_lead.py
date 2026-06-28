"""Adaptive predictive lead (spec §6.5).

t_lead = true per-frame age (now - t_capture) + EWMA(actuation + transport
latency), recomputed each frame so prediction self-corrects under scheduling
jitter instead of using a fixed constant.
"""
from __future__ import annotations


class AdaptiveLead:
    def __init__(self, *, alpha: float = 0.1, base_latency_s: float = 0.0) -> None:
        self._alpha = alpha
        self._lat = base_latency_s

    @property
    def latency_s(self) -> float:
        return self._lat

    def observe_actuation(self, latency_s: float) -> None:
        self._lat += self._alpha * (latency_s - self._lat)

    def lead_seconds(self, t_capture_ns: int, now_ns: int) -> float:
        age = (now_ns - t_capture_ns) / 1e9
        if age < 0.0:
            age = 0.0
        return age + self._lat
