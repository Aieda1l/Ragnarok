"""Tests for AdaptiveLead frame-age + EWMA latency estimation."""
from __future__ import annotations

from ragnarok.latency.adaptive_lead import AdaptiveLead


def test_lead_includes_frame_age_plus_base():
    al = AdaptiveLead(alpha=0.1, base_latency_s=0.005)
    lead = al.lead_seconds(t_capture_ns=1_000_000_000, now_ns=1_008_000_000)
    assert abs(lead - (0.008 + 0.005)) < 1e-9   # 8 ms age + 5 ms base


def test_frame_age_never_negative():
    al = AdaptiveLead(base_latency_s=0.0)
    lead = al.lead_seconds(t_capture_ns=2_000, now_ns=1_000)  # clock skew
    assert lead >= 0.0


def test_observe_actuation_ewma():
    al = AdaptiveLead(alpha=0.5, base_latency_s=0.0)
    al.observe_actuation(0.010)   # 0 + 0.5*(0.010-0) = 0.005
    assert abs(al.latency_s - 0.005) < 1e-9
    al.observe_actuation(0.010)   # 0.005 + 0.5*(0.010-0.005) = 0.0075
    assert abs(al.latency_s - 0.0075) < 1e-9
