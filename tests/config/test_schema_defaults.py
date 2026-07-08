"""Phase 9P — snappy/steady schema defaults (Task 2)."""
from __future__ import annotations

from ragnarok.config.schema import AimConfig


def test_snappy_aim_defaults():
    a = AimConfig()
    assert a.adaptive_lead is False      # no lead-induced jitter by default
    assert a.lead_ms == 0.0
    assert a.commit == 0.85
    assert a.settle_px == 2.0


def test_aim_toggle_defaults():
    a = AimConfig()
    assert a.toggle is True               # toggle, not hold
    assert a.aim_key == "VK_XBUTTON2"     # non-obtrusive default
