"""Tests for RecoilPattern + RecoilCompensator spray-counter logic."""
from __future__ import annotations

from ragnarok.recoil.compensator import RecoilPattern, RecoilCompensator


def test_first_shot_counter_is_first_point_negated():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0), (0.0, 18.0)))
    rc = RecoilCompensator(pat, scale=1.0)
    assert rc.on_fire() == (0.0, 0.0)     # cumulative (0,0) -> no counter on shot 0


def test_subsequent_shots_counter_increments():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0), (0.0, 18.0)))
    rc = RecoilCompensator(pat)
    rc.on_fire()                          # shot 0
    assert rc.on_fire() == (0.0, -10.0)   # counter the +10 rise
    assert rc.on_fire() == (0.0, -8.0)    # counter the +8 rise


def test_scale_applied():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0)))
    rc = RecoilCompensator(pat, scale=0.5)
    rc.on_fire()
    assert rc.on_fire() == (0.0, -5.0)


def test_past_end_returns_zero():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0)))
    rc = RecoilCompensator(pat)
    rc.on_fire(); rc.on_fire()
    assert rc.on_fire() == (0.0, 0.0)


def test_release_resets_index():
    pat = RecoilPattern(points=((0.0, 0.0), (0.0, 10.0), (0.0, 18.0)))
    rc = RecoilCompensator(pat)
    rc.on_fire(); rc.on_fire()
    rc.release()
    assert rc.on_fire() == (0.0, 0.0)     # back to shot 0


def test_empty_pattern_is_safe():
    rc = RecoilCompensator(RecoilPattern(points=()))
    assert rc.on_fire() == (0.0, 0.0)
