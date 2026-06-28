"""Tests for ragnarok.aim.select — pure select_target + stateful TargetSelector.

Task 4 TDD: write tests first, confirm they fail, then implement.

CI-safe: no GPU, no display, no real mouse/keyboard.
FakeClock provides deterministic dwell timing.
"""
from __future__ import annotations

import pytest

from ragnarok.core.types import Team, Track, Tracks
from ragnarok.aim.select import TargetSelector, select_target


class FakeClock:
    """Deterministic clock for testing dwell timers (returns nanoseconds)."""

    def __init__(self, t: int = 0) -> None:
        self.t = t

    def __call__(self) -> int:
        return self.t


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------


def _enemy(tid: int, xyxy: tuple[float, float, float, float], conf: float = 0.9) -> Track:
    return Track(track_id=tid, xyxy=xyxy, confidence=conf, class_id=0, team=Team.ENEMY)


def _teammate(tid: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=0, team=Team.TEAMMATE)


def _unknown(tid: int, xyxy: tuple[float, float, float, float]) -> Track:
    return Track(track_id=tid, xyxy=xyxy, confidence=0.9, class_id=0, team=Team.UNKNOWN)


# ---------------------------------------------------------------------------
# select_target — pure function
# ---------------------------------------------------------------------------


class TestSelectTargetPure:
    CH = (192.0, 192.0)

    def test_picks_nearest_enemy(self):
        near = _enemy(1, (180.0, 180.0, 204.0, 260.0))  # close to crosshair
        far = _enemy(2, (300.0, 300.0, 324.0, 380.0))   # far from crosshair
        assert select_target(Tracks((near, far)), self.CH, 200.0) == 1

    def test_empty_returns_none(self):
        assert select_target(Tracks(()), self.CH, 200.0) is None

    def test_teammate_never_chosen(self):
        t = _teammate(1, (180.0, 180.0, 204.0, 260.0))
        assert select_target(Tracks((t,)), self.CH, 200.0) is None

    def test_unknown_never_chosen(self):
        u = _unknown(1, (180.0, 180.0, 204.0, 260.0))
        assert select_target(Tracks((u,)), self.CH, 200.0) is None

    def test_mixed_only_enemy_chosen(self):
        e = _enemy(1, (180.0, 180.0, 204.0, 260.0))
        t = _teammate(2, (182.0, 182.0, 206.0, 262.0))   # slightly closer but teammate
        assert select_target(Tracks((e, t)), self.CH, 200.0) == 1

    def test_enemy_outside_fov_returns_none(self):
        far = _enemy(1, (500.0, 500.0, 524.0, 580.0))
        assert select_target(Tracks((far,)), self.CH, 50.0) is None

    def test_switch_margin_keeps_lock(self):
        # cur and challenger are very close in cost → switch_margin=0.2 keeps the lock
        cur = _enemy(1, (186.0, 186.0, 198.0, 250.0))
        challenger = _enemy(2, (184.0, 184.0, 196.0, 250.0))
        result = select_target(
            Tracks((cur, challenger)), self.CH, 300.0,
            current_target_id=1, retain_fov_px=300.0, switch_margin=0.20,
        )
        assert result == 1

    def test_no_switch_margin_takes_best(self):
        cur = _enemy(1, (300.0, 300.0, 324.0, 380.0))        # far from crosshair
        challenger = _enemy(2, (190.0, 190.0, 210.0, 260.0))  # near crosshair
        result = select_target(
            Tracks((cur, challenger)), self.CH, 400.0,
            current_target_id=1, retain_fov_px=400.0, switch_margin=0.0,
        )
        assert result == 2

    def test_tiebreak_by_track_id(self):
        # Two enemies at identical bboxes → lower track_id wins
        a = _enemy(1, (190.0, 190.0, 210.0, 210.0))
        b = _enemy(2, (190.0, 190.0, 210.0, 210.0))
        result = select_target(Tracks((b, a)), self.CH, 300.0)
        assert result == 1

    def test_current_retained_inside_outer_but_outside_inner(self):
        # current target is outside inner fov_px but inside retain_fov_px → kept
        cur = _enemy(1, (240.0, 240.0, 260.0, 280.0))  # ~79px from crosshair
        result = select_target(
            Tracks((cur,)), self.CH, 50.0,
            current_target_id=1, retain_fov_px=300.0,
        )
        assert result == 1

    def test_current_outside_retain_not_kept(self):
        # current target is outside retain_fov_px → not retained
        cur = _enemy(1, (500.0, 500.0, 520.0, 540.0))   # very far
        result = select_target(
            Tracks((cur,)), self.CH, 50.0,
            current_target_id=1, retain_fov_px=100.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# TargetSelector — stateful class
# ---------------------------------------------------------------------------


class TestTargetSelector:
    CH = (192.0, 192.0)

    def _selector(self, clk=None, **kwargs) -> TargetSelector:
        defaults = dict(
            fov_px=300.0,
            retain_fov_px=400.0,
            dwell_ms=100.0,
            switch_margin=0.0,
            head_frac=0.15,
            clock=clk if clk is not None else FakeClock(),
        )
        defaults.update(kwargs)
        return TargetSelector(**defaults)

    def test_acquires_enemy(self):
        sel = self._selector()
        e = _enemy(1, (196.0, 196.0, 208.0, 260.0))
        assert sel.select(Tracks((e,)), *self.CH) == 1

    def test_teammate_never_chosen(self):
        sel = self._selector()
        t = _teammate(1, (196.0, 196.0, 208.0, 260.0))
        assert sel.select(Tracks((t,)), *self.CH) is None

    def test_unknown_never_chosen(self):
        sel = self._selector()
        u = _unknown(1, (196.0, 196.0, 208.0, 260.0))
        assert sel.select(Tracks((u,)), *self.CH) is None

    def test_lock_resets_on_death(self):
        sel = self._selector()
        e = _enemy(1, (196.0, 196.0, 208.0, 260.0))
        assert sel.select(Tracks((e,)), *self.CH) == 1
        # Target disappears from tracks
        assert sel.select(Tracks(()), *self.CH) is None

    def test_stays_none_after_death(self):
        sel = self._selector()
        e = _enemy(1, (196.0, 196.0, 208.0, 260.0))
        sel.select(Tracks((e,)), *self.CH)
        sel.select(Tracks(()), *self.CH)
        # Still None with no targets
        assert sel.select(Tracks(()), *self.CH) is None

    def test_lock_resets_on_fov_exit(self):
        sel = self._selector(fov_px=100.0, retain_fov_px=150.0)
        e = _enemy(1, (196.0, 196.0, 208.0, 260.0))   # within inner FOV
        assert sel.select(Tracks((e,)), *self.CH) == 1
        # Move target far outside retain FOV
        e_far = Track(
            track_id=1, xyxy=(600.0, 600.0, 624.0, 680.0),
            confidence=0.9, class_id=0, team=Team.ENEMY,
        )
        assert sel.select(Tracks((e_far,)), *self.CH) is None

    def test_switch_margin_keeps_lock(self):
        sel = self._selector(switch_margin=0.2)
        cur = _enemy(1, (186.0, 186.0, 198.0, 250.0))
        challenger = _enemy(2, (184.0, 184.0, 196.0, 250.0))
        sel.select(Tracks((cur,)), *self.CH)
        # challenger is only marginally closer → margin=0.2 keeps the lock
        result = sel.select(Tracks((cur, challenger)), *self.CH)
        assert result == 1

    def test_dwell_then_switch(self):
        clk = FakeClock()
        sel = self._selector(clk=clk, dwell_ms=100.0, switch_margin=0.0)
        # a is farther from crosshair, b is closer
        a = _enemy(1, (250.0, 250.0, 274.0, 330.0))
        b = _enemy(2, (196.0, 196.0, 208.0, 260.0))
        # Acquire a first
        assert sel.select(Tracks((a,)), *self.CH) == 1
        # b appears at t=0 — start dwell timer for challenger b
        sel.select(Tracks((a, b)), *self.CH)
        # dwell not yet elapsed → keep a
        assert sel.select(Tracks((a, b)), *self.CH) == 1
        # Advance clock past dwell (200ms >> 100ms)
        clk.t = 200_000_000
        # Now b has held long enough → switch
        assert sel.select(Tracks((a, b)), *self.CH) == 2

    def test_no_switch_before_dwell(self):
        clk = FakeClock()
        sel = self._selector(clk=clk, dwell_ms=500.0, switch_margin=0.0)
        a = _enemy(1, (250.0, 250.0, 274.0, 330.0))
        b = _enemy(2, (196.0, 196.0, 208.0, 260.0))
        sel.select(Tracks((a,)), *self.CH)
        # Only 100ms elapsed, dwell is 500ms → keep a
        clk.t = 100_000_000
        result = sel.select(Tracks((a, b)), *self.CH)
        assert result == 1

    def test_reset_clears_lock(self):
        sel = self._selector()
        e = _enemy(1, (196.0, 196.0, 208.0, 260.0))
        sel.select(Tracks((e,)), *self.CH)
        sel.reset()
        assert sel.target_id is None

    def test_target_id_property(self):
        sel = self._selector()
        assert sel.target_id is None
        e = _enemy(1, (196.0, 196.0, 208.0, 260.0))
        sel.select(Tracks((e,)), *self.CH)
        assert sel.target_id == 1

    def test_reacquire_after_death(self):
        sel = self._selector()
        e1 = _enemy(1, (196.0, 196.0, 208.0, 260.0))
        e2 = _enemy(2, (200.0, 200.0, 220.0, 280.0))
        sel.select(Tracks((e1,)), *self.CH)
        sel.select(Tracks(()), *self.CH)   # e1 dies
        # New target e2 should be acquired
        assert sel.select(Tracks((e2,)), *self.CH) == 2
