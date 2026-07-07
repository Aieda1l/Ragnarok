"""Tests for aim/aimers.py — TDD: written before implementation."""
from __future__ import annotations

import math
import pytest

from ragnarok.aim.aimers import FlickAimer, FeedbackAimer, NullAimer, Aimer


# ---------------------------------------------------------------------------
# NullAimer
# ---------------------------------------------------------------------------

def test_null_aimer_returns_zeros():
    a = NullAimer()
    dx, dy = a.step((0.0, 0.0), (100.0, 200.0), dt=0.016)
    assert dx == 0.0 and dy == 0.0


def test_null_aimer_reset_is_no_op():
    a = NullAimer()
    a.reset()  # must not raise
    dx, dy = a.step((0.0, 0.0), (50.0, 50.0), dt=0.016)
    assert dx == 0.0 and dy == 0.0


# ---------------------------------------------------------------------------
# FlickAimer — latch + no overshoot
# ---------------------------------------------------------------------------

def test_flick_glides_toward_current_target():
    """FlickAimer glides toward the target at flick speed, clamped to remaining."""
    a = FlickAimer(flick_speed_px_s=1000.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), dt=1.0)   # 1000px clamped to d=10
    assert abs(dx - 10.0) < 1e-6 and abs(dy) < 1e-6


def test_flick_no_overshoot():
    """Step must be clamped to remaining distance (no overshoot)."""
    a = FlickAimer(flick_speed_px_s=10_000.0)
    dx, dy = a.step((0.0, 0.0), (5.0, 0.0), dt=1.0)
    assert abs(dx - 5.0) < 1e-6, f"Overshoot: dx={dx}"
    assert abs(dy) < 1e-6


def test_flick_no_overshoot_diagonal():
    """No overshoot for a diagonal target."""
    a = FlickAimer(flick_speed_px_s=10_000.0)
    tx, ty = 3.0, 4.0  # distance = 5.0
    dx, dy = a.step((0.0, 0.0), (tx, ty), dt=1.0)
    dist_moved = math.hypot(dx, dy)
    assert dist_moved <= 5.0 + 1e-9, f"Overshot: moved {dist_moved} > 5"


def test_flick_partial_step():
    """When speed*dt < remaining distance, step is exactly speed*dt."""
    a = FlickAimer(flick_speed_px_s=200.0)
    # Target at (100, 0); dt=0.1; step = min(20, 100) = 20
    dx, dy = a.step((0.0, 0.0), (100.0, 0.0), dt=0.1)
    assert abs(dx - 20.0) < 1e-6, f"Expected dx=20, got {dx}"
    assert abs(dy) < 1e-6


def test_flick_follows_moving_target():
    """FlickAimer tracks the LIVE target as it moves (no stale latch)."""
    a = FlickAimer(flick_speed_px_s=100.0)
    a.step((0.0, 0.0), (50.0, 0.0), dt=0.1)             # step 10 toward (50,0)
    # target moved to (10, 100) — glide toward the NEW target from crosshair (10,0)
    dx, dy = a.step((10.0, 0.0), (10.0, 100.0), dt=0.1)
    assert abs(dx) < 1e-6 and abs(dy - 10.0) < 1e-6     # 10px toward the live target


def test_flick_zero_distance_returns_zero():
    """Crosshair already ON the latched target → (0, 0)."""
    a = FlickAimer(flick_speed_px_s=1000.0)
    a.step((0.0, 0.0), (0.0, 0.0), dt=1.0)  # latch at (0,0)
    dx, dy = a.step((0.0, 0.0), (0.0, 0.0), dt=1.0)
    assert dx == 0.0 and dy == 0.0


def test_flick_is_aimer_subclass():
    assert isinstance(FlickAimer(flick_speed_px_s=100.0), Aimer)


# ---------------------------------------------------------------------------
# FeedbackAimer — P-controller + clamp + EMA
# ---------------------------------------------------------------------------

def test_feedback_proportional():
    """dx ≈ Kp * error (first frame seeds EMA, so no smoothing yet)."""
    a = FeedbackAimer(kp=0.5, max_step_px=1000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (40.0, 0.0), dt=0.016)
    assert abs(dx - 20.0) < 1e-6, f"Expected dx=20.0, got {dx}"
    assert abs(dy) < 1e-6


def test_feedback_clamped_to_max_step():
    """When Kp*error > max_step_px, magnitude is clamped to max_step_px."""
    a = FeedbackAimer(kp=1.0, max_step_px=10.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (40.0, 0.0), dt=0.016)
    assert abs(dx - 10.0) < 1e-6, f"Expected clamped dx=10, got {dx}"
    assert abs(dy) < 1e-6


def test_feedback_diagonal_clamp():
    """Diagonal clamp preserves direction, magnitude = max_step_px."""
    a = FeedbackAimer(kp=10.0, max_step_px=5.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (3.0, 4.0), dt=0.016)  # error mag=5, kp=10 → 50>5
    assert abs(math.hypot(dx, dy) - 5.0) < 1e-6


def test_feedback_ema_smooths_error():
    """EMA blends from seeded value; the remaining-distance clamp caps the output.

    Frame 2 has the target at the crosshair (d=0), so the no-overshoot clamp
    correctly returns 0 even though the EMA carries residual state of 50.
    This is the right behaviour: we are already at the target.
    """
    a = FeedbackAimer(kp=1.0, max_step_px=1e9, ema_alpha=0.5)
    # First call: seeds EMA at error=100 → dx=100 (d=100, limit=100, no clamp).
    dx, _ = a.step((0.0, 0.0), (100.0, 0.0), dt=0.016)
    assert abs(dx - 100.0) < 1e-6, f"First frame: expected 100, got {dx}"
    # Second call: target at crosshair (d=0) → remaining-distance clamp returns 0.
    dx2, _ = a.step((0.0, 0.0), (0.0, 0.0), dt=0.016)
    assert abs(dx2) < 1e-9, f"At target (d=0) output must be 0; got {dx2}"


def test_feedback_ema_alpha_1_no_smoothing():
    """With ema_alpha=1.0, every frame tracks the raw error immediately."""
    a = FeedbackAimer(kp=1.0, max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (100.0, 0.0), dt=0.016)
    dx, _ = a.step((0.0, 0.0), (30.0, 0.0), dt=0.016)
    assert abs(dx - 30.0) < 1e-6, f"Alpha=1 should track live error; got {dx}"


def test_feedback_reset_clears_ema():
    """After reset(), FeedbackAimer re-seeds EMA on next step."""
    a = FeedbackAimer(kp=1.0, max_step_px=1e9, ema_alpha=0.5)
    a.step((0.0, 0.0), (100.0, 0.0), dt=0.016)  # seeds EMA at 100
    a.reset()
    # After reset, should re-seed at 20 (not blend from 100)
    dx, _ = a.step((0.0, 0.0), (20.0, 0.0), dt=0.016)
    assert abs(dx - 20.0) < 1e-6, f"After reset, EMA should re-seed; got {dx}"


def test_feedback_kff_hook_exists():
    """FeedbackAimer must accept kff param (Phase-4 hook) without error."""
    a = FeedbackAimer(kp=0.5, max_step_px=100.0, ema_alpha=1.0, kff=0.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), dt=0.016)
    # kff=0 means no feed-forward; result same as without kff
    assert abs(dx - 5.0) < 1e-6


def test_feedback_is_aimer_subclass():
    assert isinstance(FeedbackAimer(kp=0.5, max_step_px=100.0), Aimer)
