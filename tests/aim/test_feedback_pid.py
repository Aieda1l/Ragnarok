"""Tests for FeedbackAimer's 2-DOF PID upgrade (Ki/Kd + anti-windup)."""
from __future__ import annotations
from ragnarok.aim.aimers import FeedbackAimer


def test_defaults_reproduce_p_controller():
    a = FeedbackAimer(kp=0.5, max_step_px=1e9, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), 0.01)
    assert abs(dx - 5.0) < 1e-9 and abs(dy) < 1e-9     # pure P, unchanged


def test_integral_accumulates_and_adds():
    a = FeedbackAimer(kp=0.0, ki=2.0, max_step_px=1e9, ema_alpha=1.0)
    # kp=0 isolates I. error=10 held; integral grows by error*dt each step.
    a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # integral=1.0 -> ki*I=2.0
    dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # integral=2.0 -> ki*I=4.0
    assert abs(dx - 4.0) < 1e-9


def test_integral_contribution_clamp():
    a = FeedbackAimer(kp=0.0, ki=10.0, integral_clamp=3.0, max_step_px=1e9, ema_alpha=1.0)
    for _ in range(10):
        dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)
    assert abs(dx - 3.0) < 1e-9          # ki*I clamped to +3


def test_conditional_integration_only_when_close():
    a = FeedbackAimer(kp=0.0, ki=5.0, cond_integ_thresh_px=5.0,
                      max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (100.0, 0.0), 0.1)   # |e|=100 > 5 -> no integration
    dx, _ = a.step((0.0, 0.0), (100.0, 0.0), 0.1)
    assert abs(dx) < 1e-9                    # integral still 0


def test_derivative_on_filtered_error_opposes_rapid_approach():
    a = FeedbackAimer(kp=0.0, kd=1.0, max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (0.0, 0.0), 0.1)     # error 0 (seed)
    dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # error jumps 0->10, deriv=+100
    assert dx > 0.0                          # kd*derivative term present


def test_reset_clears_integral():
    a = FeedbackAimer(kp=0.0, ki=2.0, max_step_px=1e9, ema_alpha=1.0)
    a.step((0.0, 0.0), (10.0, 0.0), 0.1)
    a.reset()
    dx, _ = a.step((0.0, 0.0), (10.0, 0.0), 0.1)   # integral restarts at error*dt
    assert abs(dx - 2.0) < 1e-9


def test_freeze_on_saturation_prevents_windup():
    # ki=1, kp=0, max_step=1, error=10 held: the magnitude clamp saturates every
    # frame, so anti-windup #3 must back out each integral increment -> _ix stays
    # bounded near 1 instead of growing to ~100.
    a = FeedbackAimer(kp=0.0, ki=1.0, max_step_px=1.0, ema_alpha=1.0)
    for _ in range(100):
        a.step((0.0, 0.0), (10.0, 0.0), 0.1)
    assert a._ix < 5.0     # without freeze-on-saturation this would reach ~100


def test_derivative_is_on_filtered_error_not_raw():
    # ema_alpha=0.5 so filtered error lags raw. kd=1, dt=1, kp=0, large max_step.
    # Step 2 error jumps 0->100: filtered error = 0 + 0.5*(100-0) = 50, so the
    # derivative term is (50-0)/1 = 50, NOT the raw 100 (which would be a kick).
    a = FeedbackAimer(kp=0.0, kd=1.0, ema_alpha=0.5, max_step_px=1e9)
    a.step((0.0, 0.0), (0.0, 0.0), 1.0)              # seed at error 0
    dx, _ = a.step((0.0, 0.0), (100.0, 0.0), 1.0)    # error jumps to 100
    assert abs(dx - 50.0) < 1e-6                      # filtered (50), not raw (100)
