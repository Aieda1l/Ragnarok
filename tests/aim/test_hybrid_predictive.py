from ragnarok.aim.aimers import NullAimer, FlickAimer, FeedbackAimer, HybridAimer


def test_all_aimers_accept_target_vel_kwarg():
    # Uniform signature so the controller can always pass velocity.
    assert NullAimer().step((0, 0), (10, 0), 0.01, target_vel=(5.0, 0.0)) == (0.0, 0.0)
    fl = FlickAimer(flick_speed_px_s=1000.0)
    dx, dy = fl.step((0, 0), (10, 0), 0.01, target_vel=(5.0, 0.0))
    assert dx > 0 and dy == 0
    fb = FeedbackAimer(kp=0.5, max_step_px=100.0, ema_alpha=1.0)
    dx, dy = fb.step((0, 0), (10, 0), 0.01, target_vel=(0.0, 0.0))
    assert dx > 0


def test_hybrid_far_is_proportional_not_full():
    a = HybridAimer(kp=0.3, max_step_px=100.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (200.0, 0.0), 0.01)   # error 200 >> flick_dist
    assert 0 < dx < 200.0          # proportional: a fraction of the error
    assert abs(dx - 0.3 * 200.0) < 1e-6


def test_hybrid_close_snaps_full_error():
    a = HybridAimer(kp=0.3, max_step_px=100.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (5.0, 0.0), 0.01)     # error 5 < flick_dist
    assert abs(dx - 5.0) < 1e-6 and abs(dy) < 1e-6    # full snap, no overshoot


def test_hybrid_never_overshoots_close():
    a = HybridAimer(kp=2.0, max_step_px=1000.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (10.0, 0.0), 0.01)
    assert 0 < dx <= 10.0 + 1e-9


def test_hybrid_far_regime_never_overshoots_high_kp():
    # kp>1 in the far regime must still not overshoot the remaining distance.
    import math
    a = HybridAimer(kp=2.0, max_step_px=1000.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (25.0, 0.0), 0.01)   # error 25 > flick_dist 20 -> far branch
    assert 0 < dx <= 25.0 + 1e-9 and abs(dy) < 1e-6


def test_hybrid_far_regime_max_step_clamp():
    import math
    a = HybridAimer(kp=1.0, max_step_px=5.0, flick_dist_px=20.0,
                    flick_speed_px_s=4000.0, ema_alpha=1.0)
    dx, dy = a.step((0.0, 0.0), (200.0, 0.0), 0.01)  # error 200, kp=1 -> 200 clamped to max_step 5
    assert abs(math.hypot(dx, dy) - 5.0) < 1e-6
