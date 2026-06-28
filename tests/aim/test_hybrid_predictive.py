from ragnarok.aim.aimers import NullAimer, FlickAimer, FeedbackAimer


def test_all_aimers_accept_target_vel_kwarg():
    # Uniform signature so the controller can always pass velocity.
    assert NullAimer().step((0, 0), (10, 0), 0.01, target_vel=(5.0, 0.0)) == (0.0, 0.0)
    fl = FlickAimer(flick_speed_px_s=1000.0)
    dx, dy = fl.step((0, 0), (10, 0), 0.01, target_vel=(5.0, 0.0))
    assert dx > 0 and dy == 0
    fb = FeedbackAimer(kp=0.5, max_step_px=100.0, ema_alpha=1.0)
    dx, dy = fb.step((0, 0), (10, 0), 0.01, target_vel=(0.0, 0.0))
    assert dx > 0
