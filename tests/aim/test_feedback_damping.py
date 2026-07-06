"""NeuralBot-style feedback damping: quadratic creep zone + sign-flip anti-windup."""
from ragnarok.aim.aimers import FeedbackAimer


def test_creep_disabled_by_default_matches_p_controller():
    a = FeedbackAimer(kp=1.0, max_step_px=1000.0, ema_alpha=1.0)   # creep_px defaults 0
    dx, _ = a.step((0.0, 0.0), (5.0, 0.0), 0.016)
    assert abs(dx - 5.0) < 1e-9                                     # full kp*error, no creep


def test_creep_zone_quadratic_falloff_near_target():
    a = FeedbackAimer(kp=1.0, max_step_px=1000.0, ema_alpha=1.0, creep_px=10.0)
    dx, _ = a.step((0.0, 0.0), (20.0, 0.0), 0.016)                 # >creep -> no damping
    assert abs(dx - 20.0) < 1e-9                                    # (20/10)^2=4 clamped to 1
    a.reset()
    dx, _ = a.step((0.0, 0.0), (5.0, 0.0), 0.016)                  # within creep
    assert abs(dx - 5.0 * (5.0 / 10.0) ** 2) < 1e-9                # 5 * 0.25 = 1.25


def test_creep_is_per_axis():
    a = FeedbackAimer(kp=1.0, max_step_px=1000.0, ema_alpha=1.0, creep_px=10.0)
    dx, dy = a.step((0.0, 0.0), (20.0, 4.0), 0.016)               # x far, y inside creep
    assert abs(dx - 20.0) < 1e-9                                    # x: no damping
    assert abs(dy - 4.0 * (4.0 / 10.0) ** 2) < 1e-9               # y: 4 * 0.16 = 0.64


def test_sign_flip_zeroes_integral():
    a = FeedbackAimer(kp=0.5, max_step_px=1000.0, ema_alpha=1.0, ki=1.0)
    a.step((0.0, 0.0), (10.0, 0.0), 0.1)          # +error -> integral goes positive
    assert a._ix > 0.0
    a.step((0.0, 0.0), (-10.0, 0.0), 0.1)         # error reverses -> integral reset then re-integrated
    # reset to 0 then += (-10)*0.1 = -1.0; WITHOUT the reset it would be ~0.0
    assert abs(a._ix - (-1.0)) < 1e-9
