import math
import numpy as np
from ragnarok.core.types import Frame
from ragnarok.tracking.egomotion import CommandedMotionBuffer, FeedForwardGMC
from ragnarok.aim.fov import focal_length_px


def test_buffer_integrates_window():
    b = CommandedMotionBuffer()
    b.push(100, 1.0, 2.0)
    b.push(200, 3.0, 4.0)
    b.push(300, 5.0, 6.0)
    assert b.integrate(150, 250) == (3.0, 4.0)          # only t=200 in window
    assert b.integrate(100, 300) == (9.0, 12.0)         # all three


def test_identity_when_no_commanded_motion():
    g = FeedForwardGMC(hfov_deg=90.0, screen_width_px=1920, deg_per_count=0.02)
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=1_000_000_000, region=(0, 0, 4, 4))
    aff = g.estimate(frame)
    assert np.allclose(aff, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))


def test_none_frame_is_identity():
    g = FeedForwardGMC(hfov_deg=90.0, screen_width_px=1920, deg_per_count=0.02)
    assert np.allclose(g.estimate(None), np.eye(2, 3, dtype=np.float32))


def test_rightward_pan_translates_world_left():
    # Commanded +X counts => camera yaws right => static world shifts left on screen
    # => translation t_x should be negative.
    buf = CommandedMotionBuffer()
    g = FeedForwardGMC(hfov_deg=90.0, screen_width_px=1920, deg_per_count=0.02,
                       tau_render_s=0.0, frame_dt_s=0.01, buffer=buf)
    t_cap = 1_000_000_000
    buf.push(t_cap - 5_000_000, 100.0, 0.0)             # inside [t_cap-0.01, t_cap]
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=t_cap, region=(0, 0, 4, 4))
    aff = g.estimate(frame)
    yaw = math.radians(100.0 * 0.02)
    expected_tx = -focal_length_px(90.0, 1920) * math.tan(yaw)
    assert abs(aff[0, 2] - expected_tx) < 1e-3
    assert abs(aff[1, 2]) < 1e-6
    assert aff[0, 2] < 0.0
