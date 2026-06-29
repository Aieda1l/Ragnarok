"""Integration: a commanded pan in the shared buffer produces a real GMC warp.

Proves the Phase 5B activation wiring end-to-end without a GPU/game: the tracker
built with gmc='feedforward' shares the AimController's CommandedMotionBuffer,
and a pushed commanded pan yields a non-identity, correctly-signed warp. The full
residual-collapse validation against a live static target is a box-only smoke.
"""
from __future__ import annotations
import math
import numpy as np
from ragnarok.config.schema import AppConfig
from ragnarok.wiring import build_tracker
from ragnarok.tracking.egomotion import CommandedMotionBuffer, FeedForwardGMC
from ragnarok.core.types import Frame
from ragnarok.aim.fov import focal_length_px


def test_commanded_pan_produces_non_identity_warp_through_shared_buffer():
    buf = CommandedMotionBuffer()
    cfg = AppConfig(
        aim={"hfov_deg": 90.0, "screen_width_px": 1920},
        capture={"target_fps": 100},
        tracking={"gmc": "feedforward", "deg_per_count": 0.02, "tau_render_s": 0.0},
    )
    tracker = build_tracker(cfg, gmc_buffer=buf)
    assert isinstance(tracker.ego, FeedForwardGMC)
    assert tracker.ego.buffer is buf

    t_cap = 1_000_000_000
    # Push a commanded rightward pan inside the GMC window [t_cap - frame_dt, t_cap].
    buf.push(t_cap - 5_000_000, 100.0, 0.0)        # frame_dt = 1/100 s = 10 ms
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=t_cap, region=(0, 0, 4, 4))
    aff = tracker.ego.estimate(frame)

    yaw = math.radians(100.0 * 0.02)
    expected_tx = -focal_length_px(90.0, 1920) * math.tan(yaw)
    assert abs(aff[0, 2] - expected_tx) < 1e-3      # correct back-projected translation
    assert aff[0, 2] < 0.0                          # rightward pan -> world shifts left
    assert abs(aff[1, 2]) < 1e-9


def test_no_commanded_motion_is_identity_warp():
    buf = CommandedMotionBuffer()
    cfg = AppConfig(tracking={"gmc": "feedforward", "deg_per_count": 0.02})
    tracker = build_tracker(cfg, gmc_buffer=buf)
    frame = Frame(np.zeros((4, 4, 3), np.uint8), t_capture_ns=1_000_000_000, region=(0, 0, 4, 4))
    aff = tracker.ego.estimate(frame)               # empty buffer
    assert np.allclose(aff, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
