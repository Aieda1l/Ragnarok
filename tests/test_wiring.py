"""Tests for the live-wiring builders: config -> real Tracker / Classifier."""
from __future__ import annotations

from ragnarok.config.schema import AppConfig
from ragnarok.wiring import build_tracker, build_classifier
from ragnarok.tracking.base import IdentityTracker
from ragnarok.tracking.botsort import BotSortTracker
from ragnarok.classification.base import NullClassifier, HSVRingClassifier


def test_build_tracker_default_is_botsort():
    assert isinstance(build_tracker(AppConfig()), BotSortTracker)


def test_build_tracker_identity_backend():
    cfg = AppConfig(tracking={"backend": "identity"})
    assert isinstance(build_tracker(cfg), IdentityTracker)


def test_build_tracker_uses_capture_fps_as_frame_rate():
    # frame_rate flows into the vendored core's buffer_size = frame_rate/30 * track_buffer
    cfg = AppConfig(capture={"target_fps": 240})  # default track_buffer=30 -> buffer_size 240
    trk = build_tracker(cfg)
    assert trk._core.buffer_size == 240


def test_build_classifier_default_is_hsv_ring():
    assert isinstance(build_classifier(AppConfig()), HSVRingClassifier)


def test_build_classifier_disabled_is_null():
    cfg = AppConfig(classification={"enabled": False})
    assert isinstance(build_classifier(cfg), NullClassifier)


def test_build_classifier_resolves_profile():
    cfg = AppConfig(classification={"palette": "wong", "enemy_color": "sky_blue"})
    clf = build_classifier(cfg)
    assert clf._profile.name == "sky_blue"


# ---------------------------------------------------------------------------
# Phase 4 builder tests
# ---------------------------------------------------------------------------
from ragnarok.wiring import build_aimer, build_shaper, build_recoil
from ragnarok.aim.aimers import FlickAimer, FeedbackAimer, HybridAimer, PredictiveAimer
from ragnarok.motion.shaper import NullShaper, WindMouseShaper


def test_build_aimer_variants():
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "flick"})), FlickAimer)
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "feedback"})), FeedbackAimer)
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "hybrid"})), HybridAimer)
    assert isinstance(build_aimer(AppConfig(aim={"aimer": "predictive"})), PredictiveAimer)


def test_build_shaper_variants():
    assert isinstance(build_shaper(AppConfig()), NullShaper)              # default "none"
    assert isinstance(build_shaper(AppConfig(motion={"shaper": "windmouse"})), WindMouseShaper)


def test_build_recoil_disabled_is_none():
    assert build_recoil(AppConfig()) is None                              # disabled by default


def test_build_recoil_enabled():
    cfg = AppConfig(recoil={"enabled": True, "pattern": ((0.0, 0.0), (0.0, 10.0))})
    rc = build_recoil(cfg)
    assert rc is not None
    rc.on_fire()
    assert rc.on_fire() == (0.0, -10.0)


def test_build_aimer_feedback_mode_p_zeroes_gains():
    a = build_aimer(AppConfig(aim={"aimer": "feedback", "controller_mode": "p",
                                   "ki": 9.0, "kd": 9.0}))
    assert a._ki == 0.0 and a._kd == 0.0     # mode 'p' overrides the gains


def test_build_aimer_feedback_mode_pid_applies_gains():
    a = build_aimer(AppConfig(aim={"aimer": "feedback", "controller_mode": "pid",
                                   "ki": 0.2, "kd": 0.05}))
    assert a._ki == 0.2 and a._kd == 0.05
