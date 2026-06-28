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
