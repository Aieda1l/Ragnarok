"""Config -> concrete component builders (the live-wiring follow-up).

Keeps the heavy/optional imports (vendored BoT-SORT, OpenCV HSV classifier) out
of ``app.py`` and importable/testable without PySide6 or a GPU. ``app.py`` calls
these to turn a frozen ``AppConfig`` into the real ``Tracker`` and friend/foe
``FriendFoeClassifier`` that make the aim core engage against the live game.
"""
from __future__ import annotations

from ragnarok.config.schema import AppConfig
from ragnarok.tracking.base import Tracker, IdentityTracker
from ragnarok.classification.base import FriendFoeClassifier, NullClassifier


def build_tracker(cfg: AppConfig) -> Tracker:
    t = cfg.tracking
    if t.backend == "identity":
        return IdentityTracker()
    from ragnarok.tracking.botsort import BotSortTracker
    return BotSortTracker(
        track_high_thresh=t.track_high_thresh,
        track_low_thresh=t.track_low_thresh,
        new_track_thresh=t.new_track_thresh,
        track_buffer=t.track_buffer,
        match_thresh=t.match_thresh,
        proximity_thresh=t.proximity_thresh,
        frame_rate=cfg.capture.target_fps,
    )


def build_classifier(cfg: AppConfig) -> FriendFoeClassifier:
    c = cfg.classification
    if not c.enabled:
        return NullClassifier()
    from ragnarok.classification.base import HSVRingClassifier
    from ragnarok.classification.color import resolve_enemy_profile
    profile = resolve_enemy_profile(c.palette, c.enemy_color)
    return HSVRingClassifier(
        profile,
        frac_threshold=c.frac_threshold,
        thickness=c.thickness,
        vote_window=c.vote_window,
        vote_min=c.vote_min,
    )
