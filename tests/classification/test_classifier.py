"""Tests for FriendFoeClassifier ABC, NullClassifier, and HSVRingClassifier."""
from __future__ import annotations

import numpy as np
import pytest

from ragnarok.classification.base import FriendFoeClassifier, HSVRingClassifier, NullClassifier
from ragnarok.classification.color import DEFAULT_ENEMY_PROFILES
from ragnarok.core.types import Detection, Detections, Team, Track, Tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(track_id: int, xyxy=(5.0, 5.0, 55.0, 55.0), team=Team.UNKNOWN) -> Track:
    det = Detection(xyxy=xyxy, confidence=0.9, class_id=0)
    return Track.from_detection(det, track_id, team=team)


def _enemy_frame(size: int = 60, xyxy=(5.0, 5.0, 55.0, 55.0), thickness: int = 6) -> np.ndarray:
    """BGR image with a yellow ring — should trigger YELLOW profile."""
    from ragnarok.classification.color import ring_mask

    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = ring_mask((size, size), xyxy, thickness=thickness)
    img[mask] = (0, 255, 255)  # BGR yellow
    return img


def _neutral_frame(size: int = 60) -> np.ndarray:
    """All-grey frame — no profile should trigger."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# NullClassifier
# ---------------------------------------------------------------------------

class TestNullClassifier:
    def test_returns_tracks_unchanged(self):
        nc = NullClassifier()
        track = _make_track(1)
        tracks = Tracks((track,))
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        result = nc.classify(tracks, frame)
        assert result is tracks or result == tracks

    def test_empty_tracks(self):
        nc = NullClassifier()
        result = nc.classify(Tracks.empty(), np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(result) == 0

    def test_is_friend_foe_classifier(self):
        assert isinstance(NullClassifier(), FriendFoeClassifier)


# ---------------------------------------------------------------------------
# HSVRingClassifier
# ---------------------------------------------------------------------------

class TestHSVRingClassifier:
    XYXY = (5.0, 5.0, 55.0, 55.0)
    THICKNESS = 6

    def test_enemy_track_flips_on_frame_3(self):
        """With vote_min=3, team flips to ENEMY exactly on the third matching frame."""
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        clf = HSVRingClassifier(profile=profile, vote_window=5, vote_min=3,
                                thickness=self.THICKNESS)
        track = _make_track(1, xyxy=self.XYXY)
        tracks = Tracks((track,))
        frame = _enemy_frame(xyxy=self.XYXY, thickness=self.THICKNESS)

        # Frame 1 — still UNKNOWN
        result1 = clf.classify(tracks, frame)
        assert result1.items[0].team == Team.UNKNOWN, "Should still be UNKNOWN after frame 1"

        # Frame 2 — still UNKNOWN
        result2 = clf.classify(tracks, frame)
        assert result2.items[0].team == Team.UNKNOWN, "Should still be UNKNOWN after frame 2"

        # Frame 3 — now ENEMY
        result3 = clf.classify(tracks, frame)
        assert result3.items[0].team == Team.ENEMY, "Should be ENEMY after frame 3"

    def test_non_matching_track_stays_unknown(self):
        """A track with no enemy color should always be UNKNOWN."""
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        clf = HSVRingClassifier(profile=profile, vote_window=5, vote_min=3,
                                thickness=self.THICKNESS)
        track = _make_track(2, xyxy=self.XYXY)
        tracks = Tracks((track,))
        frame = _neutral_frame()

        for _ in range(5):
            result = clf.classify(tracks, frame)
        assert result.items[0].team == Team.UNKNOWN

    def test_prunes_dead_tracks(self):
        """After a track is gone, its vote history should be pruned."""
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        clf = HSVRingClassifier(profile=profile, vote_window=5, vote_min=3,
                                thickness=self.THICKNESS)
        track1 = _make_track(1, xyxy=self.XYXY)
        frame = _enemy_frame(xyxy=self.XYXY, thickness=self.THICKNESS)

        # Classify track 1 for 3 frames so it becomes ENEMY
        for _ in range(3):
            clf.classify(Tracks((track1,)), frame)

        # Now classify with only track 2 (track 1 gone) — vote book for id=1 pruned
        track2 = _make_track(2, xyxy=self.XYXY)
        clf.classify(Tracks((track2,)), frame)
        # Calling label directly on vote book
        assert clf._votes.label(1) == "unknown"

    def test_returns_tracks_type(self):
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        clf = HSVRingClassifier(profile=profile)
        track = _make_track(1, xyxy=self.XYXY)
        frame = _neutral_frame()
        result = clf.classify(Tracks((track,)), frame)
        assert isinstance(result, Tracks)

    def test_is_friend_foe_classifier(self):
        profile = DEFAULT_ENEMY_PROFILES["yellow"]
        assert isinstance(HSVRingClassifier(profile=profile), FriendFoeClassifier)
