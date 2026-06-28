"""Tests for Team, Track, Tracks types added in Phase 2."""
from ragnarok.core.types import Detection, Team, Track, Tracks


def test_team_values():
    assert Team.UNKNOWN.value == "unknown"
    assert Team.ENEMY.value == "enemy"
    assert Team.TEAMMATE.value == "teammate"


def test_track_from_detection_center():
    det = Detection(xyxy=(0.0, 0.0, 10.0, 20.0), confidence=0.9, class_id=0)
    track = Track.from_detection(det, track_id=5)
    assert track.center == (5.0, 10.0)
    assert track.track_id == 5
    assert track.team == Team.UNKNOWN


def test_track_from_detection_defaults():
    det = Detection(xyxy=(0.0, 0.0, 10.0, 20.0), confidence=0.9, class_id=0)
    track = Track.from_detection(det, track_id=5)
    assert track.age == 0
    assert track.hits == 1
    assert track.time_since_update == 0
    assert track.confidence == 0.9
    assert track.class_id == 0
    assert track.xyxy == (0.0, 0.0, 10.0, 20.0)


def test_track_team_override():
    det = Detection(xyxy=(1.0, 2.0, 3.0, 4.0), confidence=0.5, class_id=1)
    track = Track.from_detection(det, track_id=7, team=Team.ENEMY)
    assert track.team == Team.ENEMY


def test_tracks_empty():
    tracks = Tracks.empty()
    assert len(tracks) == 0


def test_tracks_with_item():
    det = Detection(xyxy=(0.0, 0.0, 10.0, 20.0), confidence=0.9, class_id=0)
    track = Track.from_detection(det, track_id=1)
    tracks = Tracks((track,))
    assert len(tracks) == 1
    assert list(tracks) == [track]


def test_track_is_frozen():
    det = Detection(xyxy=(0.0, 0.0, 10.0, 20.0), confidence=0.9, class_id=0)
    track = Track.from_detection(det, track_id=1)
    import pytest
    with pytest.raises((AttributeError, TypeError)):
        track.track_id = 99  # type: ignore
