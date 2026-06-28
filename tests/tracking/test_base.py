import numpy as np

from ragnarok.core.types import Detection, Detections, Team, Tracks
from ragnarok.tracking.base import IDENTITY_AFFINE, IdentityTracker, Tracker
from ragnarok.tracking.egomotion import EgoMotion, IdentityEgoMotion


def test_identity_affine():
    assert IDENTITY_AFFINE.shape == (2, 3)
    assert np.allclose(IDENTITY_AFFINE, np.array([[1, 0, 0], [0, 1, 0]]))


def test_identity_egomotion_estimate():
    ego = IdentityEgoMotion()
    H = ego.estimate(object())
    assert H.shape == (2, 3)
    assert np.allclose(H, np.eye(2, 3))
    assert isinstance(ego, EgoMotion)


def test_identity_tracker_echoes_detection():
    tr = IdentityTracker()
    assert isinstance(tr, Tracker)
    det = Detection((0.0, 0.0, 10.0, 20.0), 0.9, 0)
    out = tr.update(Detections((det,)))
    assert isinstance(out, Tracks)
    assert len(out) == 1
    t = out.items[0]
    assert t.xyxy == (0.0, 0.0, 10.0, 20.0)
    assert t.confidence == 0.9
    assert t.class_id == 0
    assert t.team == Team.UNKNOWN


def test_identity_tracker_increments_ids():
    tr = IdentityTracker()
    d0 = Detection((0.0, 0.0, 1.0, 1.0), 0.5, 0)
    d1 = Detection((2.0, 2.0, 3.0, 3.0), 0.5, 1)
    out = tr.update(Detections((d0, d1)))
    ids = [t.track_id for t in out]
    assert len(set(ids)) == 2


def test_identity_tracker_empty():
    tr = IdentityTracker()
    out = tr.update(Detections.empty())
    assert len(out) == 0
