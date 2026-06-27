import numpy as np
from types import SimpleNamespace
from ragnarok.detection.base import to_detections

def test_to_detections_maps_fields():
    sv = SimpleNamespace(
        xyxy=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
    )
    dets = to_detections(sv)
    assert len(dets) == 2
    assert list(dets)[0].xyxy == (1.0, 2.0, 3.0, 4.0)
    assert list(dets)[1].confidence == 0.8

def test_to_detections_empty():
    sv = SimpleNamespace(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([]))
    assert len(to_detections(sv)) == 0
