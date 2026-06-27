import numpy as np
from ragnarok.core.types import Frame, Detection, Detections

def test_frame_holds_image_and_timestamp():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    f = Frame(image=img, t_capture_ns=123, region=(0, 0, 4, 4))
    assert f.t_capture_ns == 123
    assert f.image.shape == (4, 4, 3)

def test_detection_center():
    d = Detection(xyxy=(10.0, 20.0, 30.0, 60.0), confidence=0.9, class_id=0)
    assert d.center == (20.0, 40.0)

def test_detections_container():
    empty = Detections.empty()
    assert len(empty) == 0
    one = Detections(items=(Detection((0, 0, 2, 2), 0.5, 0),))
    assert len(one) == 1
    assert list(one)[0].confidence == 0.5
