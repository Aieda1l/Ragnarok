import numpy as np

from ragnarok.config.schema import DetectionConfig
from ragnarok.core.types import Frame, Detections
from ragnarok.detection.rfdetr_trt import RFDETRTensorRTDetector
from ragnarok.worker.loop import WorkerLoop
from ragnarok.telemetry.snapshot import SnapshotPublisher


class _Sess:
    def __init__(self):
        self.thresh = None

    def infer(self, img, *, threshold):
        self.thresh = threshold
        return ([], [], [])


def _frame():
    return Frame(np.zeros((10, 10, 3), np.uint8), t_capture_ns=0, region=(0, 0, 10, 10))


def test_set_confidence_takes_effect_in_detect():
    s = _Sess()
    d = RFDETRTensorRTDetector(DetectionConfig(confidence=0.5), session=s)
    d.detect(_frame())
    assert s.thresh == 0.5
    d.set_confidence(0.3)                        # live update, no rebuild
    d.detect(_frame())
    assert s.thresh == 0.3


class _Det:
    def __init__(self):
        self.conf = None

    def detect(self, f):
        return Detections(items=())

    def set_confidence(self, c):
        self.conf = c


class _Cap:
    def grab(self):
        return None

    def stop(self):
        pass


class _Prof:
    def record(self, *a):
        pass

    def percentiles(self, *a):
        return (0.0, 0.0)


def test_loop_set_detector_and_confidence():
    d1, d2 = _Det(), _Det()
    loop = WorkerLoop(_Cap(), d1, _Prof(), SnapshotPublisher())
    loop.set_detector_confidence(0.4)
    assert d1.conf == 0.4
    loop.set_detector(d2)                        # hot-swap the detector
    loop.set_detector_confidence(0.6)
    assert d2.conf == 0.6
