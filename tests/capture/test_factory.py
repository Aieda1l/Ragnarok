from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.factory import create_capturer
from ragnarok.capture.mss_capturer import MssCapturer

def test_factory_returns_mss_for_mss_backend():
    cap = create_capturer(CaptureConfig(backend="mss"))
    assert isinstance(cap, MssCapturer)
