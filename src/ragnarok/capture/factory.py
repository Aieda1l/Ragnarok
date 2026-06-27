from __future__ import annotations
from ragnarok.config.schema import CaptureConfig
from ragnarok.capture.base import Capturer

def _screen_size() -> tuple[int, int]:
    from PySide6.QtGui import QGuiApplication
    app = QGuiApplication.instance()
    if app is not None:
        geo = QGuiApplication.primaryScreen().geometry()
        return (geo.width(), geo.height())
    return (1920, 1080)  # safe default before a QApplication exists

def create_capturer(config: CaptureConfig) -> Capturer:
    size = _screen_size()
    if config.backend == "bettercam":
        from ragnarok.capture.bettercam_capturer import BetterCamCapturer
        return BetterCamCapturer(config, size)
    from ragnarok.capture.mss_capturer import MssCapturer
    return MssCapturer(config, size)
