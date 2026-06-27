from __future__ import annotations
import sys
from pathlib import Path
import os
from PySide6.QtWidgets import QApplication
from ragnarok.config.store import load_config
from ragnarok.capture.factory import create_capturer
from ragnarok.detection.factory import create_detector
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.worker.loop import WorkerLoop
from ragnarok.gui.worker_thread import WorkerThread
from ragnarok.gui.main_window import MainWindow

def _config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Ragnarok"
    return base / "config.toml"

def main() -> int:
    app = QApplication(sys.argv)
    cfg = load_config(_config_path())
    publisher = SnapshotPublisher()
    loop = WorkerLoop(
        create_capturer(cfg.capture), create_detector(cfg.detection),
        StageProfiler(), publisher,
    )
    worker = WorkerThread(loop)
    window = MainWindow(publisher)
    window.show()
    worker.start()
    app.aboutToQuit.connect(worker.stop)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
