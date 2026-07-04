from __future__ import annotations
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget, QMainWindow
from ragnarok.telemetry.snapshot import SnapshotPublisher

class MainWindow(QMainWindow):
    def __init__(self, publisher: SnapshotPublisher, controls: QWidget | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Ragnarok")
        self._pub = publisher
        central = QWidget()
        layout = QVBoxLayout(central)
        self.preview_label = QLabel("no signal")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.stats_label = QLabel("--")
        self.stats_label.setObjectName("mono")   # monospaced telemetry numerals (theme)
        layout.addWidget(self.preview_label)
        layout.addWidget(self.stats_label)
        self.controls = controls
        if controls is not None:
            layout.addWidget(controls)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 Hz
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    def refresh(self) -> None:
        snap = self._pub.latest()
        if snap is None:
            return
        self.stats_label.setText(
            f"FPS {snap.fps:.1f}  |  loop p50 {snap.loop_ms_p50:.1f} ms  "
            f"p99 {snap.loop_ms_p99:.1f} ms  |  detections {snap.detection_count}"
        )
        if snap.preview is not None:
            img = snap.preview
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888).copy()
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))
