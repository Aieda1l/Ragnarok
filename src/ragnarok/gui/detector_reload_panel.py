"""Reload the detector live — after retraining or swapping the TensorRT engine —
without restarting the app (spec §13). Confidence already updates live; this
rebuilds + hot-swaps the whole detector. The rebuild is box-only (loads the
engine); ``reload`` is testable with injected builder + loop.
"""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class DetectorReloadPanel(QWidget):
    def __init__(self, handle, loop, build_detector) -> None:
        super().__init__()
        self._handle = handle
        self._loop = loop
        self._build = build_detector

        root = QVBoxLayout(self)
        header = QLabel("DETECTOR")
        header.setObjectName("header")
        root.addWidget(header)
        root.addWidget(QLabel(
            "Confidence updates live. Reload after retraining or swapping the engine "
            "(set detection.engine_path first):"))
        btn = QPushButton("Reload detector")
        btn.clicked.connect(self.reload)
        root.addWidget(btn)
        self.status = QLabel("")
        root.addWidget(self.status)

    def reload(self) -> None:
        try:
            self._loop.set_detector(self._build(self._handle.current))
            self.status.setText("detector reloaded")
        except Exception as exc:  # noqa: BLE001 — box-only build can fail; keep GUI alive
            self.status.setText(f"reload failed: {exc}")
