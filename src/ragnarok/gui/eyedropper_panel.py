"""Enemy-colour eyedropper tab (spec §11).

Shows the live capture preview; click the enemy's outline to sample its colour
into a custom HSV band (classification.custom_band), which overrides the palette
so any in-game colour can be matched. The sample->band->config apply + clear are
offscreen-testable; the preview render + click mapping are box-only.
"""
from __future__ import annotations

from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ragnarok.classification.eyedropper import hsv_band_from_bgr


class EyedropperPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, handle, publisher) -> None:
        super().__init__()
        self._handle = handle
        self._pub = publisher
        self._img = None                 # latest preview BGR ndarray

        root = QVBoxLayout(self)
        header = QLabel("ENEMY COLOR EYEDROPPER")
        header.setObjectName("header")
        root.addWidget(header)
        root.addWidget(QLabel("Click the enemy's outline in the preview to set a custom color."))
        self.preview = QLabel("(no preview)")
        self.preview.setMinimumSize(240, 135)
        self.preview.mousePressEvent = self._on_click        # box-only
        root.addWidget(self.preview)
        self.result = QLabel("")
        root.addWidget(self.result)
        clear = QPushButton("Clear (use palette)")
        clear.clicked.connect(self.clear)
        root.addWidget(clear)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ---- testable apply/clear ----------------------------------------------
    def apply_sample(self, bgr) -> None:
        band = hsv_band_from_bgr(bgr)
        self._swap_band(band)
        self.result.setText(f"custom HSV band set: {band}")

    def clear(self) -> None:
        self._swap_band(None)
        self.result.setText("custom color cleared (using palette)")

    def _swap_band(self, band) -> None:
        cfg = self._handle.current
        new_c = cfg.classification.__class__(**{**cfg.classification.model_dump(),
                                                "custom_band": band})
        new = cfg.model_copy(update={"classification": new_c})
        self._handle.swap(new)
        self.configChanged.emit(new)

    # ---- box-only preview render + click sampling --------------------------
    def _refresh(self) -> None:  # pragma: no cover — box-only rendering
        snap = self._pub.latest()
        if snap is None or snap.preview is None:
            return
        self._img = snap.preview
        from PySide6.QtGui import QImage, QPixmap
        h, w = self._img.shape[:2]
        rgb = self._img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview.width(), self.preview.height()))

    def _on_click(self, event) -> None:  # pragma: no cover — box-only
        if self._img is None:
            return
        h, w = self._img.shape[:2]
        px = int(event.position().x() / max(1, self.preview.width()) * w)
        py = int(event.position().y() / max(1, self.preview.height()) * h)
        px, py = max(0, min(w - 1, px)), max(0, min(h - 1, py))
        self.apply_sample(tuple(int(c) for c in self._img[py, px]))
