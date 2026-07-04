"""CP2077-style horizontal value slider (spec §10.1).

A dark rounded track with a red fill up to the value, a red handle block, and the
value shown in cyan — like the Cyberpunk 2077 settings sliders. Exposes a
QDoubleSpinBox-compatible surface (``value``/``setValue``/``minimum``/``maximum``/
``setMinimum``/``setMaximum``/``setSingleStep``/``setDecimals`` + an
``editingFinished`` signal) so it drops into ``TuningPanel`` for float fields.

``setValue`` never emits ``editingFinished`` (only a user drag-release does), so
programmatic refresh from a config swap can't re-commit.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from ragnarok.gui import theme


class CyberSlider(QWidget):
    editingFinished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._min = 0.0
        self._max = 1.0
        self._step = 0.01
        self._decimals = 3
        self._value = 0.0
        self._dragging = False
        self.setFixedHeight(26)
        self.setMinimumWidth(170)
        self.setCursor(Qt.PointingHandCursor)

    # -- QDoubleSpinBox-compatible surface ------------------------------
    def minimum(self) -> float:
        return self._min

    def maximum(self) -> float:
        return self._max

    def setMinimum(self, v) -> None:
        self._min = float(v)
        self._value = self._clamp(self._value)

    def setMaximum(self, v) -> None:
        self._max = float(v)
        self._value = self._clamp(self._value)

    def setSingleStep(self, v) -> None:
        if v:
            self._step = float(v)

    def setDecimals(self, d) -> None:
        self._decimals = int(d)

    def value(self) -> float:
        return self._value

    def setValue(self, v) -> None:
        self._value = self._clamp(float(v))        # no emit -> refresh-safe
        self.update()

    # -- helpers --------------------------------------------------------
    def _clamp(self, v: float) -> float:
        return max(self._min, min(self._max, v))

    def _frac(self) -> float:
        span = self._max - self._min
        return 0.0 if span <= 0.0 else (self._value - self._min) / span

    def _value_from_x(self, x: float) -> float:
        w = max(1, self.width())
        frac = max(0.0, min(1.0, x / w))
        v = self._min + frac * (self._max - self._min)
        if self._step > 0.0:                        # snap to step on drag
            v = self._min + round((v - self._min) / self._step) * self._step
        return self._clamp(v)

    # -- interaction ----------------------------------------------------
    def mousePressEvent(self, event) -> None:
        self._dragging = True
        self.setValue(self._value_from_x(event.position().x()))

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            self.setValue(self._value_from_x(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self.editingFinished.emit()

    # -- paint ----------------------------------------------------------
    def paintEvent(self, event) -> None:
        w, h = float(self.width()), float(self.height())
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            track = QRectF(0, 0, w, h)
            p.fillRect(track, QColor(theme.PANEL_ALT))
            fill_w = self._frac() * w
            if fill_w > 0:
                fill = QColor(theme.RED)
                fill.setAlpha(150)
                p.fillRect(QRectF(0, 0, fill_w, h), fill)
                p.fillRect(QRectF(max(0.0, fill_w - 4.0), 0, 4.0, h),
                           QColor(theme.RED))        # handle block at the fill edge
            p.setPen(QColor(theme.BORDER))
            p.drawRect(track.adjusted(0, 0, -1, -1))
            p.setPen(QColor(theme.CYAN))             # value read-out
            p.drawText(track, Qt.AlignCenter, f"{self._value:.{self._decimals}f}")
        finally:
            p.end()
