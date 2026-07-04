"""CP2077-style segmented OFF/ON toggle (spec §10.1).

Two side-by-side segments: the active side is a filled block — RED for OFF, CYAN
for ON — the inactive side is dim, exactly like the Cyberpunk 2077 settings
toggles. Exposes a QCheckBox-compatible surface (``isChecked``/``setChecked`` +
``stateChanged``) so it drops into ``TuningPanel`` in place of a check box.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from ragnarok.gui import theme


class SegmentedToggle(QWidget):
    stateChanged = Signal(int)                     # mirrors QCheckBox (emits 0/1)

    def __init__(self, checked: bool = False) -> None:
        super().__init__()
        self._checked = bool(checked)
        self.setFixedHeight(26)
        self.setMinimumWidth(140)
        self.setCursor(Qt.PointingHandCursor)

    # -- QCheckBox-compatible surface -----------------------------------
    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value) -> None:
        value = bool(value)
        if value != self._checked:
            self._checked = value
            self.update()
            self.stateChanged.emit(int(value))     # blockSignals() suppresses this

    # -- interaction ----------------------------------------------------
    def mousePressEvent(self, event) -> None:
        # left half selects OFF, right half selects ON
        self.setChecked(event.position().x() >= self.width() / 2.0)

    # -- paint ----------------------------------------------------------
    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        half = w / 2.0
        bg = QColor(theme.BG)
        dim = QColor(theme.TEXT_DIM)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            off_rect = QRectF(0, 0, half, h)
            on_rect = QRectF(half, 0, w - half, h)
            self._segment(p, off_rect, "OFF", QColor(theme.RED), bg, dim,
                          active=not self._checked)
            self._segment(p, on_rect, "ON", QColor(theme.CYAN), bg, dim,
                          active=self._checked)
        finally:
            p.end()

    @staticmethod
    def _segment(p: QPainter, rect: QRectF, text: str, accent: QColor,
                 bg: QColor, dim: QColor, *, active: bool) -> None:
        if active:
            p.fillRect(rect, accent)
            p.setPen(bg)                           # dark text on the filled block
        else:
            p.fillRect(rect, QColor(theme.PANEL))
            p.setPen(dim)
        p.drawText(rect, Qt.AlignCenter, text)
