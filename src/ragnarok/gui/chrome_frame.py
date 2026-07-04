"""Cyberpunk chrome container: wraps a child widget and paints corner brackets,
a left accent bar, and a notched angular border around it (spec §10.1).

Thin QPainter shell over the pure ``chrome`` geometry. Construction + a render
smoke are offscreen-testable; the exact pixels are a box-only visual detail.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ragnarok.gui import theme
from ragnarok.gui.chrome import (
    accent_bar_rect, corner_bracket_segments, notch_polygon)


class ChromeFrame(QWidget):
    def __init__(self, content: QWidget, *, arm: int = 16, accent_w: int = 4,
                 margin: int = 10, cut: int = 12) -> None:
        super().__init__()
        self._arm = arm
        self._accent_w = accent_w
        self._margin = margin
        self._cut = cut
        layout = QVBoxLayout(self)
        # leave room for the accent bar on the left + brackets all around
        layout.setContentsMargins(margin + accent_w, margin, margin, margin)
        layout.addWidget(content)

    def paintEvent(self, event) -> None:
        m = self._margin
        x0, y0 = float(m), float(m)
        x1, y1 = float(self.width() - m), float(self.height() - m)
        if x1 <= x0 or y1 <= y0:
            return
        yellow = QColor(theme.ELECTRIC_YELLOW)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            # notched angular border
            border = QColor(theme.BORDER)
            p.setPen(QPen(border, 1))
            p.setBrush(Qt.NoBrush)
            poly = notch_polygon(x0, y0, x1, y1, self._cut)
            p.drawPolygon(QPolygonF([QPointF(px, py) for px, py in poly]))

            # left accent bar (filled)
            bx, by, bw, bh = accent_bar_rect(x0, y0, x1, y1, self._accent_w)
            p.fillRect(int(bx), int(by), int(bw), int(bh), yellow)

            # corner brackets
            p.setPen(QPen(yellow, 2))
            for (a, b) in corner_bracket_segments(x0, y0, x1, y1, self._arm):
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
        finally:
            p.end()
