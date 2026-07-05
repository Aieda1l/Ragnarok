"""Custom frameless-window title bar (spec §10.1).

A CP2077-styled bar: cyan title on the left, red min / max / close buttons on the
right (close goes red on hover), drag-to-move, double-click to maximise. Drives an
injected window so it is offscreen-testable without a real window manager.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class TitleBar(QWidget):
    def __init__(self, window, title: str = "RAGNAROK") -> None:
        super().__init__()
        self._win = window
        self._drag_offset = None
        self.setFixedHeight(34)

        row = QHBoxLayout(self)
        row.setContentsMargins(12, 0, 6, 0)
        row.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("titlebar")
        row.addWidget(self.title_label)
        row.addStretch(1)

        self.btn_min = QPushButton("—")
        self.btn_max = QPushButton("▢")
        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("titleclose")
        for b, slot in ((self.btn_min, self._minimize),
                        (self.btn_max, self.toggle_maximize),
                        (self.btn_close, self._close)):
            b.setFixedSize(34, 24)
            b.setFlat(True)
            b.clicked.connect(slot)
            row.addWidget(b)

    # -- window controls ------------------------------------------------
    def _minimize(self) -> None:
        self._win.showMinimized()

    def toggle_maximize(self) -> None:
        if self._win.isMaximized():
            self._win.showNormal()
        else:
            self._win.showMaximized()

    def _close(self) -> None:
        self._win.close()

    # -- drag to move ---------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self._win.frameGeometry().topLeft())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            if not self._win.isMaximized():
                self._win.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None

    def mouseDoubleClickEvent(self, event) -> None:
        self.toggle_maximize()
