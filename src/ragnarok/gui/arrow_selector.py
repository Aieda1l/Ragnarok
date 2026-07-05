"""CP2077-style ◁ value ▷ selector (spec §10.1).

Red prev/next arrows around a cyan value read-out, cycling a fixed list of
choices — like the Cyberpunk 2077 settings selectors. Exposes a
QComboBox-compatible surface (``addItems``/``currentText``/``setCurrentText`` +
``currentIndexChanged``) so it drops into ``TuningPanel`` for choice fields.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ArrowSelector(QWidget):
    currentIndexChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._items: list[str] = []
        self._index = 0

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self._prev = QPushButton("◁")
        self._next = QPushButton("▷")
        for b in (self._prev, self._next):
            b.setFixedWidth(28)
            b.setFlat(True)
        self._label = QLabel("")
        self._label.setObjectName("mono")            # cyan value read-out
        self._label.setAlignment(Qt.AlignCenter)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(+1))
        row.addWidget(self._prev)
        row.addWidget(self._label, 1)
        row.addWidget(self._next)

    # -- QComboBox-compatible surface -----------------------------------
    def addItems(self, items) -> None:
        self._items = list(items)
        self._index = 0
        self._refresh()

    def currentText(self) -> str:
        return self._items[self._index] if self._items else ""

    def setCurrentText(self, text) -> None:
        text = str(text)
        if text in self._items:
            i = self._items.index(text)
            changed = i != self._index
            self._index = i
            self._refresh()
            if changed:
                self.currentIndexChanged.emit(i)     # blockSignals() suppresses this

    # -- interaction ----------------------------------------------------
    def _step(self, delta: int) -> None:
        new = self._index + delta
        if 0 <= new < len(self._items):
            self._index = new
            self._refresh()
            self.currentIndexChanged.emit(new)

    def _refresh(self) -> None:
        self._label.setText(self.currentText())
        self._prev.setEnabled(self._index > 0)
        self._next.setEnabled(self._index < len(self._items) - 1)
