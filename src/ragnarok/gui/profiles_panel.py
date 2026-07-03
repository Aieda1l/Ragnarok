"""Config profiles tab (per-weapon/per-game presets, spec §13).

Thin Qt shell over ProfileStore. Load funnels through ConfigHandle.swap +
configChanged (like the settings panels) so app.py refreshes the tabs and
hot-reloads the worker. Save persists handle.current under a chosen name.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)


class ProfilesPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, store, handle) -> None:
        super().__init__()
        self._store = store
        self._handle = handle

        root = QVBoxLayout(self)
        self.combo = QComboBox()
        root.addWidget(self.combo)

        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New profile name")
        save = QPushButton("Save as")
        save.clicked.connect(self._save_as)
        row.addWidget(self.name_edit)
        row.addWidget(save)
        root.addLayout(row)

        row2 = QHBoxLayout()
        load = QPushButton("Load")
        load.clicked.connect(self._load)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete)
        row2.addWidget(load)
        row2.addWidget(delete)
        root.addLayout(row2)

        self._refresh_list()

    def _refresh_list(self) -> None:
        current = self.combo.currentText()
        self.combo.clear()
        self.combo.addItems(self._store.list())
        idx = self.combo.findText(current)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _save_as(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            return
        self._store.save(name, self._handle.current)
        self._refresh_list()
        idx = self.combo.findText(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _load(self) -> None:
        name = self.combo.currentText()
        if not name:
            return
        cfg = self._store.load(name)
        self._handle.swap(cfg)
        self.configChanged.emit(cfg)

    def _delete(self) -> None:
        name = self.combo.currentText()
        if not name:
            return
        self._store.delete(name)
        self._refresh_list()
