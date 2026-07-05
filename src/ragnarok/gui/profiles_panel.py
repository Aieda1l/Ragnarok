"""Config profiles tab (per-weapon/per-game presets, spec §13).

Thin Qt shell over ProfileStore. Load funnels through ConfigHandle.swap +
configChanged (like the settings panels) so app.py refreshes the tabs and
hot-reloads the worker. Save persists handle.current under a chosen name.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)

from ragnarok.config.portable import export_config, import_config


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

        row3 = QHBoxLayout()
        imp = QPushButton("Import…")
        imp.clicked.connect(self._import_dialog)
        exp = QPushButton("Export…")
        exp.clicked.connect(self._export_dialog)
        row3.addWidget(imp)
        row3.addWidget(exp)
        root.addLayout(row3)

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

    # --- import/export: testable path methods + box-only file dialogs ---
    def import_path(self, path) -> None:
        """Import a config from ``path`` and make it live (swap + configChanged)."""
        cfg = import_config(path)
        self._handle.swap(cfg)
        self.configChanged.emit(cfg)

    def export_path(self, path) -> None:
        """Write the live config to ``path``."""
        export_config(self._handle.current, path)

    def _import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import config", "", "TOML (*.toml)")
        if path:
            self.import_path(path)

    def _export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export config", "ragnarok.toml",
                                              "TOML (*.toml)")
        if path:
            self.export_path(path)
