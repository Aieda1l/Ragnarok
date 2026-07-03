"""Live aim tuning panel (spec §10.3 "Aim" tab, §13 snapshot-swap).

Thin Qt shell over the pure binding layer (``tuning_model``). Each field becomes
a labelled row; committing a row funnels through ``apply_field`` -> the
``ConfigHandle`` swaps -> ``configChanged`` fires so ``app.py`` can hot-reload
the worker. Full Cyberpunk styling is a later box-only pass.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QPushButton, QVBoxLayout,
    QWidget,
)

from ragnarok.gui.tuning_model import AIM_FIELDS, apply_field, get_field


class TuningPanel(QWidget):
    configChanged = Signal(object)          # emits the new AppConfig

    def __init__(self, handle, fields=AIM_FIELDS, *, on_save=None) -> None:
        super().__init__()
        self._handle = handle
        self._fields = tuple(fields)
        self._on_save = on_save
        self._widgets: dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        form = QFormLayout()
        cfg = handle.current
        for spec in self._fields:
            w = self._build_widget(spec, get_field(cfg, spec.path))
            self._widgets[spec.path] = w
            form.addRow(spec.label, w)
        root.addLayout(form)

        save = QPushButton("Save to config")
        save.clicked.connect(self._save)
        root.addWidget(save)

    # -- public ----------------------------------------------------------
    def widget_for(self, path: str) -> QWidget:
        return self._widgets[path]

    def refresh(self) -> None:
        """Repaint every widget from the live config (e.g. after a profile load).

        Signals are blocked so setting a value does not re-fire _commit (which
        would feed back into ConfigHandle / trigger a needless worker reload)."""
        cfg = self._handle.current
        for path, w in self._widgets.items():
            value = get_field(cfg, path)
            w.blockSignals(True)
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(value))
                else:
                    self._fit_range(w, value)
                    w.setValue(float(value))
            finally:
                w.blockSignals(False)

    @staticmethod
    def _fit_range(w, value) -> None:
        """Expand the spin box range to include a loaded value rather than
        clamping it. Some fields are schema-unbounded beyond the default GUI
        range; a profile/config value past the cap must display (and re-commit)
        faithfully, not be silently downgraded."""
        v = float(value)
        if v < w.minimum():
            w.setMinimum(v)
        if v > w.maximum():
            w.setMaximum(v)

    # -- construction ----------------------------------------------------
    def _build_widget(self, spec, value) -> QWidget:
        path = spec.path
        if spec.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.stateChanged.connect(lambda _s, p=path: self._commit(p))
            return w
        if spec.kind == "choice":
            w = QComboBox()
            w.addItems(list(spec.choices))
            w.setCurrentText(str(value))
            w.currentIndexChanged.connect(lambda _i, p=path: self._commit(p))
            return w
        # float / int -> spin box
        w = QDoubleSpinBox()
        w.setDecimals(0 if spec.kind == "int" else 3)
        if spec.minimum is not None:
            w.setMinimum(spec.minimum)
        if spec.maximum is not None:
            w.setMaximum(spec.maximum)
        if spec.step is not None:
            w.setSingleStep(spec.step)
        self._fit_range(w, value)                       # never clamp a loaded value
        w.setValue(float(value))
        w.editingFinished.connect(lambda p=path: self._commit(p))
        return w

    # -- commit ----------------------------------------------------------
    def _read(self, path: str):
        w = self._widgets[path]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.value()

    def _commit(self, path: str) -> None:
        value = self._read(path)
        # editingFinished fires on focus-out even without an edit; skip no-op
        # commits so a mere focus change never swaps config or reloads the worker.
        if value == get_field(self._handle.current, path):
            return
        new_cfg = apply_field(self._handle, path, value)
        self.configChanged.emit(new_cfg)

    def _save(self) -> None:
        if self._on_save is not None:
            self._on_save(self._handle.current)
