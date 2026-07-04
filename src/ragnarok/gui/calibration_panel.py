"""Calibration Wizards tab (spec §10.3, §11).

Sensitivity / GMC wizard: enter a known calibration turn (mouse counts moved +
degrees the view rotated) and apply deg_per_count + sensitivity. The τ_render
auto-collection (optical flow) is box-only; the Tracking tab has a manual
tau_render_s field meanwhile. Full Cyberpunk styling is a later box-only pass.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ragnarok.gui.calibration_model import apply_sensitivity


class CalibrationPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, handle) -> None:
        super().__init__()
        self._handle = handle
        self._spins: dict[str, QDoubleSpinBox] = {}

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Sensitivity / GMC calibration: do a known in-game turn against a\n"
            "static reference, then enter the mouse counts moved and the degrees\n"
            "the view rotated (negative if inverted)."))
        form = QFormLayout()
        for key, label, lo, hi, step, default in (
            ("counts", "Mouse counts moved", -1e7, 1e7, 10.0, 1000.0),
            ("degrees", "Degrees rotated (signed)", -3600.0, 3600.0, 1.0, 360.0),
        ):
            w = QDoubleSpinBox()
            w.setDecimals(2)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(default)
            self._spins[key] = w
            form.addRow(label, w)
        root.addLayout(form)

        self.result_label = QLabel("deg/count —")
        root.addWidget(self.result_label)

        solve = QPushButton("Solve & apply")
        solve.clicked.connect(self._solve)
        root.addWidget(solve)

    def widget_for(self, key: str) -> QDoubleSpinBox:
        return self._spins[key]

    def _solve(self) -> None:
        counts = self._spins["counts"].value()
        degrees = self._spins["degrees"].value()
        try:
            new_cfg = apply_sensitivity(self._handle, total_counts=counts,
                                        measured_deg=degrees)
        except Exception as exc:  # noqa: BLE001 — bad input (zero counts / zero turn)
            self.result_label.setText(f"invalid calibration: {exc}")
            return
        self.result_label.setText(
            f"deg/count {new_cfg.tracking.deg_per_count:.4g}  "
            f"(sensitivity {new_cfg.aim.sensitivity:.4g})")
        self.configChanged.emit(new_cfg)
