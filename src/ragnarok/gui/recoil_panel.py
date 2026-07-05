"""Recoil tab: per-weapon spray-pattern editor (spec §10.3, §11).

Enabled toggle + scale slider + an editable cumulative pattern (one ``dx dy`` px
line per shot). Apply funnels through ConfigHandle.swap + configChanged like the
other panels. The wall-learner that auto-captures a pattern (recoil.learner) is
box-only (needs firing + optical flow); this edits/loads it manually meanwhile.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from ragnarok.gui.segmented_toggle import SegmentedToggle
from ragnarok.gui.cyber_slider import CyberSlider
from ragnarok.gui.recoil_model import apply_recoil, format_pattern_text, parse_pattern_text


class RecoilPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, handle) -> None:
        super().__init__()
        self._handle = handle
        r = handle.current.recoil

        root = QVBoxLayout(self)
        header = QLabel("RECOIL")
        header.setObjectName("header")
        root.addWidget(header)

        form = QFormLayout()
        self.enabled = SegmentedToggle(bool(r.enabled))
        self.scale = CyberSlider()
        self.scale.setMinimum(0.0)
        self.scale.setMaximum(5.0)
        self.scale.setSingleStep(0.1)
        self.scale.setValue(float(r.scale))
        form.addRow("Recoil comp enabled", self.enabled)
        form.addRow("Recoil scale", self.scale)
        root.addLayout(form)

        root.addWidget(QLabel(
            "Spray pattern — cumulative crosshair drift (px), one 'dx dy' per shot:"))
        self.pattern_edit = QPlainTextEdit(format_pattern_text(r.pattern))
        root.addWidget(self.pattern_edit)

        apply_btn = QPushButton("Apply recoil")
        apply_btn.clicked.connect(self._apply)
        root.addWidget(apply_btn)

    def _apply(self) -> None:
        points = parse_pattern_text(self.pattern_edit.toPlainText())
        new_cfg = apply_recoil(self._handle, points, scale=self.scale.value(),
                               enabled=self.enabled.isChecked())
        self.configChanged.emit(new_cfg)
