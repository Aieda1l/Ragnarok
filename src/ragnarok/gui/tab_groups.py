"""Stack several titled setting panels into one scrollable top-level tab.

Each panel already renders its own ``#header`` title, so stacking them under one
tab gives visually grouped sections without touching panel logic — the mechanism
for collapsing the many single-purpose field tabs into a few logical groups.
"""
from __future__ import annotations

from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget


def grouped_tab(widgets: list[QWidget]) -> QScrollArea:
    inner = QWidget()
    lay = QVBoxLayout(inner)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch(1)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    return area
