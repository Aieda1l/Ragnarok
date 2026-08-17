"""Dashboard tab: rolling FPS + loop-latency sparklines (spec §10.3).

Read-only telemetry view. Pulls the latest snapshot on its own timer (dedup by
seq), pushes to a pure TelemetryHistory, and paints sparklines with QPainter over
the pure geometry (no pyqtgraph). Full Cyberpunk styling is a later box-only pass.
"""
from __future__ import annotations

from PySide6.QtCore import QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ragnarok.gui import theme
from ragnarok.gui.dashboard_model import TelemetryHistory, sparkline_points

_SERIES = (("fps", theme.ELECTRIC_YELLOW), ("p50", theme.CYAN), ("p99", theme.ALERT_RED))


class DashboardPanel(QWidget):
    def __init__(self, publisher, *, interval_ms: int = 200) -> None:
        super().__init__()
        self._pub = publisher
        self.history = TelemetryHistory()
        self._last_seq = None
        self.setMinimumHeight(160)

        layout = QVBoxLayout(self)
        self.status_label = QLabel("AIM --   TRIGGER --")   # live toggle state (Phase 9P)
        self.fps_label = QLabel("FPS --")
        self.lat_label = QLabel("loop p50 -- ms  p99 -- ms")
        self.status_label.setObjectName("mono")
        self.fps_label.setObjectName("mono")     # monospaced telemetry numerals (theme)
        self.lat_label.setObjectName("mono")
        layout.addWidget(self.status_label)
        layout.addWidget(self.fps_label)
        layout.addWidget(self.lat_label)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        snap = self._pub.latest()
        if snap is None or snap.seq == self._last_seq:
            return
        self._last_seq = snap.seq
        self.history.push_snapshot(snap)
        s = self.history.stats()
        self.status_label.setText(self._fmt_status(snap.aim_on, snap.trigger_on))
        self.fps_label.setText(f"FPS {s['fps']:.1f}")
        self.lat_label.setText(f"loop p50 {s['p50']:.1f} ms  p99 {s['p99']:.1f} ms")
        self.update()

    @staticmethod
    def _fmt_status(aim_on, trigger_on) -> str:
        def chip(name: str, on) -> str:
            color = theme.CYAN if on else theme.ALERT_RED
            return f'{name} <span style="color:{color}">{"ON" if on else "OFF"}</span>'
        return chip("AIM", bool(aim_on)) + "&nbsp;&nbsp;&nbsp;" + chip("TRIGGER", bool(trigger_on))

    def paintEvent(self, event) -> None:
        w = float(self.width())
        band = max(1.0, (self.height() - 8) / len(_SERIES))
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            for i, (key, color) in enumerate(_SERIES):
                pts = sparkline_points(self.history.series(key),
                                       x0=4.0, y0=4.0 + i * band, w=w - 8.0,
                                       h=band - 4.0)
                if len(pts) < 2:
                    continue
                p.setPen(QPen(QColor(color), 1))
                p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
        finally:
            p.end()
