"""Frameless, click-through, always-on-top smart-lock FOV overlay (spec §10.2).

Function-first: all geometry lives in ``overlay_model.build_scene`` (pure,
tested). This widget is the thin QPainter renderer + Qt window plumbing.
Painting *correctness* (pixels) is box-only; construction, window flags, and a
render smoke are offscreen-testable via qtbot.

BOX-ONLY DEFERRALS (perf / OS integration, not implemented here):
  * swap ``QWidget`` -> ``QOpenGLWidget`` for the GL-backed path (avoids slow GDI).
  * apply Win32 ``WS_EX_TRANSPARENT | WS_EX_LAYERED`` via ctypes on ``show()``
    for OS-level click-through (``WA_TransparentForMouseEvents`` handles the
    Qt-level pass-through cross-platform meanwhile).
  * position/size the window to the captured region and translate the painter
    for multi-monitor origins.
  * full Cyberpunk restyle: glitch/chromatic-aberration, scanlines, gauges.
  * off-screen direction hints stay latent until an off-screen target source
    exists (full-frame detection / wider capture / coasted tracks leaving the
    ROI); the single centered-ROI path keeps every target inside the viewport.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Team
from ragnarok.gui import theme
from ragnarok.gui.overlay_model import LockAgeTracker, build_scene


class FovOverlay(QWidget):
    def __init__(self, publisher, config_provider, *, interval_ms: int = 16,
                 clock=now_ns) -> None:
        super().__init__()
        self._pub = publisher
        self._cfg = config_provider
        self._clock = clock
        self._lock_age = LockAgeTracker()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)          # own timer, decoupled from hot loop
        self._timer.timeout.connect(self.update)      # schedule a repaint
        self._timer.start()

    # -- rendering -------------------------------------------------------
    def paintEvent(self, event) -> None:
        snap = self._pub.latest()
        if snap is None:
            return
        now = self._clock()
        lock_age = self._lock_age.update(snap.locked_target_id, now)
        vp = (0.0, 0.0, float(self.width()), float(self.height()))
        scene = build_scene(snapshot=snap, cfg=self._cfg(), viewport=vp,
                            lock_age_s=lock_age)
        if not scene.has_signal:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._draw_scene(p, scene)
        finally:
            p.end()

    def _draw_scene(self, p: QPainter, scene) -> None:
        yellow = QColor(theme.ELECTRIC_YELLOW)
        red = QColor(theme.ALERT_RED)

        # FOV ring: acquire (solid) + retain (dim)
        if scene.fov is not None:
            cx, cy = scene.fov.center
            self._ring(p, cx, cy, scene.fov.acquire_radius, yellow, 2)
            dim = QColor(yellow)
            dim.setAlpha(90)
            self._ring(p, cx, cy, scene.fov.retain_radius, dim, 1)

        # crosshair tick
        p.setPen(QPen(yellow, 1))
        cx, cy = scene.crosshair
        p.drawLine(QPointF(cx - 6, cy), QPointF(cx + 6, cy))
        p.drawLine(QPointF(cx, cy - 6), QPointF(cx, cy + 6))

        # markers: team-colored box + diamond for enemies; lock = red.
        # Targets outside the acquisition cone are dimmed (in_fov -> full accent).
        for m in scene.markers:
            if m.locked:
                col = QColor(red)
            else:
                col = QColor(theme.team_color(m.team))
                if not m.in_fov:
                    col.setAlpha(110)
            p.setPen(QPen(col, 2 if m.locked else 1))
            x1, y1, x2, y2 = m.box
            p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if m.team is Team.ENEMY:
                self._diamond(p, m.diamond[0], m.diamond[1], 6, col)

        # lock-on convergence brackets (locked target)
        p.setPen(QPen(yellow, 2))
        for (a, b) in scene.bracket_segments:
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # thin dashed tracking line: crosshair -> locked diamond
        if scene.locked_line is not None:
            a, b = scene.locked_line
            pen = QPen(yellow, 1)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # off-screen direction hints: small markers at the viewport edge
        for h in scene.offscreen:
            col = QColor(theme.team_color(h.team))
            self._diamond(p, h.edge_point[0], h.edge_point[1], 5, col)

    @staticmethod
    def _ring(p: QPainter, cx: float, cy: float, r: float, col: QColor, w: int) -> None:
        p.setPen(QPen(col, w))
        p.drawEllipse(QPointF(cx, cy), r, r)

    @staticmethod
    def _diamond(p: QPainter, cx: float, cy: float, r: float, col: QColor) -> None:
        poly = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                          QPointF(cx, cy + r), QPointF(cx - r, cy)])
        p.setPen(QPen(col, 2))
        p.drawPolygon(poly)
