"""Frameless, click-through, always-on-top smart-weapon FOV overlay (spec §10.2).

Modeled on the Cyberpunk 2077 smart-link reticle: a square FOV framed by two
brackets (thin vertical line + bold ~45° arms) and red diamonds on detected aim
points. All geometry lives in ``overlay_model.build_scene`` (pure, tested); this
widget is the thin QPainter renderer + Qt window plumbing. Painting *correctness*
(pixels) is box-only; construction, window flags, and a render smoke are
offscreen-testable via qtbot.

BOX-ONLY DEFERRALS (perf / OS integration, not implemented here):
  * swap ``QWidget`` -> ``QOpenGLWidget`` for the GL-backed path (avoids slow GDI).
  * apply Win32 ``WS_EX_TRANSPARENT | WS_EX_LAYERED`` via ctypes on ``show()``
    for OS-level click-through (``WA_TransparentForMouseEvents`` handles the
    Qt-level pass-through cross-platform meanwhile).
  * position/size the window to the captured region and translate the painter
    for multi-monitor origins.
  * glitch/chromatic-aberration and scanline effects.
  * off-screen direction hints stay latent until an off-screen target source
    exists (full-frame detection / wider capture / coasted tracks leaving the
    ROI); the single centered-ROI path keeps every target inside the viewport.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ragnarok.core.types import Team
from ragnarok.gui import theme
from ragnarok.gui.overlay_model import build_scene


class FovOverlay(QWidget):
    def __init__(self, publisher, config_provider, *, interval_ms: int = 16) -> None:
        super().__init__()
        self._pub = publisher
        self._cfg = config_provider
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
        vp = (0.0, 0.0, float(self.width()), float(self.height()))
        scene = build_scene(snapshot=snap, cfg=self._cfg(), viewport=vp)
        if not scene.has_signal:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._draw_scene(p, scene)
        finally:
            p.end()

    def _draw_scene(self, p: QPainter, scene) -> None:
        cyan = QColor(theme.CYAN)
        red = QColor(theme.ALERT_RED)

        # FOV brackets: thin verticals + bold 45° corner arms (smart-weapon frame,
        # cyan/teal like the CP2077 smart-link reticle)
        p.setPen(QPen(cyan, 1))
        for (a, b) in scene.fov_thin:
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
        p.setPen(QPen(cyan, 3))                        # arms are bolder than the verticals
        for (a, b) in scene.fov_thick:
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # crosshair tick
        p.setPen(QPen(cyan, 1))
        cx, cy = scene.crosshair
        p.drawLine(QPointF(cx - 6, cy), QPointF(cx + 6, cy))
        p.drawLine(QPointF(cx, cy - 6), QPointF(cx, cy + 6))

        # markers: red diamonds on ENEMY aim points (locked = filled + larger);
        # teammates/unknown get a small do-not-shoot box in their team colour.
        for m in scene.markers:
            if m.team is Team.ENEMY:
                col = QColor(red)
                if not m.in_fov:
                    col.setAlpha(120)                  # dim enemies outside the cone
                self._diamond(p, m.diamond[0], m.diamond[1],
                              8 if m.locked else 6, col, filled=m.locked)
            else:
                col = QColor(theme.team_color(m.team))
                p.setPen(QPen(col, 1))
                p.setBrush(Qt.NoBrush)
                x1, y1, x2, y2 = m.box
                p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

        # thin dashed tracking line: crosshair -> locked diamond
        if scene.locked_line is not None:
            a, b = scene.locked_line
            pen = QPen(cyan, 1)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # off-screen direction hints: red diamonds at the viewport edge
        for h in scene.offscreen:
            self._diamond(p, h.edge_point[0], h.edge_point[1], 5, red)

    @staticmethod
    def _diamond(p: QPainter, cx: float, cy: float, r: float, col: QColor,
                 *, filled: bool = False) -> None:
        poly = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                          QPointF(cx, cy + r), QPointF(cx - r, cy)])
        p.setPen(QPen(col, 2))
        p.setBrush(QBrush(col) if filled else Qt.NoBrush)
        p.drawPolygon(poly)
        p.setBrush(Qt.NoBrush)
