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
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)   # never steal game focus
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)          # own timer, decoupled from hot loop
        self._timer.timeout.connect(self.update)      # schedule a repaint
        self._timer.start()
        # Re-assert topmost periodically so other apps can't cover the overlay.
        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(1000)
        self._topmost_timer.timeout.connect(self._reassert_topmost)
        self._topmost_timer.start()

    # -- OS-level window integration (Win32; box-only, no-op elsewhere) ---
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._apply_click_through()

    def _apply_click_through(self) -> None:
        """Apply Win32 WS_EX_TRANSPARENT|LAYERED|TOPMOST|NOACTIVATE so clicks pass
        through to the game and the overlay never activates. Qt's
        WA_TransparentForMouseEvents is Qt-level only; games need the native style."""
        try:
            import ctypes
            GWL_EXSTYLE = -20
            EX = 0x20 | 0x80000 | 0x8 | 0x08000000 | 0x80   # TRANSPARENT|LAYERED|TOPMOST|NOACTIVATE|TOOLWINDOW
            u = ctypes.windll.user32
            u.GetWindowLongW.restype = ctypes.c_long
            hwnd = int(self.winId())
            style = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
            u.SetWindowLongW(hwnd, GWL_EXSTYLE, style | EX)
            self._reassert_topmost()
        except Exception:
            pass                                      # non-Windows / no native handle

    def _reassert_topmost(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes
            u = ctypes.windll.user32
            u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
            SWP = 0x0001 | 0x0002 | 0x0010            # NOSIZE|NOMOVE|NOACTIVATE
            u.SetWindowPos(wintypes.HWND(int(self.winId())), wintypes.HWND(-1),
                           0, 0, 0, 0, SWP)           # HWND_TOPMOST = -1
        except Exception:
            pass

    # -- rendering -------------------------------------------------------
    def paintEvent(self, event) -> None:
        snap = self._pub.latest()
        if snap is None:
            return
        cfg = self._cfg()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            self._draw_status(p, snap)                 # AIM/TRIGGER ON-OFF (always shown)
            vp = (0.0, 0.0, float(self.width()), float(self.height()))
            scene = build_scene(snapshot=snap, cfg=cfg, viewport=vp)
            if scene.has_signal:
                self._draw_scene(p, scene, cfg.overlay)
        finally:
            p.end()

    def _draw_status(self, p: QPainter, snap) -> None:
        """Top-left AIM/TRIGGER toggle state so the user always knows what's armed
        (toggles have no held-key feedback). Cyan = ON, red = OFF."""
        aim_on = getattr(snap, "aim_on", None)
        trig_on = getattr(snap, "trigger_on", None)
        p.setPen(QPen(QColor(theme.CYAN if aim_on else theme.ALERT_RED), 1))
        p.drawText(12, 20, f"AIM {'ON' if aim_on else 'OFF'}")
        p.setPen(QPen(QColor(theme.CYAN if trig_on else theme.ALERT_RED), 1))
        p.drawText(96, 20, f"TRIGGER {'ON' if trig_on else 'OFF'}")

    def _draw_scene(self, p: QPainter, scene, ov) -> None:
        cyan = QColor(theme.CYAN)
        red = QColor(theme.ALERT_RED)
        scale = float(ov.diamond_scale)

        # FOV brackets: thin verticals + bold 45° corner arms (smart-weapon frame).
        if ov.show_fov:
            p.setPen(QPen(cyan, 1))
            for (a, b) in scene.fov_thin:
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))
            p.setPen(QPen(cyan, 3))                    # arms bolder than the verticals
            for (a, b) in scene.fov_thick:
                p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # crosshair tick (optional chromatic-aberration ghost)
        cx, cy = scene.crosshair
        if ov.chroma:
            for off, gc in ((-2, QColor(255, 40, 40, 150)), (2, QColor(40, 220, 255, 150))):
                p.setPen(QPen(gc, 1))
                p.drawLine(QPointF(cx - 6 + off, cy), QPointF(cx + 6 + off, cy))
                p.drawLine(QPointF(cx + off, cy - 6), QPointF(cx + off, cy + 6))
        p.setPen(QPen(cyan, 1))
        p.drawLine(QPointF(cx - 6, cy), QPointF(cx + 6, cy))
        p.drawLine(QPointF(cx, cy - 6), QPointF(cx, cy + 6))

        # markers: red diamonds on ENEMY aim points (locked = filled + larger);
        # teammates/unknown get a small do-not-shoot box in their team colour.
        for m in scene.markers:
            if m.team is Team.ENEMY:
                col = QColor(red)
                if not m.in_fov:
                    col.setAlpha(120)                  # dim enemies outside the cone
                r = (8 if m.locked else 6) * scale
                self._diamond(p, m.diamond[0], m.diamond[1], r, col, filled=m.locked)
                if ov.show_confidence:
                    p.setPen(QPen(col, 1))
                    p.drawText(int(m.diamond[0] + r + 2), int(m.diamond[1] + 4),
                               f"{m.confidence:.2f}")
            elif ov.show_boxes:
                col = QColor(theme.team_color(m.team))
                p.setPen(QPen(col, 1))
                p.setBrush(Qt.NoBrush)
                x1, y1, x2, y2 = m.box
                p.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                if ov.show_confidence:
                    p.drawText(int(x1), int(y1) - 2, f"{m.confidence:.2f}")

        # thin dashed tracking line: crosshair -> locked diamond
        if ov.show_tracking_line and scene.locked_line is not None:
            a, b = scene.locked_line
            pen = QPen(cyan, 1)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(a[0], a[1]), QPointF(b[0], b[1]))

        # off-screen direction hints: red diamonds at the viewport edge
        for h in scene.offscreen:
            self._diamond(p, h.edge_point[0], h.edge_point[1], 5 * scale, red)

        self._draw_fx(p, ov)

    def _draw_fx(self, p: QPainter, ov) -> None:
        """Cosmetic CRT scanlines across the whole overlay (spec §10 aesthetic)."""
        if not ov.scanlines:
            return
        p.setPen(QPen(QColor(0, 0, 0, 60), 1))
        w, h = self.width(), self.height()
        y = 0
        while y < h:
            p.drawLine(0, y, w, y)
            y += 3

    @staticmethod
    def _diamond(p: QPainter, cx: float, cy: float, r: float, col: QColor,
                 *, filled: bool = False) -> None:
        poly = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy),
                          QPointF(cx, cy + r), QPointF(cx - r, cy)])
        p.setPen(QPen(col, 2))
        p.setBrush(QBrush(col) if filled else Qt.NoBrush)
        p.drawPolygon(poly)
        p.setBrush(Qt.NoBrush)
