"""Sensitivity-calibration panel: live raw HID mouse-count readout + one-click
"set sensitivity from a 360° turn" (spec §11).

The count accumulation (``_on_counts``/``reset``) and the apply (``_apply_360``,
which reuses ``calibration_model.apply_sensitivity`` with measured_deg=360) are
offscreen-testable. The raw-input capture (native event filter + registration)
is box-only and runs only while this tab is visible.
"""
from __future__ import annotations

from PySide6.QtCore import Signal, QAbstractNativeEventFilter, QTimer
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

from ragnarok.gui.calibration_model import apply_sensitivity
from ragnarok.gui import raw_mouse


class _RawMouseFilter(QAbstractNativeEventFilter):
    def __init__(self, on_counts) -> None:
        super().__init__()
        self._on = on_counts

    def nativeEventFilter(self, eventType, message):  # pragma: no cover — box-only
        if eventType == b"windows_generic_MSG":
            c = raw_mouse.counts_from_native_message(message)
            if c is not None:
                self._on(*c)
        return False, 0


class CountsCalibratePanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, handle, *, reset_provider=None, apply_provider=None) -> None:
        super().__init__()
        self._handle = handle
        self._x = 0
        self._y = 0
        self._filter = _RawMouseFilter(self._on_counts)
        self._installed = False
        # Global hotkeys (polled) so the whole 360° can be done in-game without
        # GUI->game mouse travel skewing the count. Providers injectable for tests.
        cal = handle.current.calibration
        self._reset_prov = reset_provider
        self._apply_prov = apply_provider
        self._reset_key = cal.reset_key
        self._apply_key = cal.apply_key
        self._reset_was = False
        self._apply_was = False
        self._key_timer = QTimer(self)
        self._key_timer.setInterval(30)
        self._key_timer.timeout.connect(self._poll_keys)

        root = QVBoxLayout(self)
        header = QLabel("SENSITIVITY CALIBRATION")
        header.setObjectName("header")
        root.addWidget(header)
        root.addWidget(QLabel(
            f"1. Reset ({cal.reset_key}).   2. One full 360° horizontal turn in-game.   "
            f"3. Set sensitivity ({cal.apply_key}).   Use the hotkeys IN-GAME so GUI "
            "mouse travel isn't counted."))
        self.counts = QLabel("X: 0    Y: 0   counts")
        self.counts.setObjectName("mono")
        root.addWidget(self.counts)

        reset = QPushButton("Reset counter")
        reset.clicked.connect(self.reset)
        root.addWidget(reset)
        apply_btn = QPushButton("Set sensitivity from 360° turn")
        apply_btn.clicked.connect(self._apply_360)
        root.addWidget(apply_btn)
        self.result = QLabel("")
        root.addWidget(self.result)

    # ---- testable model surface --------------------------------------------
    def _on_counts(self, dx: int, dy: int) -> None:
        self._x += int(dx)
        self._y += int(dy)
        self.counts.setText(f"X: {self._x:+d}    Y: {self._y:+d}   counts")

    def reset(self) -> None:
        self._x = self._y = 0
        self.counts.setText("X: 0    Y: 0   counts")
        self.result.setText("")

    def _poll_keys(self) -> None:
        """Rising-edge fire of the reset / apply hotkeys (testable; providers injected)."""
        if self._reset_prov is not None:
            down = self._reset_prov.is_down()
            if down and not self._reset_was:
                self.reset()
            self._reset_was = down
        if self._apply_prov is not None:
            down = self._apply_prov.is_down()
            if down and not self._apply_was:
                self._apply_360()
            self._apply_was = down

    def _apply_360(self) -> None:
        try:
            new = apply_sensitivity(self._handle, total_counts=float(self._x),
                                    measured_deg=360.0)
        except Exception as exc:  # zero counts -> sensitivity gt=0 violated, etc.
            self.result.setText(f"⚠ {exc} — do a full 360° turn first")
            return
        self.result.setText(
            f"sensitivity = {new.aim.sensitivity:.5f} deg/count  "
            f"(from {abs(self._x)} counts / 360°)")
        self.configChanged.emit(new)

    # ---- box-only raw-input lifecycle (capture only while tab is shown) -----
    def showEvent(self, event) -> None:  # pragma: no cover — box-only
        super().showEvent(event)
        try:
            QApplication.instance().installNativeEventFilter(self._filter)
            raw_mouse.register_raw_mouse(int(self.window().winId()))
            self._installed = True
        except Exception:
            pass
        if self._reset_prov is None or self._apply_prov is None:
            try:
                from ragnarok.aim.keys import AsyncKeyStateProvider
                if self._reset_prov is None:
                    self._reset_prov = AsyncKeyStateProvider(self._reset_key)
                if self._apply_prov is None:
                    self._apply_prov = AsyncKeyStateProvider(self._apply_key)
            except Exception:
                pass                              # unknown key / non-Windows -> hotkeys off
        self._reset_was = self._apply_was = False
        self._key_timer.start()

    def hideEvent(self, event) -> None:  # pragma: no cover — box-only
        super().hideEvent(event)
        self._key_timer.stop()
        if self._installed:
            try:
                QApplication.instance().removeNativeEventFilter(self._filter)
                raw_mouse.unregister_raw_mouse()
            except Exception:
                pass
            self._installed = False
