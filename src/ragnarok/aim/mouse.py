"""Mouse driver: MouseDriver ABC, NullMouseDriver, and SendInputMouseDriver.

SendInputMouseDriver emits relative moves via Win32 SendInput (ctypes).  All
Win32 DLL access is lazily deferred to connect() so the module imports safely
on Linux/macOS CI runners without raising an ImportError.

Sub-pixel fractional accumulator
---------------------------------
Aimers produce float pixel deltas.  A steady 0.4 px/frame command would
truncate to 0 every frame without an accumulator and the crosshair would never
move.  The driver keeps per-axis float remainders and emits a whole-pixel event
only when the cumulative value crosses ±1.
"""
from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Win32 constants (winuser.h)
# ---------------------------------------------------------------------------

INPUT_MOUSE: int = 0

MOUSEEVENTF_MOVE: int = 0x0001
MOUSEEVENTF_ABSOLUTE: int = 0x8000  # NOT used — we want relative deltas

# Deferred (Phase 4 trigger bot):
MOUSEEVENTF_LEFTDOWN: int = 0x0002
MOUSEEVENTF_LEFTUP: int = 0x0004
MOUSEEVENTF_RIGHTDOWN: int = 0x0008
MOUSEEVENTF_RIGHTUP: int = 0x0010
MOUSEEVENTF_MIDDLEDOWN: int = 0x0020
MOUSEEVENTF_MIDDLEUP: int = 0x0040

# Windows applies the pointer-speed slider (Settings > Mouse) as a linear
# multiplier to *relative* SendInput deltas BEFORE they move the cursor, so an
# aimer commanding N counts moves fewer pixels unless we pre-compensate. Slider
# 1..20 -> multiplier (10 = neutral 1.0x); the documented MouseSensitivity curve.
_POINTER_SPEED_MULT: dict[int, float] = {
    1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.4, 6: 0.5, 7: 0.6, 8: 0.7, 9: 0.8,
    10: 1.0, 11: 1.1, 12: 1.2, 13: 1.3, 14: 1.4, 15: 1.5, 16: 1.6, 17: 1.7,
    18: 1.8, 19: 1.9, 20: 2.0,
}


def pointer_speed_multiplier(speed: int) -> float:
    """Windows pointer-speed slider (1..20) -> linear multiplier on relative
    SendInput deltas. Out-of-range / unknown -> 1.0 (assume neutral)."""
    return _POINTER_SPEED_MULT.get(int(speed), 1.0)

# ---------------------------------------------------------------------------
# ctypes structs — exact field order/types matching winuser.h
# ---------------------------------------------------------------------------
# Imported lazily inside _make_real_send() so that on non-Windows platforms
# ctypes.wintypes is still importable (Python ships it on all platforms) but
# ctypes.WinDLL is only called at runtime.

try:
    from ctypes import wintypes

    # wintypes.ULONG_PTR was removed in Python 3.9+; fall back to c_size_t
    # which is always the pointer-sized unsigned integer on the current platform.
    try:
        ULONG_PTR = wintypes.ULONG_PTR  # type: ignore[attr-defined]
    except AttributeError:
        ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),          # signed; relative when ABSOLUTE unset
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),       # 0 → system supplies timestamp
            ("dwExtraInfo", ULONG_PTR),     # MUST be pointer-width (8 bytes on 64-bit)
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUTunion(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", _INPUTunion),
        ]

    _STRUCTS_AVAILABLE = True

except Exception:  # pragma: no cover — non-Windows with no wintypes
    _STRUCTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Real SendInput callable (the only injectable side effect)
# ---------------------------------------------------------------------------

def _make_real_send() -> Callable[[int, int, int], int]:
    """Bind user32.SendInput and return a (dx, dy, flags) -> int callable.

    Called lazily inside connect() so importing the module on CI never
    touches user32.dll.
    """
    if not _STRUCTS_AVAILABLE:
        raise RuntimeError(  # pragma: no cover
            "SendInput structs not available on this platform"
        )
    user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]

    # Pre-compensate Windows pointer ballistics so N commanded counts move N px.
    # SPI_GETMOUSESPEED (0x70) -> 1..20 slider; SPI_GETMOUSE (0x03)[2] -> EPP flag.
    SPI_GETMOUSE, SPI_GETMOUSESPEED = 0x0003, 0x0070
    speed = ctypes.c_int()
    user32.SystemParametersInfoA(SPI_GETMOUSESPEED, 0, ctypes.byref(speed), 0)
    accel = (ctypes.c_int * 3)()
    user32.SystemParametersInfoA(SPI_GETMOUSE, 0, ctypes.byref(accel), 0)
    if accel[2]:  # "Enhance pointer precision" — non-linear, cannot invert cleanly
        import warnings
        warnings.warn(
            "Windows 'Enhance pointer precision' is ON: SendInput moves are "
            "accelerated non-linearly and cannot be fully compensated. Disable it "
            "(Settings > Mouse) or use the Arduino driver for accurate aim.",
            stacklevel=2,
        )
    inv = 1.0 / pointer_speed_multiplier(speed.value)

    fn = user32.SendInput
    fn.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    fn.restype = wintypes.UINT
    cbsize = ctypes.sizeof(INPUT)

    def _send(dx: int, dy: int, flags: int) -> int:
        if flags == MOUSEEVENTF_MOVE:          # compensate motion only, never clicks
            dx = int(round(dx * inv))
            dy = int(round(dy * inv))
        inp = INPUT(
            type=INPUT_MOUSE,
            u=_INPUTunion(mi=MOUSEINPUT(dx, dy, 0, flags, 0, 0)),
        )
        n = fn(1, ctypes.byref(inp), cbsize)
        if n != 1:
            raise OSError(ctypes.get_last_error(), "SendInput injected 0 events")
        return n

    return _send


# ---------------------------------------------------------------------------
# MouseButton enum
# ---------------------------------------------------------------------------

class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


# ---------------------------------------------------------------------------
# MouseDriver ABC
# ---------------------------------------------------------------------------

class MouseDriver(ABC):
    """Abstract interface for mouse output drivers.

    All concrete drivers must implement connect/close for lifecycle management
    and move_relative for sending relative cursor motion.  set_button is
    stubbed in Phase 3 (trigger-bot wiring is Phase 4).
    """

    @abstractmethod
    def connect(self) -> None:
        """Initialise the driver (bind OS handles, zero accumulator, …)."""

    @abstractmethod
    def close(self) -> None:
        """Release the driver."""

    @abstractmethod
    def move_relative(self, dx: float, dy: float) -> None:
        """Send a relative mouse motion of (dx, dy) screen pixels."""

    @abstractmethod
    def set_button(self, button: MouseButton, down: bool) -> None:
        """Press (down=True) or release (down=False) a mouse button."""

    # Context-manager convenience
    def __enter__(self) -> "MouseDriver":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Sub-pixel fractional accumulator
# ---------------------------------------------------------------------------

@dataclass
class _FracAccumulator:
    """Carry fractional pixel remainders so small deltas aren't lost to int()."""

    _rx: float = 0.0
    _ry: float = 0.0

    def step(self, dx: float, dy: float) -> tuple[int, int]:
        sx = self._rx + dx
        sy = self._ry + dy
        ix = int(sx)  # truncates toward zero
        iy = int(sy)
        self._rx = sx - ix  # keep fractional remainder
        self._ry = sy - iy
        return ix, iy

    def reset(self) -> None:
        self._rx = 0.0
        self._ry = 0.0


# ---------------------------------------------------------------------------
# SendInputMouseDriver (real driver, injectable for testing)
# ---------------------------------------------------------------------------

class SendInputMouseDriver(MouseDriver):
    """Windows SendInput-based relative mouse driver.

    Parameters
    ----------
    send:
        Optional callable ``(dx: int, dy: int, flags: int) -> int``.  When
        *None*, connect() resolves the real user32.SendInput lazily.  Inject a
        fake in tests so the real cursor is never touched.
    max_px_per_tick:
        Hard clamp on each integer delta before SendInput to prevent runaway
        filter spikes from snapping the view.
    """

    def __init__(
        self,
        *,
        send: Optional[Callable[[int, int, int], int]] = None,
        max_px_per_tick: int = 32767,
    ) -> None:
        self._send_injectable = send   # may be None (real) or a fake
        self._acc = _FracAccumulator()
        self._max = max_px_per_tick
        self._connected = False
        self._send: Callable[[int, int, int], int]  # set in connect()

    def connect(self) -> None:
        if self._send_injectable is not None:
            self._send = self._send_injectable
        else:
            self._send = _make_real_send()  # lazy — CI never hits this branch
        self._acc.reset()
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def move_relative(self, dx: float, dy: float) -> None:
        ix, iy = self._acc.step(dx, dy)
        if ix == 0 and iy == 0:
            return
        # Per-tick safety clamp
        ix = max(-self._max, min(self._max, ix))
        iy = max(-self._max, min(self._max, iy))
        self._send(ix, iy, MOUSEEVENTF_MOVE)  # RELATIVE: no ABSOLUTE flag

    _BUTTON_FLAGS = {
        MouseButton.LEFT: (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
        MouseButton.RIGHT: (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        MouseButton.MIDDLE: (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
    }

    def set_button(self, button: MouseButton, down: bool) -> None:
        down_flag, up_flag = self._BUTTON_FLAGS[button]
        self._send(0, 0, down_flag if down else up_flag)


# ---------------------------------------------------------------------------
# NullMouseDriver (CI-safe fake: records calls, never touches the cursor)
# ---------------------------------------------------------------------------

@dataclass
class NullMouseDriver(MouseDriver):
    """Drop-in fake driver for tests and non-Windows environments.

    Shares the same _FracAccumulator logic as SendInputMouseDriver so tests of
    sub-pixel accumulation work against NullMouseDriver too.
    """

    moves: list[tuple[int, int]] = field(default_factory=list)
    buttons: list[tuple[MouseButton, bool]] = field(default_factory=list)
    connected: bool = False
    _acc: _FracAccumulator = field(default_factory=_FracAccumulator)

    def connect(self) -> None:
        self._acc.reset()
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def move_relative(self, dx: float, dy: float) -> None:
        ix, iy = self._acc.step(dx, dy)
        if ix or iy:
            self.moves.append((ix, iy))

    def set_button(self, button: MouseButton, down: bool) -> None:
        self.buttons.append((button, down))
