"""Aim-key provider: ABC, CI-safe fake, and Windows AsyncKeyState implementation.

AsyncKeyStateProvider binds ctypes.WinDLL lazily inside __init__ so that
importing this module on Linux/CI never executes any Windows API calls.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

# ---------------------------------------------------------------------------
# Virtual-key name → code map (winuser.h subset used by aim)
# ---------------------------------------------------------------------------

VK: dict[str, int] = {
    "VK_LBUTTON":  0x01,
    "VK_RBUTTON":  0x02,
    "VK_MBUTTON":  0x04,
    "VK_XBUTTON1": 0x05,
    "VK_XBUTTON2": 0x06,
    "VK_BACK":     0x08,
    "VK_TAB":      0x09,
    "VK_RETURN":   0x0D,
    "VK_SHIFT":    0x10,
    "VK_CONTROL":  0x11,
    "VK_MENU":     0x12,   # Alt
    "VK_CAPITAL":  0x14,
    "VK_ESCAPE":   0x1B,
    "VK_SPACE":    0x20,
    "VK_PRIOR":    0x21,   # Page Up
    "VK_NEXT":     0x22,   # Page Down
    "VK_END":      0x23,
    "VK_HOME":     0x24,
    "VK_INSERT":   0x2D,
    "VK_DELETE":   0x2E,
    "VK_F1":       0x70,
    "VK_F2":       0x71,
    "VK_F3":       0x72,
    "VK_F4":       0x73,
    "VK_F5":       0x74,
    "VK_F6":       0x75,
    "VK_F7":       0x76,
    "VK_F8":       0x77,
    "VK_F9":       0x78,
    "VK_F10":      0x79,
    "VK_F11":      0x7A,
    "VK_F12":      0x7B,
}


def resolve_vk(key_name: str) -> int:
    """Resolve a configured keybind to a Windows virtual-key code.

    Accepts either a ``VK_*`` name from :data:`VK` or a single character —
    ``A``-``Z`` and ``0``-``9`` are their own virtual-key codes on Windows
    (0x41-0x5A and 0x30-0x39, identical to ASCII). Both forms are
    case-insensitive and tolerate surrounding whitespace, matching the
    "VK_ or char" contract the GUI keybind fields advertise.

    Raises:
        KeyError: if the name is neither a known ``VK_*`` entry nor a single
            alphanumeric character. The message names the offending value.
    """
    name = key_name.strip().upper()
    if name in VK:
        return VK[name]
    if len(name) == 1 and (name.isdigit() or "A" <= name <= "Z"):
        return ord(name)
    raise KeyError(
        f"unknown keybind {key_name!r}: expected a single letter/digit "
        f"(e.g. 'T', '5') or one of {', '.join(sorted(VK))}"
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AimKeyProvider(ABC):
    """Query whether the aim key/button is currently pressed."""

    @abstractmethod
    def is_down(self) -> bool:
        """Return True when the key/button is physically held down."""
        ...


# ---------------------------------------------------------------------------
# CI-safe fake (no OS calls)
# ---------------------------------------------------------------------------

class FakeKeyProvider(AimKeyProvider):
    """Deterministic provider for tests — set `.down` directly."""

    def __init__(self, down: bool = False) -> None:
        self.down: bool = down

    def is_down(self) -> bool:
        return self.down


# ---------------------------------------------------------------------------
# Windows implementation (lazy WinDLL — safe to import anywhere)
# ---------------------------------------------------------------------------

class AsyncKeyStateProvider(AimKeyProvider):
    """Read the physical key state via ``GetAsyncKeyState`` (Windows only).

    The WinDLL binding happens inside ``__init__``, never at module import
    time, so the package can be imported on Linux/CI without errors.
    """

    def __init__(self, key_name: str) -> None:
        import ctypes  # noqa: PLC0415 — intentionally lazy

        self._vk: int = resolve_vk(key_name)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._get_async_key_state = user32.GetAsyncKeyState

    def is_down(self) -> bool:
        # MSB (0x8000) is set when the key is currently physically held down.
        return bool(self._get_async_key_state(self._vk) & 0x8000)


# ---------------------------------------------------------------------------
# Active-aim closure factory
# ---------------------------------------------------------------------------

def make_aim_active(
    provider: AimKeyProvider,
    *,
    toggle: bool,
) -> Callable[[], bool]:
    """Return an ``is_aim_active()`` closure.

    Hold mode (``toggle=False``):
        Returns the raw ``provider.is_down()`` state each call.

    Toggle mode (``toggle=True``):
        Flips an internal flag on each **rising edge** (key transitions from
        up → down). Holding the key does not keep flipping.
    """
    if not toggle:
        # Simple hold: no state needed.
        def _hold() -> bool:
            return provider.is_down()

        return _hold

    # Toggle: track previous physical state and an internal on/off flag.
    _state: dict[str, bool] = {"on": False, "prev": False}

    def _toggle() -> bool:
        down = provider.is_down()
        if down and not _state["prev"]:
            _state["on"] = not _state["on"]
        _state["prev"] = down
        return _state["on"]

    return _toggle
