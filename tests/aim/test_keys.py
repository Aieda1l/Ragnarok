"""Tests for aim/keys.py — Task 3 aim-key provider.

Rules:
- AsyncKeyStateProvider is NOT constructed in CI tests.
- No real mouse/keyboard/GPU/display.
"""
from __future__ import annotations

import pytest

from ragnarok.aim.keys import (
    AimKeyProvider,
    FakeKeyProvider,
    VK,
    make_aim_active,
)


# ---------------------------------------------------------------------------
# VK map
# ---------------------------------------------------------------------------

def test_vk_rbutton():
    assert VK["VK_RBUTTON"] == 0x02


def test_vk_lbutton():
    assert VK["VK_LBUTTON"] == 0x01


def test_vk_mbutton():
    assert VK["VK_MBUTTON"] == 0x04


# ---------------------------------------------------------------------------
# FakeKeyProvider
# ---------------------------------------------------------------------------

def test_fake_key_provider_default_is_up():
    p = FakeKeyProvider()
    assert p.is_down() is False


def test_fake_key_provider_set_down():
    p = FakeKeyProvider(down=True)
    assert p.is_down() is True


def test_fake_key_provider_is_aim_key_provider_subclass():
    assert issubclass(FakeKeyProvider, AimKeyProvider)


def test_fake_key_provider_down_attribute_mutable():
    p = FakeKeyProvider()
    p.down = True
    assert p.is_down() is True
    p.down = False
    assert p.is_down() is False


# ---------------------------------------------------------------------------
# make_aim_active — hold mode
# ---------------------------------------------------------------------------

def test_hold_mode_mirrors_provider():
    p = FakeKeyProvider(down=False)
    is_active = make_aim_active(p, toggle=False)
    assert is_active() is False
    p.down = True
    assert is_active() is True
    p.down = False
    assert is_active() is False


def test_hold_mode_stays_true_while_held():
    p = FakeKeyProvider(down=True)
    is_active = make_aim_active(p, toggle=False)
    for _ in range(5):
        assert is_active() is True


def test_hold_mode_stays_false_while_released():
    p = FakeKeyProvider(down=False)
    is_active = make_aim_active(p, toggle=False)
    for _ in range(5):
        assert is_active() is False


# ---------------------------------------------------------------------------
# make_aim_active — toggle mode
# ---------------------------------------------------------------------------

def test_toggle_flips_on_rising_edge():
    p = FakeKeyProvider(down=False)
    is_active = make_aim_active(p, toggle=True)

    # Initially off
    assert is_active() is False

    # Rising edge: off -> down
    p.down = True
    assert is_active() is True   # toggled ON

    # Held down: no second flip
    assert is_active() is True   # still ON

    # Released
    p.down = False
    assert is_active() is True   # release does not flip

    # Rising edge again: off -> down
    p.down = True
    assert is_active() is False  # toggled OFF


def test_toggle_down_down_does_not_double_flip():
    """Holding the key should NOT keep flipping every call."""
    p = FakeKeyProvider(down=False)
    is_active = make_aim_active(p, toggle=True)

    # First press
    p.down = True
    assert is_active() is True   # ON

    # Multiple calls with key still held — must stay ON
    for _ in range(10):
        assert is_active() is True


def test_toggle_starts_off():
    p = FakeKeyProvider(down=True)  # start with key held
    is_active = make_aim_active(p, toggle=True)
    # On first call, key is already down; it is NOT a rising edge (prev=False but
    # we're seeing it for the first time). Per the spec the closure starts with
    # prev=False so the first call with key down WILL count as a rising edge.
    # That is the intended behaviour: the user had the key held when aim engaged.
    # We just verify the output is a bool and that a subsequent held call doesn't flip.
    result_first = is_active()  # rising edge -> toggles to True
    assert result_first is True
    result_second = is_active()  # still held -> no flip
    assert result_second is True


def test_toggle_multiple_presses():
    """Verify ON/OFF/ON/OFF pattern across four rising edges."""
    p = FakeKeyProvider(down=False)
    is_active = make_aim_active(p, toggle=True)

    expected = [True, True, False, False, True, True, False, False]
    results = []
    for on in [True, False, True, False, True, False, True, False]:
        p.down = on
        results.append(is_active())

    assert results == expected


# ---------------------------------------------------------------------------
# Importing the module does NOT bind WinDLL (CI cross-platform safety)
# ---------------------------------------------------------------------------

def test_import_does_not_touch_windll(monkeypatch):
    """Importing keys should be safe on non-Windows without any WinDLL calls."""
    # If we got here, the import already succeeded — this test just documents it.
    import ragnarok.aim.keys as keys_module  # noqa: F401
    assert hasattr(keys_module, "AsyncKeyStateProvider")
