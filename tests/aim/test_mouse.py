"""Tests for MouseDriver ABC, NullMouseDriver, and SendInputMouseDriver (Task 2).

All tests inject a fake send callable — no real cursor movement, CI-safe.
"""
from __future__ import annotations

import pytest

from ragnarok.aim.mouse import (
    MOUSEEVENTF_ABSOLUTE,
    MOUSEEVENTF_MOVE,
    MouseButton,
    MouseDriver,
    NullMouseDriver,
    SendInputMouseDriver,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_driver(max_px: int = 32767) -> tuple[SendInputMouseDriver, list[tuple[int, int, int]]]:
    """Return a SendInputMouseDriver with an injected fake send and the call log."""
    calls: list[tuple[int, int, int]] = []

    def fake_send(dx: int, dy: int, flags: int) -> int:
        calls.append((dx, dy, flags))
        return 1

    d = SendInputMouseDriver(send=fake_send, max_px_per_tick=max_px)
    d.connect()
    return d, calls


# ---------------------------------------------------------------------------
# ABC contract: MouseDriver cannot be instantiated directly
# ---------------------------------------------------------------------------

class TestMouseDriverABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            MouseDriver()  # type: ignore[abstract]

    def test_send_input_driver_is_mouse_driver(self):
        d, _ = make_driver()
        assert isinstance(d, MouseDriver)

    def test_null_driver_is_mouse_driver(self):
        assert isinstance(NullMouseDriver(), MouseDriver)


# ---------------------------------------------------------------------------
# SendInputMouseDriver — flags
# ---------------------------------------------------------------------------

class TestSendInputFlags:
    def test_relative_flag_set(self):
        """MOUSEEVENTF_MOVE must be present in every emitted call."""
        d, calls = make_driver()
        d.move_relative(10.0, 0.0)
        assert calls, "expected at least one call"
        for dx, dy, flags in calls:
            assert flags & MOUSEEVENTF_MOVE, "MOUSEEVENTF_MOVE must be set"

    def test_absolute_flag_never_set(self):
        """MOUSEEVENTF_ABSOLUTE must NEVER appear — we want relative moves only."""
        d, calls = make_driver()
        d.move_relative(10.0, -5.0)
        for dx, dy, flags in calls:
            assert (flags & MOUSEEVENTF_ABSOLUTE) == 0, "MOUSEEVENTF_ABSOLUTE must NOT be set"

    def test_only_move_flag_set(self):
        """Exactly MOUSEEVENTF_MOVE and nothing else on a plain relative move."""
        d, calls = make_driver()
        d.move_relative(5.0, 5.0)
        assert calls
        assert calls[0][2] == MOUSEEVENTF_MOVE


# ---------------------------------------------------------------------------
# SendInputMouseDriver — zero-delta guard
# ---------------------------------------------------------------------------

class TestZeroDeltaGuard:
    def test_zero_zero_emits_nothing(self):
        d, calls = make_driver()
        d.move_relative(0.0, 0.0)
        assert calls == []

    def test_sub_pixel_below_threshold_emits_nothing_until_accumulated(self):
        """0.3 px per frame should not emit anything for 2 frames (0.3, 0.6 < 1)."""
        d, calls = make_driver()
        d.move_relative(0.3, 0.0)
        assert calls == [], "0.3 px should not emit yet"
        d.move_relative(0.3, 0.0)
        assert calls == [], "0.6 px cumulative should not emit yet"


# ---------------------------------------------------------------------------
# SendInputMouseDriver — sub-pixel fractional accumulator
# ---------------------------------------------------------------------------

class TestSubPixelAccumulator:
    def test_accumulates_and_emits_on_crossing_whole_pixel(self):
        """0.4 + 0.4 + 0.4 = 1.2 → one emit of (1, 0) on the third frame."""
        d, calls = make_driver()
        d.move_relative(0.4, 0.0)
        assert calls == [], "frame 1: 0.4 cumulative, no emit"
        d.move_relative(0.4, 0.0)
        assert calls == [], "frame 2: 0.8 cumulative, no emit"
        d.move_relative(0.4, 0.0)
        assert len(calls) == 1, "frame 3: 1.2 cumulative, should emit once"
        assert calls[0] == (1, 0, MOUSEEVENTF_MOVE)

    def test_remainder_carries_over_correctly(self):
        """After emitting (1, 0), the 0.2 remainder carries into the next frame.

        Trace: after 3×0.4, remainder=0.2.
        Frame 4: 0.2+0.4=0.6 → no emit, remainder=0.6.
        Frame 5: 0.6+0.4=1.0 → emit (1,0), remainder=0.0.
        """
        d, calls = make_driver()
        for _ in range(3):
            d.move_relative(0.4, 0.0)  # cumulative 0.4→0.8→1.2; emits on 3rd
        assert calls == [(1, 0, MOUSEEVENTF_MOVE)]
        # Remainder is 0.2 after first emit.
        d.move_relative(0.4, 0.0)  # 0.2+0.4=0.6 < 1 → no emit yet
        assert len(calls) == 1, "0.6 cumulative — not yet"
        d.move_relative(0.4, 0.0)  # 0.6+0.4=1.0 → emit
        assert len(calls) == 2, "should emit second time"
        assert calls[1] == (1, 0, MOUSEEVENTF_MOVE)

    def test_total_displacement_close_to_commanded(self):
        """Over N frames of 0.6 px, the sum of emitted ints ≈ commanded total."""
        d, calls = make_driver()
        n = 20
        per_frame = 0.6
        for _ in range(n):
            d.move_relative(per_frame, 0.0)
        total_commanded = n * per_frame
        total_emitted = sum(abs(c[0]) for c in calls)
        # Max error is at most 1 px (fractional remainder at end)
        assert abs(total_emitted - total_commanded) <= 1.0

    def test_accumulator_resets_on_connect(self):
        """connect() must zero the accumulator so leftover state doesn't bleed."""
        calls: list[tuple[int, int, int]] = []

        def fake_send(dx: int, dy: int, flags: int) -> int:
            calls.append((dx, dy, flags))
            return 1

        d = SendInputMouseDriver(send=fake_send)
        d.connect()
        d.move_relative(0.7, 0.0)  # accumulates 0.7, no emit
        assert calls == []
        d.connect()  # second connect should reset accumulator
        d.move_relative(0.7, 0.0)  # fresh start: still 0.7, no emit
        assert calls == []

    def test_negative_accumulator(self):
        """Negative sub-pixel deltas also accumulate correctly."""
        d, calls = make_driver()
        d.move_relative(-0.4, 0.0)
        d.move_relative(-0.4, 0.0)
        assert calls == []
        d.move_relative(-0.4, 0.0)
        assert len(calls) == 1
        assert calls[0] == (-1, 0, MOUSEEVENTF_MOVE)

    def test_both_axes_accumulate_independently(self):
        """x and y accumulators are independent."""
        d, calls = make_driver()
        # x: 0.6 × 2 = 1.2 → emits (1, _, _)
        # y: 0.3 × 2 = 0.6 → not yet
        d.move_relative(0.6, 0.3)
        d.move_relative(0.6, 0.3)
        assert len(calls) == 1
        assert calls[0][0] == 1  # x emitted 1
        assert calls[0][1] == 0  # y not yet (0.6 < 1)


# ---------------------------------------------------------------------------
# SendInputMouseDriver — per-tick clamp
# ---------------------------------------------------------------------------

class TestMaxClamp:
    def test_large_delta_clamped_to_max(self):
        max_px = 100
        d, calls = make_driver(max_px=max_px)
        d.move_relative(500.0, 0.0)
        assert calls
        assert calls[0][0] == max_px

    def test_large_negative_delta_clamped(self):
        max_px = 100
        d, calls = make_driver(max_px=max_px)
        d.move_relative(-500.0, 0.0)
        assert calls
        assert calls[0][0] == -max_px

    def test_within_max_not_clamped(self):
        d, calls = make_driver(max_px=32767)
        d.move_relative(50.0, 30.0)
        assert calls[0][:2] == (50, 30)


# ---------------------------------------------------------------------------
# SendInputMouseDriver — connect / close lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_connect_marks_connected(self):
        d, _ = make_driver()
        assert d._connected is True

    def test_close_marks_disconnected(self):
        d, _ = make_driver()
        d.close()
        assert d._connected is False

    def test_context_manager(self):
        calls: list = []
        d = SendInputMouseDriver(send=lambda dx, dy, f: calls.append((dx, dy, f)) or 1)
        with d:
            assert d._connected is True
            d.move_relative(5.0, 0.0)
        assert d._connected is False
        assert calls == [(5, 0, MOUSEEVENTF_MOVE)]

    def test_set_button_raises_not_implemented(self):
        d, _ = make_driver()
        with pytest.raises(NotImplementedError):
            d.set_button(MouseButton.LEFT, True)


# ---------------------------------------------------------------------------
# NullMouseDriver
# ---------------------------------------------------------------------------

class TestNullMouseDriver:
    def test_records_whole_pixel_moves(self):
        null = NullMouseDriver()
        null.connect()
        null.move_relative(3.0, -2.0)
        assert null.moves == [(3, -2)]

    def test_subpixel_accumulates_in_null_driver(self):
        """NullMouseDriver also uses the accumulator so tests of sub-pixel work."""
        null = NullMouseDriver()
        null.connect()
        null.move_relative(0.4, 0.0)
        null.move_relative(0.4, 0.0)
        assert null.moves == [], "0.8 total not yet whole pixel"
        null.move_relative(0.4, 0.0)
        assert null.moves == [(1, 0)]

    def test_zero_zero_not_recorded(self):
        null = NullMouseDriver()
        null.connect()
        null.move_relative(0.0, 0.0)
        assert null.moves == []

    def test_set_button_records(self):
        null = NullMouseDriver()
        null.connect()
        null.set_button(MouseButton.LEFT, True)
        assert null.buttons == [(MouseButton.LEFT, True)]

    def test_connect_close_tracking(self):
        null = NullMouseDriver()
        assert not null.connected
        null.connect()
        assert null.connected
        null.close()
        assert not null.connected

    def test_multiple_moves_accumulate(self):
        null = NullMouseDriver()
        null.connect()
        null.move_relative(10.0, 5.0)
        null.move_relative(3.0, -1.0)
        assert null.moves == [(10, 5), (3, -1)]
