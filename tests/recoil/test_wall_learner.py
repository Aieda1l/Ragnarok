import numpy as np
import pytest

from ragnarok.recoil.wall_learner import measure_shift, accumulate_drift, resample_at_shots


def test_measure_shift_recovers_synthetic_translation():
    rng = np.random.default_rng(0)
    img = rng.random((96, 96)).astype(np.float32)
    shifted = np.roll(img, shift=(3, 5), axis=(0, 1))     # dy=3 rows, dx=5 cols
    dx, dy = measure_shift(img, shifted)
    assert abs(abs(dx) - 5.0) < 1.0                        # recovers |dx|≈5
    assert abs(abs(dy) - 3.0) < 1.0                        # recovers |dy|≈3


def test_accumulate_drift_negates_and_cumsums():
    # scene shifts down/right -> view kicked up/left -> drift is negated cumsum
    assert accumulate_drift([(1.0, 2.0), (1.0, 2.0), (0.0, 1.0)]) == (
        (-1.0, -2.0), (-2.0, -4.0), (-2.0, -5.0))
    assert accumulate_drift([]) == ()


def test_resample_at_shots_linear_interp():
    # cumulative drift rises 0->10 px over 1.0s of frames (11 frames @ 0.1s)
    cumulative = tuple((i, 0.0) for i in range(11))        # (0,0)..(10,0), dt=0.1s
    # 10 rps -> shots at 0.0,0.1,...  -> drift 0,1,2,... px
    pat = resample_at_shots(cumulative, dt_frame_s=0.1, rps=10.0, num_shots=5)
    assert [x for x, _ in pat] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert all(y == 0.0 for _, y in pat)


def test_resample_guards_bad_input():
    assert resample_at_shots((), 0.1, 10.0, 5) == ()
    assert resample_at_shots(((0.0, 0.0),), 0.1, 0.0, 5) == ()   # rps=0
    assert resample_at_shots(((0.0, 0.0),), 0.1, 10.0, 0) == ()  # num_shots=0
