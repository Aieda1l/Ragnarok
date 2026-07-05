from ragnarok.aim.mouse import pointer_speed_multiplier


def test_neutral_and_measured_scale():
    assert pointer_speed_multiplier(10) == 1.0     # slider 6/11 notch = neutral
    assert pointer_speed_multiplier(6) == 0.5       # matches the live-measured 0.5x
    assert pointer_speed_multiplier(20) == 2.0
    assert pointer_speed_multiplier(1) == 0.1


def test_out_of_range_is_neutral():
    assert pointer_speed_multiplier(0) == 1.0
    assert pointer_speed_multiplier(99) == 1.0
