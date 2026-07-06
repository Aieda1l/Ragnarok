from ragnarok.gui.raw_mouse import RAWINPUT, RIM_TYPEMOUSE, extract_mouse_counts


def test_extract_mouse_counts_reads_relative_deltas():
    ri = RAWINPUT()
    ri.header.dwType = RIM_TYPEMOUSE
    ri.data.mouse.lLastX = 50
    ri.data.mouse.lLastY = -30
    assert extract_mouse_counts(ri) == (50, -30)


def test_extract_non_mouse_returns_none():
    ri = RAWINPUT()
    ri.header.dwType = 1                      # RIM_TYPEKEYBOARD
    assert extract_mouse_counts(ri) is None
