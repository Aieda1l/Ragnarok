from PySide6.QtCore import Qt

from ragnarok.gui.overlay_window import FovOverlay


class _Pub:
    def latest(self):
        return None


def test_overlay_sets_clickthrough_and_noactivate_attributes(qtbot):
    ov = FovOverlay(_Pub(), lambda: None)
    qtbot.addWidget(ov)
    assert ov.testAttribute(Qt.WA_TransparentForMouseEvents)   # Qt-level click-through
    assert ov.testAttribute(Qt.WA_ShowWithoutActivating)       # never steal game focus
    assert ov.testAttribute(Qt.WA_TranslucentBackground)


def test_win32_helpers_are_safe_to_call(qtbot):
    # Box-only Win32 styling; must be a no-op (never raise) on the CI/offscreen path.
    ov = FovOverlay(_Pub(), lambda: None)
    qtbot.addWidget(ov)
    ov._apply_click_through()
    ov._reassert_topmost()
