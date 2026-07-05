from ragnarok.gui.title_bar import TitleBar


class _FakeWindow:
    def __init__(self):
        self.calls = []
        self._maxed = False
    def showMinimized(self): self.calls.append("min")
    def showMaximized(self): self._maxed = True; self.calls.append("max")
    def showNormal(self): self._maxed = False; self.calls.append("normal")
    def close(self): self.calls.append("close")
    def isMaximized(self): return self._maxed


def test_titlebar_buttons_drive_the_window(qtbot):
    win = _FakeWindow()
    bar = TitleBar(win, "RAGNAROK")
    qtbot.addWidget(bar)
    assert bar.title_label.text() == "RAGNAROK"
    bar.btn_min.click()
    assert win.calls == ["min"]
    bar.btn_close.click()
    assert "close" in win.calls


def test_titlebar_maximize_toggles(qtbot):
    win = _FakeWindow()
    bar = TitleBar(win)
    qtbot.addWidget(bar)
    bar.toggle_maximize()
    assert win.isMaximized() is True                 # normal -> maximized
    bar.toggle_maximize()
    assert win.isMaximized() is False                # maximized -> normal
