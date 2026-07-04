from ragnarok.gui.arrow_selector import ArrowSelector


def test_combobox_compatible_surface(qtbot):
    s = ArrowSelector()
    qtbot.addWidget(s)
    s.addItems(["flick", "feedback", "hybrid", "predictive"])
    assert s.currentText() == "flick"                # first item after addItems
    s.setCurrentText("hybrid")
    assert s.currentText() == "hybrid"


def test_arrows_step_and_emit(qtbot):
    s = ArrowSelector()
    qtbot.addWidget(s)
    s.addItems(["a", "b", "c"])
    with qtbot.waitSignal(s.currentIndexChanged, timeout=1000) as blocker:
        s._step(+1)
    assert s.currentText() == "b" and blocker.args == [1]
    s._step(-1)
    assert s.currentText() == "a"


def test_arrows_clamp_at_ends(qtbot):
    s = ArrowSelector()
    qtbot.addWidget(s)
    s.addItems(["a", "b"])
    s._step(-1)                                      # already at first -> no-op
    assert s.currentText() == "a"
    s._step(+1); s._step(+1)                         # clamp at last
    assert s.currentText() == "b"


def test_setcurrenttext_same_value_no_emit(qtbot):
    s = ArrowSelector()
    qtbot.addWidget(s)
    s.addItems(["a", "b"])
    with qtbot.assertNotEmitted(s.currentIndexChanged):
        s.setCurrentText("a")                        # unchanged -> no emit
