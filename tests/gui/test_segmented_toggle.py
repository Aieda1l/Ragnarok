from PySide6.QtGui import QImage
from ragnarok.gui.segmented_toggle import SegmentedToggle


def test_setchecked_toggles_and_emits(qtbot):
    t = SegmentedToggle(False)
    qtbot.addWidget(t)
    assert t.isChecked() is False
    with qtbot.waitSignal(t.stateChanged, timeout=1000) as blocker:
        t.setChecked(True)
    assert t.isChecked() is True and blocker.args == [1]


def test_setchecked_same_value_does_not_emit(qtbot):
    t = SegmentedToggle(True)
    qtbot.addWidget(t)
    with qtbot.assertNotEmitted(t.stateChanged):
        t.setChecked(True)                          # no change -> no emit


def test_blocksignals_suppresses_emit(qtbot):
    t = SegmentedToggle(False)
    qtbot.addWidget(t)
    t.blockSignals(True)
    with qtbot.assertNotEmitted(t.stateChanged):
        t.setChecked(True)                          # refresh path: no re-commit
    t.blockSignals(False)
    assert t.isChecked() is True


def test_paints_without_error(qtbot):
    t = SegmentedToggle(True)
    qtbot.addWidget(t)
    t.resize(160, 26)
    img = QImage(160, 26, QImage.Format_ARGB32)
    img.fill(0)
    t.render(img)                                   # exercises paintEvent (both states)
