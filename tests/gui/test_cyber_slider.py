from PySide6.QtGui import QImage
from ragnarok.gui.cyber_slider import CyberSlider


def test_spinbox_compatible_surface(qtbot):
    s = CyberSlider()
    qtbot.addWidget(s)
    s.setMinimum(0.0)
    s.setMaximum(2.0)
    s.setSingleStep(0.01)
    s.setDecimals(3)
    s.setValue(0.35)
    assert s.value() == 0.35
    assert s.minimum() == 0.0 and s.maximum() == 2.0


def test_setvalue_clamps_to_range(qtbot):
    s = CyberSlider()
    qtbot.addWidget(s)
    s.setMinimum(0.0)
    s.setMaximum(1.0)
    s.setValue(5.0)
    assert s.value() == 1.0                          # clamped to max


def test_setvalue_does_not_emit_editing_finished(qtbot):
    s = CyberSlider()
    qtbot.addWidget(s)
    s.setMaximum(10.0)
    with qtbot.assertNotEmitted(s.editingFinished):  # refresh path must not re-commit
        s.setValue(4.0)
    assert s.value() == 4.0


def test_paints_without_error(qtbot):
    s = CyberSlider()
    qtbot.addWidget(s)
    s.setMaximum(100.0)
    s.setValue(42.0)
    s.resize(180, 26)
    img = QImage(180, 26, QImage.Format_ARGB32)
    img.fill(0)
    s.render(img)                                    # exercises the fill + value paint
