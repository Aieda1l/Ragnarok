from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel
from ragnarok.gui.chrome_frame import ChromeFrame


def test_chrome_frame_embeds_content(qtbot):
    child = QLabel("content")
    frame = ChromeFrame(child)
    qtbot.addWidget(frame)
    assert child.parent() is not None                       # reparented into the frame


def test_chrome_frame_paints_without_error(qtbot):
    frame = ChromeFrame(QLabel("x"))
    qtbot.addWidget(frame)
    frame.resize(300, 200)
    img = QImage(300, 200, QImage.Format_ARGB32)
    img.fill(0)
    frame.render(img)                                       # exercises paintEvent


def test_chrome_frame_degenerate_size_is_safe(qtbot):
    frame = ChromeFrame(QLabel("x"))
    qtbot.addWidget(frame)
    frame.resize(4, 4)                                      # smaller than margins -> no-op paint
    img = QImage(4, 4, QImage.Format_ARGB32)
    img.fill(0)
    frame.render(img)
