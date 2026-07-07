from PySide6.QtWidgets import QLabel, QScrollArea

from ragnarok.gui.tab_groups import grouped_tab


def test_grouped_tab_stacks_all_widgets(qtbot):
    a, b = QLabel("A"), QLabel("B")
    tab = grouped_tab([a, b])
    qtbot.addWidget(tab)
    assert isinstance(tab, QScrollArea)
    kids = tab.widget().findChildren(QLabel)
    assert a in kids and b in kids
