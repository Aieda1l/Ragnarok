import re
from ragnarok.core.types import Team
from ragnarok.gui import theme


def test_palette_tokens_are_valid_distinct_hex():
    tokens = [theme.ELECTRIC_YELLOW, theme.CYAN, theme.ALERT_RED, theme.NEAR_BLACK]
    for t in tokens:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", t)
    assert theme.ELECTRIC_YELLOW.upper() == "#FCEE0A"     # spec §10.1 primary accent
    assert len(set(t.upper() for t in tokens)) == 4


def test_team_color_maps_all_teams_distinctly():
    cols = {theme.team_color(t) for t in Team}
    assert len(cols) == 3
    # enemy = warm/orange, teammate = blue (mirrors gui/overlay.TEAM_BGR)
    assert theme.team_color(Team.ENEMY) != theme.team_color(Team.TEAMMATE)


def test_stylesheet_is_wellformed_and_uses_red_cyan():
    qss = theme.build_stylesheet()
    assert isinstance(qss, str) and len(qss) > 200
    assert theme.RED in qss and theme.CYAN in qss and theme.BG in qss
    assert qss.count("{") == qss.count("}")               # balanced braces
    assert "QTabBar::tab" in qss and "QPushButton" in qss and "QLabel#mono" in qss


def test_load_fonts_absent_dir_is_empty_and_qt_free(tmp_path):
    assert theme.load_fonts(tmp_path / "does_not_exist") == []
    assert theme.load_fonts(tmp_path) == []               # empty dir -> nothing


def test_apply_theme_sets_stylesheet_without_a_real_qapp():
    class _App:
        def __init__(self): self.qss = None
        def setStyleSheet(self, s): self.qss = s
    app = _App()
    theme.apply_theme(app, font_dir="/nonexistent")       # absent dir -> no Qt font import
    assert app.qss and theme.RED in app.qss
