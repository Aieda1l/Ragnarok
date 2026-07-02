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
