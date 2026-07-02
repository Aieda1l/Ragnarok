"""Cyberpunk 2077 palette tokens (spec §10.1).

Function-first slice: just the color tokens the smart-lock overlay needs. The
full QSS stylesheet, condensed fonts, scanlines, and glitch effects are a later
(box-only) aesthetic pass.
"""
from __future__ import annotations

from ragnarok.core.types import Team

ELECTRIC_YELLOW = "#FCEE0A"   # primary accent (FOV ring, brackets, lock line)
CYAN = "#00F0FF"              # secondary
ALERT_RED = "#FF3B3B"         # locked-target highlight / alerts
NEAR_BLACK = "#0A0A0C"        # backgrounds

# Team colors in RGB hex, mirroring gui/overlay.TEAM_BGR (orange / blue / gray).
TEAM_RGB = {
    Team.ENEMY.value: "#FF8C00",
    Team.TEAMMATE.value: "#0080FF",
    Team.UNKNOWN.value: "#A0A0A0",
}


def team_color(team: Team) -> str:
    return TEAM_RGB.get(team.value, TEAM_RGB[Team.UNKNOWN.value])
