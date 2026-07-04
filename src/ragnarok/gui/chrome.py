"""Pure geometry for the Cyberpunk panel chrome (spec §10.1).

ZERO Qt: computes the corner-bracket line segments, the side accent-bar rect, and
the notched-corner border polygon for a panel rectangle. ``chrome_frame.py``
paints them with QPainter; the math lives here so it is unit-testable.
"""
from __future__ import annotations

_Seg = tuple[tuple[float, float], tuple[float, float]]


def corner_bracket_segments(x0: float, y0: float, x1: float, y1: float,
                            arm: float) -> tuple[_Seg, ...]:
    """L-shaped brackets at the 4 corners of the rect, arms pointing inward.

    Returns 8 segments (a horizontal + vertical arm per corner)."""
    corners = ((x0, y0, +1, +1), (x1, y0, -1, +1),
               (x0, y1, +1, -1), (x1, y1, -1, -1))
    segs: list[_Seg] = []
    for cx, cy, sx, sy in corners:
        segs.append(((cx, cy), (cx + sx * arm, cy)))   # horizontal arm
        segs.append(((cx, cy), (cx, cy + sy * arm)))    # vertical arm
    return tuple(segs)


def accent_bar_rect(x0: float, y0: float, x1: float, y1: float,
                    width: float) -> tuple[float, float, float, float]:
    """A filled accent bar down the LEFT edge: ``(x, y, w, h)``."""
    return (x0, y0, width, y1 - y0)


def notch_polygon(x0: float, y0: float, x1: float, y1: float,
                  cut: float) -> tuple[tuple[float, float], ...]:
    """Border polygon with the top-right corner clipped at 45° by ``cut`` px
    (the signature CP2077 angular panel edge). Points are clockwise from
    top-left."""
    return (
        (x0, y0),
        (x1 - cut, y0),
        (x1, y0 + cut),
        (x1, y1),
        (x0, y1),
    )
