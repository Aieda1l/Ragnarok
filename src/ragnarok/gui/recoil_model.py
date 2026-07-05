"""Pure helpers for the recoil pattern editor (spec §10.3 Recoil tab).

ZERO Qt: parse/format the spray pattern as text (one ``dx dy`` per line) and apply
a pattern + scale + enabled into a re-validated AppConfig via ConfigHandle.swap.
"""
from __future__ import annotations


def format_pattern_text(points) -> str:
    """Render pattern points as one ``dx dy`` line each (2-decimal)."""
    return "\n".join(f"{x:.2f} {y:.2f}" for x, y in points)


def parse_pattern_text(text: str) -> tuple[tuple[float, float], ...]:
    """Parse ``dx dy`` lines into pattern points; blank/malformed lines are skipped."""
    pts: list[tuple[float, float]] = []
    for line in text.split("\n"):
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return tuple(pts)


def apply_recoil(handle, points, *, scale: float, enabled: bool):
    """Set recoil.pattern/scale/enabled on a NEW re-validated AppConfig and swap."""
    cfg = handle.current
    new_recoil = cfg.recoil.__class__(**{**cfg.recoil.model_dump(),
                                         "pattern": tuple(points),
                                         "scale": float(scale),
                                         "enabled": bool(enabled)})
    new_cfg = cfg.model_copy(update={"recoil": new_recoil})
    handle.swap(new_cfg)
    return new_cfg
