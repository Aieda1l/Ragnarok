"""Cyberpunk 2077 theme: palette tokens, bundled-font loading, and the app-wide
QSS stylesheet (spec §10.1).

The palette tokens and ``build_stylesheet`` are pure (no Qt import) so they stay
unit-testable. ``load_fonts``/``apply_theme`` touch Qt and import it lazily.

Aesthetic: near-black backgrounds, signature electric-yellow (#FCEE0A) primary
accent, cyan/teal + alert-red secondaries, angular (zero-radius) panels, thin
borders, condensed techy type (Rajdhani/Saira/Chakra Petch if bundled), and
monospaced numerals for telemetry. Corner-bracket chrome, scanlines and glitch
transitions are a later custom-paint pass; this establishes the base skin.
"""
from __future__ import annotations

from ragnarok.core.types import Team

# --- signature accents (spec §10.1) ---------------------------------------
ELECTRIC_YELLOW = "#FCEE0A"   # primary accent (FOV ring, brackets, lock line, focus)
CYAN = "#00F0FF"              # secondary (hover / p50)
ALERT_RED = "#FF3B3B"         # locked-target highlight / alerts / p99
NEAR_BLACK = "#0A0A0C"        # base background

# --- surfaces & text ------------------------------------------------------
BG = NEAR_BLACK
PANEL = "#111218"             # panels / buttons / tabs
PANEL_ALT = "#16171F"         # inputs
BORDER = "#2A2C38"            # thin dividers / idle input borders
TEXT = "#D8DAE0"
TEXT_DIM = "#7A7C88"

# --- typography -----------------------------------------------------------
# Condensed techy stack with graceful fallback when the faces aren't bundled.
FONT_STACK = '"Rajdhani","Saira Condensed","Chakra Petch","Segoe UI",sans-serif'
MONO_STACK = '"Chakra Petch","JetBrains Mono","Consolas",monospace'

# Team colors in RGB hex, mirroring gui/overlay.TEAM_BGR (orange / blue / gray).
TEAM_RGB = {
    Team.ENEMY.value: "#FF8C00",
    Team.TEAMMATE.value: "#0080FF",
    Team.UNKNOWN.value: "#A0A0A0",
}


def team_color(team: Team) -> str:
    return TEAM_RGB.get(team.value, TEAM_RGB[Team.UNKNOWN.value])


def build_stylesheet() -> str:
    """The app-wide Cyberpunk QSS. Pure string builder (no Qt) so it is testable.

    Labels/telemetry with ``objectName == "mono"`` render in the monospaced
    numeral stack (spec §10.1)."""
    return f"""
    QWidget {{
        background: {BG};
        color: {TEXT};
        font-family: {FONT_STACK};
        font-size: 13px;
    }}
    QMainWindow, QDialog {{ background: {BG}; }}
    QLabel {{ background: transparent; }}
    QLabel#mono {{ font-family: {MONO_STACK}; color: {ELECTRIC_YELLOW}; }}

    /* Tabs — angular, yellow-underlined selection */
    QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; }}
    QTabBar::tab {{
        background: {PANEL};
        color: {TEXT_DIM};
        padding: 6px 14px;
        border: 1px solid {BORDER};
        border-bottom: none;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {BG};
        color: {ELECTRIC_YELLOW};
        border-top: 2px solid {ELECTRIC_YELLOW};
    }}
    QTabBar::tab:hover {{ color: {CYAN}; }}

    /* Buttons — CP2077 fill-on-hover */
    QPushButton {{
        background: {PANEL};
        color: {ELECTRIC_YELLOW};
        border: 1px solid {ELECTRIC_YELLOW};
        border-radius: 0px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ background: {ELECTRIC_YELLOW}; color: {BG}; }}
    QPushButton:pressed {{ background: {CYAN}; color: {BG}; border-color: {CYAN}; }}
    QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}

    /* Inputs — dark, yellow focus */
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        background: {PANEL_ALT};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 0px;
        padding: 3px 6px;
        selection-background-color: {ELECTRIC_YELLOW};
        selection-color: {BG};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ELECTRIC_YELLOW};
    }}
    QComboBox::drop-down {{ border: none; width: 16px; }}
    QComboBox QAbstractItemView {{
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {ELECTRIC_YELLOW};
        selection-background-color: {ELECTRIC_YELLOW};
        selection-color: {BG};
    }}

    /* Check boxes */
    QCheckBox {{ spacing: 6px; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {BORDER}; background: {PANEL_ALT};
    }}
    QCheckBox::indicator:checked {{ background: {ELECTRIC_YELLOW}; border: 1px solid {ELECTRIC_YELLOW}; }}
    QCheckBox::indicator:hover {{ border: 1px solid {CYAN}; }}

    /* Scroll bars — thin, yellow on hover */
    QScrollBar:vertical {{ background: {BG}; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {BORDER}; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {ELECTRIC_YELLOW}; }}
    QScrollBar:horizontal {{ background: {BG}; height: 10px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {BORDER}; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ background: {ELECTRIC_YELLOW}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    """


def load_fonts(font_dir=None) -> list[str]:
    """Register bundled .ttf/.otf faces from ``font_dir`` (default gui/fonts/).

    Returns the loaded family names. Safe when the directory is absent (returns
    ``[]`` without importing Qt) — the QSS falls back to a system condensed face.
    Drop Rajdhani/Saira/Chakra-Petch .ttf files into gui/fonts/ for the exact look.
    """
    from pathlib import Path
    font_dir = Path(font_dir) if font_dir is not None else Path(__file__).parent / "fonts"
    if not font_dir.exists():
        return []
    from PySide6.QtGui import QFontDatabase
    families: list[str] = []
    for f in sorted(font_dir.glob("*.ttf")) + sorted(font_dir.glob("*.otf")):
        fid = QFontDatabase.addApplicationFont(str(f))
        if fid != -1:
            families.extend(QFontDatabase.applicationFontFamilies(fid))
    return families


def apply_theme(app, *, font_dir=None) -> None:
    """Load bundled fonts and apply the Cyberpunk stylesheet to the QApplication."""
    load_fonts(font_dir)
    app.setStyleSheet(build_stylesheet())
