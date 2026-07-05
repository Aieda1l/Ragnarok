"""Cyberpunk 2077 theme: palette tokens, bundled-font loading, and the app-wide
QSS stylesheet (spec §10.1).

Palette follows the CP2077 *settings menu*: a dark red-tinted background with
salmon-red labels/borders and cyan values/active states (the electric yellow is
kept only for the HUD/overlay accents). The palette tokens and
``build_stylesheet`` are pure (no Qt import) so they stay unit-testable;
``load_fonts``/``apply_theme`` touch Qt and import it lazily.
"""
from __future__ import annotations

from ragnarok.core.types import Team

# --- signature accents ----------------------------------------------------
RED = "#F0413C"               # primary GUI accent (labels, borders, OFF toggle)
CYAN = "#38E0F0"              # secondary (values, active/ON, hover, selected tab)
ELECTRIC_YELLOW = "#FCEE0A"   # HUD/overlay accent (dashboard FPS trace)
ALERT_RED = "#FF3B3B"         # overlay target diamonds / p99 trace
NEAR_BLACK = "#0A0A0C"

# --- surfaces & text (dark, subtly red-tinted like the CP2077 menus) ------
BG = "#100608"
PANEL = "#1C0C0E"             # panels / buttons / tabs
PANEL_ALT = "#281114"        # inputs
BORDER = "#6A2E30"           # dark-red dividers / idle input borders
TEXT = "#E6D2D2"             # warm light body text
TEXT_DIM = "#8A6668"
LABEL = "#E8908C"            # salmon-red field labels

# --- typography -----------------------------------------------------------
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
    """The app-wide CP2077 QSS (red labels/borders, cyan values/active). Pure
    string builder (no Qt) so it is testable. Labels/telemetry with
    ``objectName == "mono"`` render in the cyan monospaced numeral stack."""
    return f"""
    QWidget {{
        background: {BG};
        color: {TEXT};
        font-family: {FONT_STACK};
        font-size: 13px;
    }}
    QMainWindow, QDialog {{ background: {BG}; }}
    /* frameless window border + custom title bar */
    QWidget#root {{ background: {BG}; border: 1px solid {RED}; }}
    QLabel#titlebar {{ color: {CYAN}; font-weight: bold; font-size: 14px;
                       letter-spacing: 3px; }}
    QPushButton#titleclose {{ color: {RED}; border: none; background: transparent;
                              font-size: 14px; }}
    QPushButton#titleclose:hover {{ background: {RED}; color: {BG}; }}
    QLabel {{ background: transparent; color: {LABEL}; }}
    QLabel#mono {{ font-family: {MONO_STACK}; color: {CYAN}; }}
    /* section header — light, bold, with a cyan underline divider */
    QLabel#header {{
        color: {TEXT}; font-size: 15px; font-weight: bold;
        padding: 4px 0px; border-bottom: 1px solid {CYAN}; margin-bottom: 4px;
    }}
    /* flat ◁ ▷ selector arrows (ArrowSelector) */
    QPushButton:flat {{ color: {RED}; border: none; background: transparent;
                        font-size: 15px; padding: 0px; }}
    QPushButton:flat:hover {{ color: {CYAN}; }}
    QPushButton:flat:disabled {{ color: {TEXT_DIM}; }}

    /* Tabs — inactive red, active cyan with a cyan top edge */
    QTabWidget::pane {{ border: 1px solid {BORDER}; top: -1px; }}
    QTabBar::tab {{
        background: {PANEL};
        color: {RED};
        padding: 6px 14px;
        border: 1px solid {BORDER};
        border-bottom: none;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {BG};
        color: {CYAN};
        border-top: 2px solid {CYAN};
    }}
    QTabBar::tab:hover {{ color: {CYAN}; }}

    /* Buttons — red border, fill-on-hover */
    QPushButton {{
        background: {PANEL};
        color: {RED};
        border: 1px solid {RED};
        border-radius: 0px;
        padding: 6px 14px;
    }}
    QPushButton:hover {{ background: {RED}; color: {BG}; }}
    QPushButton:pressed {{ background: {CYAN}; color: {BG}; border-color: {CYAN}; }}
    QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}

    /* Inputs — dark, cyan value text, red border -> cyan on focus */
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        background: {PANEL_ALT};
        color: {CYAN};
        border: 1px solid {BORDER};
        border-radius: 0px;
        padding: 3px 6px;
        selection-background-color: {CYAN};
        selection-color: {BG};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
        border: 1px solid {CYAN};
    }}
    QComboBox::drop-down {{ border: none; width: 16px; }}
    QComboBox QAbstractItemView {{
        background: {PANEL};
        color: {TEXT};
        border: 1px solid {RED};
        selection-background-color: {RED};
        selection-color: {BG};
    }}

    /* Check boxes (fallback; bool fields use the segmented toggle) */
    QCheckBox {{ spacing: 6px; color: {LABEL}; }}
    QCheckBox::indicator {{
        width: 14px; height: 14px;
        border: 1px solid {BORDER}; background: {PANEL_ALT};
    }}
    QCheckBox::indicator:checked {{ background: {CYAN}; border: 1px solid {CYAN}; }}
    QCheckBox::indicator:hover {{ border: 1px solid {RED}; }}

    /* Scroll bars — thin, red handle (CP2077 accent) brightening on hover */
    QScrollBar:vertical {{ background: {BG}; width: 8px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {RED}; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {CYAN}; }}
    QScrollBar:horizontal {{ background: {BG}; height: 8px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {RED}; min-width: 30px; }}
    QScrollBar::handle:horizontal:hover {{ background: {CYAN}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollArea {{ border: none; }}
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
