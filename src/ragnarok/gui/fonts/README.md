# Bundled fonts (optional)

Drop the condensed techy faces the Cyberpunk theme prefers here as `.ttf`/`.otf`
files and they are auto-registered at startup (`theme.load_fonts`). If this
folder is empty, the UI falls back to a system condensed face (Segoe UI on
Windows) — the theme still works, it just isn't the exact Night City type.

Recommended free faces (spec §10.1):

- **Rajdhani** — https://fonts.google.com/specimen/Rajdhani
- **Saira Condensed** — https://fonts.google.com/specimen/Saira+Condensed
- **Chakra Petch** — https://fonts.google.com/specimen/Chakra+Petch (also used for monospaced telemetry numerals)

All three are OFL-licensed. Download the `.ttf` files and place them directly in
this directory (no code change needed). The QSS references them by family name:
`Rajdhani`, `Saira Condensed`, `Chakra Petch`.

Fonts are intentionally NOT committed to the repo (licensing/size); this folder
ships with just this README.
