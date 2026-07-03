"""Named config profiles (per-weapon/per-game presets, spec §13).

Pure file IO over an injected directory, reusing config.store save/load. Profile
names are path-traversal-safe; ``load`` never auto-creates (unlike load_config's
missing-file default), so an unknown profile raises instead of silently seeding
one.
"""
from __future__ import annotations

import re
from pathlib import Path

from ragnarok.config.schema import AppConfig
from ragnarok.config.store import load_config, save_config

_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]+$")


class ProfileStore:
    def __init__(self, directory) -> None:
        self._dir = Path(directory)

    def path_for(self, name: str) -> Path:
        if not name or not _NAME_RE.fullmatch(name):
            raise ValueError(f"invalid profile name {name!r}")
        return self._dir / f"{name}.toml"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def list(self) -> list[str]:
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.toml"))

    def save(self, name: str, cfg: AppConfig) -> Path:
        path = self.path_for(name)
        save_config(cfg, path)                    # creates the dir + writes TOML
        return path

    def load(self, name: str) -> AppConfig:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"no profile named {name!r}")
        return load_config(path)                  # file exists -> just reads it

    def delete(self, name: str) -> None:
        path = self.path_for(name)
        if path.exists():
            path.unlink()
