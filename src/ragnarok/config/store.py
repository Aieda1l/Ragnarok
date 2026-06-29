from __future__ import annotations
from pathlib import Path
import tomlkit
from ragnarok.config.schema import AppConfig

def load_config(path: Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        cfg = AppConfig()
        save_config(cfg, path)
        return cfg
    data = tomlkit.parse(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(dict(data))

def save_config(cfg: AppConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(cfg.model_dump(exclude_none=True)), encoding="utf-8")

class ConfigHandle:
    """Live config snapshot. Single-writer/single-reader: `swap` rebinds one
    attribute (GIL-atomic), readers always see a whole AppConfig, never torn."""
    def __init__(self, initial: AppConfig) -> None:
        self._current = initial

    @property
    def current(self) -> AppConfig:
        return self._current

    def swap(self, cfg: AppConfig) -> None:
        self._current = cfg
