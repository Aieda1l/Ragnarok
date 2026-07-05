"""Portable config import/export (spec §13).

Read/write an ``AppConfig`` at an arbitrary path — for backing up or sharing a
config outside the managed profiles directory.

Unlike ``store.load_config``, ``import_config`` does NOT auto-create a default on
a missing path: a mistyped import path fails loudly (FileNotFoundError) instead
of silently yielding a blank config. Schema-invalid files raise pydantic's
ValidationError. Export round-trips through the same TOML writer as the live
config, so an exported file re-imports to an equal AppConfig.
"""
from __future__ import annotations

from pathlib import Path

import tomlkit

from ragnarok.config.schema import AppConfig
from ragnarok.config.store import save_config


def import_config(path) -> AppConfig:
    """Load + validate an AppConfig from an existing TOML file (no auto-create)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    data = tomlkit.parse(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(dict(data))


def export_config(cfg: AppConfig, path) -> None:
    """Write ``cfg`` as TOML to ``path`` (creating parent directories)."""
    save_config(cfg, Path(path))
