"""Roboflow dataset client (spec §12 steps 2-3 & 6).

All Roboflow I/O goes through an injected Transport so unit tests use a fake and
never touch the network or the `roboflow` package. The real transport
(RoboflowSdkTransport) lazily wraps the official `roboflow` PyPI package and is a
box-only smoke. The API key comes from RAGNAROK_ROBOFLOW_API_KEY (never config).
"""
from __future__ import annotations

import os
from typing import Protocol


class Transport(Protocol):
    def upload_image(self, image_path: str, *, split: str) -> str: ...
    def download(self, version: int, fmt: str, dest: str) -> str: ...


class RoboflowClient:
    def __init__(self, transport: Transport, *, default_split: str = "train") -> None:
        self._t = transport
        self._split = default_split

    def upload_frames(self, image_paths, *, split: str | None = None) -> list[str]:
        s = split or self._split
        return [self._t.upload_image(p, split=s) for p in image_paths]

    def download_version(self, version: int, dest: str, *, fmt: str = "coco") -> str:
        return self._t.download(version, fmt, dest)
