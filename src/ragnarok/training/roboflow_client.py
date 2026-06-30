"""Roboflow dataset client (spec §12 steps 2-3 & 6).

All Roboflow I/O goes through an injected Transport so unit tests use a fake and
never touch the network or the `roboflow` package. The real transport
(RoboflowSdkTransport) lazily wraps the official `roboflow` PyPI package and is a
box-only smoke. The API key comes from RAGNAROK_ROBOFLOW_API_KEY (never config).
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol


class Transport(Protocol):
    def upload_image(self, image_path: str, *, split: str) -> str: ...
    def download(self, version: int, fmt: str, dest: str) -> str: ...


class RoboflowClient:
    def __init__(self, transport: Transport, *, default_split: str = "train") -> None:
        self._t = transport
        self._split = default_split

    def upload_frames(self, image_paths: Iterable[str], *, split: str | None = None) -> list[str]:
        s = split if split is not None else self._split
        return [self._t.upload_image(p, split=s) for p in image_paths]

    def download_version(self, version: int, dest: str, *, fmt: str = "coco") -> str:
        return self._t.download(version, fmt, dest)

    def push_hard_examples(self, records: list[tuple[str, float | None]],
                           frames_by_id: dict[str, str], *, conf_threshold: float,
                           split: str | None = None) -> list[str]:
        from ragnarok.training.hard_examples import select_hard_examples
        hard_ids = select_hard_examples(records, conf_threshold=conf_threshold)
        paths = [frames_by_id[i] for i in hard_ids if i in frames_by_id]
        return self.upload_frames(paths, split=split)


class RoboflowSdkTransport:
    """Real transport over the official `roboflow` package. BOX-ONLY (network).

    The `roboflow` import is lazy so this module imports without the package in
    CI; only live use needs `pip install roboflow`.
    """

    def __init__(self, *, api_key: str, workspace: str, project: str) -> None:
        import roboflow  # lazy: optional box-only dependency
        self._project = roboflow.Roboflow(api_key=api_key).workspace(workspace).project(project)

    def upload_image(self, image_path: str, *, split: str) -> str:
        self._project.upload(image_path, split=split)
        return image_path

    def download(self, version: int, fmt: str, dest: str) -> str:
        self._project.version(version).download(fmt, location=dest)
        return dest


def build_roboflow_transport(cfg) -> RoboflowSdkTransport:
    api_key = os.environ.get("RAGNAROK_ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RAGNAROK_ROBOFLOW_API_KEY is not set; required for Roboflow access"
        )
    t = cfg.training
    if not t.roboflow_workspace or not t.roboflow_project:
        raise RuntimeError(
            "training.roboflow_workspace and training.roboflow_project must be configured"
        )
    return RoboflowSdkTransport(
        api_key=api_key, workspace=t.roboflow_workspace, project=t.roboflow_project,
    )
