"""Tests for RoboflowClient against a fake transport (no network/roboflow)."""
from __future__ import annotations
from ragnarok.training.roboflow_client import RoboflowClient


class _FakeTransport:
    def __init__(self):
        self.uploaded: list[tuple[str, str]] = []
        self.downloaded: list[tuple[int, str, str]] = []
    def upload_image(self, image_path, *, split):
        self.uploaded.append((image_path, split))
        return f"id:{image_path}"
    def download(self, version, fmt, dest):
        self.downloaded.append((version, fmt, dest))
        return dest


def test_upload_frames_uploads_each_with_default_split():
    t = _FakeTransport()
    c = RoboflowClient(t)
    refs = c.upload_frames(["a.png", "b.png"])
    assert t.uploaded == [("a.png", "train"), ("b.png", "train")]
    assert refs == ["id:a.png", "id:b.png"]


def test_upload_frames_split_override():
    t = _FakeTransport()
    RoboflowClient(t, default_split="train").upload_frames(["v.png"], split="valid")
    assert t.uploaded == [("v.png", "valid")]


def test_download_version_passes_through():
    t = _FakeTransport()
    out = RoboflowClient(t).download_version(3, "/data/ds", fmt="coco")
    assert t.downloaded == [(3, "coco", "/data/ds")]
    assert out == "/data/ds"
