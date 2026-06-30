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


def test_push_hard_examples_uploads_only_hard_frames():
    t = _FakeTransport()
    c = RoboflowClient(t)
    records = [("a", 0.95), ("b", 0.30), ("c", None), ("d", 0.99)]   # b,c are hard
    frames_by_id = {"a": "a.png", "b": "b.png", "c": "c.png", "d": "d.png"}
    refs = c.push_hard_examples(records, frames_by_id, conf_threshold=0.5)
    assert t.uploaded == [("b.png", "train"), ("c.png", "train")]
    assert refs == ["id:b.png", "id:c.png"]


def test_push_hard_examples_skips_ids_without_a_frame_path():
    t = _FakeTransport()
    c = RoboflowClient(t)
    records = [("b", 0.1), ("missing", 0.1)]
    frames_by_id = {"b": "b.png"}                  # 'missing' has no path
    refs = c.push_hard_examples(records, frames_by_id, conf_threshold=0.5)
    assert t.uploaded == [("b.png", "train")]
    assert refs == ["id:b.png"]


def test_push_hard_examples_empty_when_all_confident():
    t = _FakeTransport()
    c = RoboflowClient(t)
    records = [("a", 0.9), ("b", 0.8)]
    assert c.push_hard_examples(records, {"a": "a.png", "b": "b.png"}, conf_threshold=0.5) == []
    assert t.uploaded == []
