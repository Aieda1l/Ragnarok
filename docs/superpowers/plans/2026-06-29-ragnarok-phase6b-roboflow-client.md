# Ragnarok Phase 6B — Roboflow Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CI-safe Roboflow client (spec §12 steps 2–3 & 6) that uploads collected frames, downloads an exported COCO dataset version, and pushes hard examples back for the next dataset version — all behind an injected transport so unit tests never touch the network.

**Architecture:** A thin `RoboflowClient` orchestrates over a `Transport` Protocol (`upload_image` / `download`). CI injects a fake transport and asserts the calls/payloads; the real transport `RoboflowSdkTransport` is a lazy wrapper over the official `roboflow` PyPI package (constructed only at runtime — never imported in unit tests). The hard-example miner reuses Phase 6A's pure `select_hard_examples` policy to choose which frames to re-upload. The Roboflow API key comes from the `RAGNAROK_ROBOFLOW_API_KEY` environment variable (never config).

**Tech Stack:** Python 3.11+, stdlib `typing.Protocol` + `os`. The official `roboflow` PyPI package is an **optional, box-only** runtime dependency (lazily imported, like `rfdetr`); it is NOT required for the test suite.

## Global Constraints

- **Self-owned offline single-player game** — closed environment; this is the dataset-management tooling for the training loop (spec §12).
- **CI-safe always:** no network / no `roboflow` package in unit tests. All Roboflow I/O goes through the injected `Transport`; CI uses a fake. The module must import without `roboflow` installed (the real SDK import is lazy, inside `RoboflowSdkTransport`).
- **No secrets in config or logs:** the API key is read from `RAGNAROK_ROBOFLOW_API_KEY` (spec §13); it is never a config field, never logged, never persisted. `TrainingConfig` (Phase 6A) already holds only `roboflow_workspace`/`roboflow_project`/`roboflow_version`.
- **Reuse, don't reinvent:** the hard-example *selection* policy already exists (`ragnarok.training.hard_examples.select_hard_examples`, Phase 6A) — the miner consumes it.
- **YAGNI:** support exactly upload + download(version, format) + push-hard-examples. Do NOT add programmatic dataset-version *generation*, annotation automation, or Label-Assist wiring (out of scope; done in the Roboflow UI).
- **TDD, frequent commits, exact file paths.** Match the codebase idiom (`from __future__ import annotations`, keyword-only constructors, module docstrings, focused files, injected collaborators).

## Scope Boundary (explicit deferrals)

- **Real Roboflow network calls** (actual upload/download against a project + API key) → box-only smoke. CI verifies the client's orchestration against a fake transport; the `RoboflowSdkTransport` methods are thin SDK pass-throughs validated manually on the user's box (`pip install roboflow`).
- **Dataset-version generation / annotation / Label-Assist** → done in the Roboflow web UI (spec §12 step 2 mentions Label Assist + active learning as a UI workflow); not automated here.
- **ONNX/TensorRT export + engine detector backend** → Plan 6C.
- **Wiring the miner into the live worker** (collecting + auto-pushing during play) → box-only follow-up once the FrameGrabber is live-wired (Phase 6A deferral).

---

## File Structure

**New files:**
- `src/ragnarok/training/roboflow_client.py` — `Transport` Protocol, `RoboflowClient`, `RoboflowSdkTransport` (lazy real adapter), `build_roboflow_transport` (env+config factory).
- `tests/training/test_roboflow_client.py` — client + miner tests against a fake transport.
- `tests/training/test_roboflow_transport.py` — factory guard tests + lazy-import test.

**Modified files:** none (Phase 6A's `TrainingConfig` and `select_hard_examples` are consumed as-is).

---

## Task 1: Transport Protocol + RoboflowClient (upload + download)

**Files:**
- Create: `src/ragnarok/training/roboflow_client.py`
- Create: `tests/training/test_roboflow_client.py`

**Interfaces:**
- Consumes: nothing (stdlib).
- Produces:
  - `Transport` (`typing.Protocol`): `upload_image(self, image_path: str, *, split: str) -> str` (returns an image ref/id); `download(self, version: int, fmt: str, dest: str) -> str` (returns the local path written).
  - `RoboflowClient(transport, *, default_split: str = "train")` with:
    - `upload_frames(image_paths, *, split: str | None = None) -> list[str]` — uploads each path via the transport (using `split` or `default_split`), returns the list of refs.
    - `download_version(version: int, dest: str, *, fmt: str = "coco") -> str` — downloads via the transport, returns the dest path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_roboflow_client.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_roboflow_client.py -v`
Expected: FAIL — `No module named 'ragnarok.training.roboflow_client'`.

- [ ] **Step 3: Implement Transport + RoboflowClient**

```python
# src/ragnarok/training/roboflow_client.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_roboflow_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/training/roboflow_client.py tests/training/test_roboflow_client.py
git commit -m "feat(training): RoboflowClient upload/download over an injected Transport"
```

---

## Task 2: Hard-example miner (push_hard_examples)

**Files:**
- Modify: `src/ragnarok/training/roboflow_client.py`
- Test: `tests/training/test_roboflow_client.py` (extend)

**Interfaces:**
- Consumes: `select_hard_examples` (`ragnarok.training.hard_examples`, Phase 6A — signature `select_hard_examples(records, *, conf_threshold) -> list` of item ids); `RoboflowClient.upload_frames` (Task 1).
- Produces: `RoboflowClient.push_hard_examples(records, frames_by_id, *, conf_threshold, split: str | None = None) -> list[str]` — `records` is the Phase-6A list of `(item_id, max_confidence | None)`; `frames_by_id` maps `item_id -> image_path`. Selects hard ids via `select_hard_examples`, maps them to paths (skipping ids absent from `frames_by_id`), uploads them, and returns the uploaded refs. This is the §12-step-6 active-learning loop: low-confidence/missed frames flow back to Roboflow for the next dataset version.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/training/test_roboflow_client.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_roboflow_client.py -k push_hard -v`
Expected: FAIL — `RoboflowClient` has no `push_hard_examples`.

- [ ] **Step 3: Implement push_hard_examples**

Add the method to `RoboflowClient` in `src/ragnarok/training/roboflow_client.py`:

```python
    def push_hard_examples(self, records, frames_by_id, *, conf_threshold: float,
                           split: str | None = None) -> list[str]:
        from ragnarok.training.hard_examples import select_hard_examples
        hard_ids = select_hard_examples(records, conf_threshold=conf_threshold)
        paths = [frames_by_id[i] for i in hard_ids if i in frames_by_id]
        return self.upload_frames(paths, split=split)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_roboflow_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/training/roboflow_client.py tests/training/test_roboflow_client.py
git commit -m "feat(training): hard-example miner push_hard_examples (active-learning loop)"
```

---

## Task 3: Real SDK transport + env/config factory

**Files:**
- Modify: `src/ragnarok/training/roboflow_client.py`
- Test: `tests/training/test_roboflow_transport.py`

**Interfaces:**
- Consumes: `AppConfig`/`TrainingConfig` (Phase 6A — `roboflow_workspace`, `roboflow_project`); `os.environ`.
- Produces:
  - `RoboflowSdkTransport(*, api_key, workspace, project)` implementing `Transport` — `__init__` lazily `import roboflow` and builds `roboflow.Roboflow(api_key=api_key).workspace(workspace).project(project)`; `upload_image` → `project.upload(image_path, split=split)` (returns `image_path`); `download` → `project.version(version).download(fmt, location=dest)` (returns `dest`). **Box-only** (real SDK + network).
  - `build_roboflow_transport(cfg) -> RoboflowSdkTransport` — reads `RAGNAROK_ROBOFLOW_API_KEY` from the environment (raises `RuntimeError` with a clear message if missing), validates `cfg.training.roboflow_workspace`/`roboflow_project` are set (raises `RuntimeError` if empty), then constructs the transport. The guard checks happen BEFORE the lazy SDK import, so the missing-key/missing-config paths are unit-testable without `roboflow` installed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/training/test_roboflow_transport.py
"""Tests for the Roboflow transport factory (CI-safe: no roboflow import reached)."""
from __future__ import annotations
import pytest
from ragnarok.config.schema import AppConfig
from ragnarok.training import roboflow_client as rc


def test_module_imports_without_roboflow_installed():
    # The real SDK import is lazy (inside RoboflowSdkTransport), so importing the
    # module must succeed even though `roboflow` is not a test dependency.
    assert hasattr(rc, "RoboflowSdkTransport")
    assert hasattr(rc, "build_roboflow_transport")


def test_build_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("RAGNAROK_ROBOFLOW_API_KEY", raising=False)
    cfg = AppConfig(training={"roboflow_workspace": "ws", "roboflow_project": "proj"})
    with pytest.raises(RuntimeError, match="RAGNAROK_ROBOFLOW_API_KEY"):
        rc.build_roboflow_transport(cfg)


def test_build_raises_without_workspace_or_project(monkeypatch):
    monkeypatch.setenv("RAGNAROK_ROBOFLOW_API_KEY", "k")
    cfg = AppConfig()                                # workspace/project default ""
    with pytest.raises(RuntimeError, match="roboflow_workspace"):
        rc.build_roboflow_transport(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/training/test_roboflow_transport.py -v`
Expected: FAIL — `RoboflowSdkTransport` / `build_roboflow_transport` not defined.

- [ ] **Step 3: Implement RoboflowSdkTransport + build_roboflow_transport**

Append to `src/ragnarok/training/roboflow_client.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/training/test_roboflow_transport.py -v`
Expected: PASS (both guard tests raise before the lazy `import roboflow` is reached; the import test confirms laziness).

- [ ] **Step 5: Run the FULL suite + commit**

Run: `python -m pytest -q`
Expected: PASS (all prior + Phase 6B).

```bash
git add src/ragnarok/training/roboflow_client.py tests/training/test_roboflow_transport.py
git commit -m "feat(training): lazy RoboflowSdkTransport + env/config factory (box-only real path)"
```

---

## Phase 6B completion checklist

- [ ] `RoboflowClient` upload/download over an injected `Transport` (T1).
- [ ] `push_hard_examples` miner reusing Phase 6A's `select_hard_examples` (T2).
- [ ] `RoboflowSdkTransport` (lazy `roboflow` import) + `build_roboflow_transport` env/config factory with guard checks before the import (T3).
- [ ] Full suite green; CI-safe (no network / no `roboflow` import in tests); API key from env, never config/logs; Scope-Boundary deferrals (real network smoke, version-generation/annotation, 6C export, live miner wiring) documented.

After merge: update memory (Phase 6B done — Roboflow upload/download/miner ready behind an injected transport; real path is `pip install roboflow` + `RAGNAROK_ROBOFLOW_API_KEY` + `training.roboflow_workspace/project`). **Box-only smoke:** set the env key + workspace/project, `build_roboflow_transport(cfg)`, upload collected frames, generate+export a COCO version in the Roboflow UI, `download_version`. Natural next: **Plan 6C** (ONNX→TensorRT export + `RFDETROnnx`/`RFDETRTensorRT` detector backend + FP16-vs-INT8 benchmark via 6A's harness), then Phase 7 (Arduino) / Phase 8 (Cyberpunk GUI).
