# Phase 8E — Config Profiles + Cross-Panel Refresh (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save/load/delete named config profiles (per-weapon/per-game presets, spec §13), and make loading a profile refresh every settings tab and hot-reload the worker — resolving the cross-panel-staleness deferral from 8B–8D along the way.

**Architecture:** A pure `ProfileStore` (file IO over an injected directory, reusing `save_config`/`load_config`, with path-traversal-safe name validation) is the CI surface. A thin `ProfilesPanel(QWidget)` lists profiles and drives save/load/delete, emitting `configChanged` like the other panels. `TuningPanel` gains a signal-blocked `refresh()` so a swapped config repaints its widgets. The app wires a Profiles tab and, on any `configChanged`, refreshes all tabs before reloading the worker.

**Tech Stack:** Python 3.11+, `pathlib`, existing `config.store` (`save_config`/`load_config`/`ConfigHandle`), PySide6 (`QComboBox`/`QLineEdit`/`QPushButton`, signal blocking), pytest-qt (`tmp_path` for the store). No torch/GPU/network/SendInput in any test.

## Global Constraints

- **Profile names are path-safe.** Reject anything outside `^[A-Za-z0-9 _-]+$` (no separators, no `..`, non-empty) with `ValueError`; profiles live only inside the injected directory. A profile file is `<dir>/<name>.toml`.
- **`ProfileStore.load` must not auto-create.** `load_config` writes a default file when the path is missing; the store must `exists()`-guard and raise `FileNotFoundError` for an unknown profile.
- **Refresh is signal-blocked.** `TuningPanel.refresh()` re-reads widgets from `handle.current` with `blockSignals(True/False)` around each `setValue`/`setChecked`/`setCurrentText`, so repainting never re-fires `_commit` (which would feed back / needlessly reload).
- **Reuse, don't duplicate.** Save/load go through `config.store.save_config`/`load_config`; profiles funnel through `ConfigHandle.swap` + `WorkerReloader.reload` exactly like the other panels.
- **Additive/backward-compatible:** `ProfileStore`, `ProfilesPanel`, and `TuningPanel.refresh` are new; existing signatures unchanged.
- TDD, one deliverable per task, commit per task. Runner: `uv run --extra dev pytest`. Baseline: **578 passed**.

---

## File Structure

- **Create** `src/ragnarok/config/profiles.py` — `ProfileStore`.
- **Modify** `src/ragnarok/gui/tuning_panel.py` — add `refresh()`.
- **Create** `src/ragnarok/gui/profiles_panel.py` — `ProfilesPanel(QWidget)`.
- **Modify** `src/ragnarok/app.py` — Profiles tab + refresh-all-on-configChanged (box-only glue).
- **Create** tests: `tests/config/test_profiles.py`, `tests/gui/test_profiles_panel.py`; extend `tests/gui/test_tuning_panel.py`.

---

### Task 1: `ProfileStore` — named config file store

**Files:**
- Create: `src/ragnarok/config/profiles.py`
- Test: `tests/config/test_profiles.py`

**Interfaces:**
- Consumes: `config.store.save_config/load_config`, `config.schema.AppConfig`.
- Produces: `ProfileStore(directory)` with `list() -> list[str]` (sorted), `save(name, cfg) -> Path`, `load(name) -> AppConfig`, `delete(name) -> None`, `exists(name) -> bool`, `path_for(name) -> Path` (validates the name).

- [ ] **Step 1: Write the failing test** — `tests/config/test_profiles.py`:

```python
import pytest
from ragnarok.config.schema import AppConfig
from ragnarok.config.profiles import ProfileStore


def test_save_list_load_roundtrip(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    assert store.list() == []
    cfg = AppConfig().model_copy(
        update={"aim": AppConfig().aim.model_copy(update={"kp": 0.77})})
    store.save("AK-47", cfg)
    assert store.list() == ["AK-47"] and store.exists("AK-47")
    loaded = store.load("AK-47")
    assert loaded.aim.kp == 0.77


def test_list_is_sorted_and_delete_removes(tmp_path):
    store = ProfileStore(tmp_path)
    store.save("zebra", AppConfig())
    store.save("alpha", AppConfig())
    assert store.list() == ["alpha", "zebra"]
    store.delete("alpha")
    assert store.list() == ["zebra"]
    store.delete("missing")                      # deleting a non-existent name is a no-op


def test_load_missing_raises_not_autocreate(tmp_path):
    store = ProfileStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")
    assert store.list() == []                    # load must NOT have created a file


def test_invalid_names_rejected(tmp_path):
    store = ProfileStore(tmp_path)
    for bad in ("", "..", "a/b", "a\\b", "we/../etc"):
        with pytest.raises(ValueError):
            store.path_for(bad)


def test_list_empty_when_dir_absent(tmp_path):
    assert ProfileStore(tmp_path / "does_not_exist").list() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/config/test_profiles.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/config/profiles.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/config/test_profiles.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/profiles.py tests/config/test_profiles.py
git commit -m "feat(profiles): path-safe ProfileStore (save/load/list/delete)"
```

---

### Task 2: `TuningPanel.refresh()` — repaint widgets from the live config

**Files:**
- Modify: `src/ragnarok/gui/tuning_panel.py`
- Test: `tests/gui/test_tuning_panel.py`

**Interfaces:**
- Produces: `TuningPanel.refresh() -> None` — re-reads every widget from `handle.current` with signals blocked (no `_commit` re-fire).

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_tuning_panel.py`:

```python
def test_refresh_repaints_widgets_without_recommitting(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    # swap in a new config behind the panel's back, then refresh
    new = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(
        update={"kp": 1.23, "enabled": True, "aimer": "hybrid"})})
    h.swap(new)
    with qtbot.assertNotEmitted(panel.configChanged):        # refresh must not re-commit
        panel.refresh()
    assert panel.widget_for("aim.kp").value() == 1.23
    assert panel.widget_for("aim.enabled").isChecked() is True
    assert panel.widget_for("aim.aimer").currentText() == "hybrid"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_tuning_panel.py::test_refresh_repaints_widgets_without_recommitting -q`
Expected: FAIL (`refresh` undefined).

- [ ] **Step 3: Implement** — in `src/ragnarok/gui/tuning_panel.py`, add a method to `TuningPanel` (e.g. after `widget_for`):

```python
    def refresh(self) -> None:
        """Repaint every widget from the live config (e.g. after a profile load).

        Signals are blocked so setting a value does not re-fire _commit (which
        would feed back into ConfigHandle / trigger a needless worker reload)."""
        cfg = self._handle.current
        for path, w in self._widgets.items():
            value = get_field(cfg, path)
            w.blockSignals(True)
            try:
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(value))
                else:
                    w.setValue(float(value))
            finally:
                w.blockSignals(False)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_tuning_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tuning_panel.py tests/gui/test_tuning_panel.py
git commit -m "feat(profiles): TuningPanel.refresh repaints widgets from live config"
```

---

### Task 3: `ProfilesPanel` widget

**Files:**
- Create: `src/ragnarok/gui/profiles_panel.py`
- Test: `tests/gui/test_profiles_panel.py`

**Interfaces:**
- Consumes: `ProfileStore` (Task 1), `ConfigHandle`.
- Produces: `ProfilesPanel(store, handle)` — a `QWidget` with:
  - `configChanged` Signal (emits the loaded `AppConfig`).
  - `combo` (`QComboBox` of names), `name_edit` (`QLineEdit`).
  - `_save_as()`, `_load()`, `_delete()`, `_refresh_list()`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_profiles_panel.py`:

```python
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.config.profiles import ProfileStore
from ragnarok.gui.profiles_panel import ProfilesPanel


def _panel(tmp_path, cfg=None):
    store = ProfileStore(tmp_path / "profiles")
    handle = ConfigHandle(cfg or AppConfig())
    return ProfilesPanel(store, handle), store, handle


def test_save_as_writes_profile_and_lists_it(qtbot, tmp_path):
    panel, store, handle = _panel(tmp_path)
    qtbot.addWidget(panel)
    panel.name_edit.setText("Sniper")
    panel._save_as()
    assert "Sniper" in store.list()
    assert panel.combo.findText("Sniper") >= 0


def test_load_swaps_handle_and_emits(qtbot, tmp_path):
    # a profile saved with kp=0.9, then loaded into a handle holding defaults
    store = ProfileStore(tmp_path / "profiles")
    saved = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"kp": 0.9})})
    store.save("Rifle", saved)
    handle = ConfigHandle(AppConfig())
    panel = ProfilesPanel(store, handle)
    qtbot.addWidget(panel)
    panel.combo.setCurrentText("Rifle")
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._load()
    assert handle.current.aim.kp == 0.9
    assert blocker.args[0].aim.kp == 0.9


def test_delete_removes_profile(qtbot, tmp_path):
    panel, store, handle = _panel(tmp_path)
    qtbot.addWidget(panel)
    store.save("Temp", AppConfig())
    panel._refresh_list()
    panel.combo.setCurrentText("Temp")
    panel._delete()
    assert "Temp" not in store.list()


def test_save_as_blank_name_is_noop(qtbot, tmp_path):
    panel, store, handle = _panel(tmp_path)
    qtbot.addWidget(panel)
    panel.name_edit.setText("   ")
    panel._save_as()
    assert store.list() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_profiles_panel.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/profiles_panel.py`:

```python
"""Config profiles tab (per-weapon/per-game presets, spec §13).

Thin Qt shell over ProfileStore. Load funnels through ConfigHandle.swap +
configChanged (like the settings panels) so app.py refreshes the tabs and
hot-reloads the worker. Save persists handle.current under a chosen name.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)


class ProfilesPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, store, handle) -> None:
        super().__init__()
        self._store = store
        self._handle = handle

        root = QVBoxLayout(self)
        self.combo = QComboBox()
        root.addWidget(self.combo)

        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("New profile name")
        save = QPushButton("Save as")
        save.clicked.connect(self._save_as)
        row.addWidget(self.name_edit)
        row.addWidget(save)
        root.addLayout(row)

        row2 = QHBoxLayout()
        load = QPushButton("Load")
        load.clicked.connect(self._load)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete)
        row2.addWidget(load)
        row2.addWidget(delete)
        root.addLayout(row2)

        self._refresh_list()

    def _refresh_list(self) -> None:
        current = self.combo.currentText()
        self.combo.clear()
        self.combo.addItems(self._store.list())
        idx = self.combo.findText(current)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _save_as(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            return
        self._store.save(name, self._handle.current)
        self._refresh_list()
        idx = self.combo.findText(name)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def _load(self) -> None:
        name = self.combo.currentText()
        if not name:
            return
        cfg = self._store.load(name)
        self._handle.swap(cfg)
        self.configChanged.emit(cfg)

    def _delete(self) -> None:
        name = self.combo.currentText()
        if not name:
            return
        self._store.delete(name)
        self._refresh_list()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_profiles_panel.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/profiles_panel.py tests/gui/test_profiles_panel.py
git commit -m "feat(profiles): ProfilesPanel widget (save-as/load/delete)"
```

---

### Task 4: App wiring — Profiles tab + refresh-all on config change

**Files:**
- Modify: `src/ragnarok/app.py`
- Test: full suite (box-only glue; seams covered by Tasks 1–3 + prior phases).

**Interfaces:**
- Consumes: `ProfileStore`, `ProfilesPanel`, `TuningPanel.refresh`, `WorkerReloader`, `_config_path`.

- [ ] **Step 1: Wire it** — in `src/ragnarok/app.py`:

Add imports:

```python
from ragnarok.config.profiles import ProfileStore
from ragnarok.gui.profiles_panel import ProfilesPanel
```

Add a profiles-dir helper next to `_config_path`:

```python
def _profiles_dir() -> Path:
    return _config_path().parent / "profiles"
```

Collect the tuning panels as they are built and add a refresh-all path. Replace the tab-building block from Phase 8D:

```python
    tabs = QTabWidget()
    aim_panel = TuningPanel(handle, on_save=_save)
    aim_panel.configChanged.connect(reloader.reload)
    tabs.addTab(aim_panel, "Aim")
    diagnostics = DiagnosticsPanel(handle)
    diagnostics.configChanged.connect(reloader.reload)
    tabs.addTab(diagnostics, "Diagnostics")
    for fields, title in ((TRACKING_FIELDS, "Tracking"),
                          (CLASSIFICATION_FIELDS, "Friend/Foe"),
                          (TRIGGER_FIELDS, "Trigger"),
                          (RECOIL_FIELDS, "Recoil"),
                          (MOTION_FIELDS, "Motion")):
        p = TuningPanel(handle, fields=fields, on_save=_save)
        p.configChanged.connect(reloader.reload)
        tabs.addTab(p, title)
```

with:

```python
    tuning_panels: list[TuningPanel] = []

    def _on_config_changed(new_cfg):
        # any panel edit / profile load: repaint the other tabs, then hot-reload
        for tp in tuning_panels:
            tp.refresh()
        reloader.reload(new_cfg)

    tabs = QTabWidget()
    aim_panel = TuningPanel(handle, on_save=_save)
    aim_panel.configChanged.connect(_on_config_changed)
    tuning_panels.append(aim_panel)
    tabs.addTab(aim_panel, "Aim")
    diagnostics = DiagnosticsPanel(handle)
    diagnostics.configChanged.connect(_on_config_changed)
    tabs.addTab(diagnostics, "Diagnostics")
    for fields, title in ((TRACKING_FIELDS, "Tracking"),
                          (CLASSIFICATION_FIELDS, "Friend/Foe"),
                          (TRIGGER_FIELDS, "Trigger"),
                          (RECOIL_FIELDS, "Recoil"),
                          (MOTION_FIELDS, "Motion")):
        p = TuningPanel(handle, fields=fields, on_save=_save)
        p.configChanged.connect(_on_config_changed)
        tuning_panels.append(p)
        tabs.addTab(p, title)
    profiles = ProfilesPanel(ProfileStore(_profiles_dir()), handle)
    profiles.configChanged.connect(_on_config_changed)
    tabs.addTab(profiles, "Profiles")
```

(Leave the rest of `main()` unchanged.)

- [ ] **Step 2: Verify the app imports cleanly (offscreen)**

Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 578 baseline + new tests, no regressions.

- [ ] **Step 4: Commit**

```bash
git add src/ragnarok/app.py
git commit -m "feat(profiles): Profiles tab + refresh-all tabs on any config change"
```

---

## Self-Review

**Spec coverage (§13 profiles/config):**
- Per-weapon/per-game profiles (save/load/list/delete) → `ProfileStore` + `ProfilesPanel` (Tasks 1, 3). ✅
- Import/export/presets → named-TOML save/load *is* the presets mechanism; explicit file import/export dialogs are a later box-only add.
- Load funnels through the immutable snapshot swap → `_load` → `ConfigHandle.swap` → `configChanged` → `WorkerReloader.reload` (Tasks 3, 4). ✅
- Loading refreshes the UI → `_on_config_changed` refreshes all tabs (Tasks 2, 4) — **resolves the 8B–8D cross-panel-staleness deferral**. ✅

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** `ProfileStore(directory)` methods (Task 1) are consumed by `ProfilesPanel(store, handle)` (Task 3) and app (Task 4). `TuningPanel.refresh()` (Task 2) is called from `_on_config_changed` (Task 4). `configChanged.emit(cfg)` matches `WorkerReloader.reload(cfg)` / `_on_config_changed(new_cfg)`. `_profiles_dir()` derives from the existing `_config_path()`.

**Honest deferrals (box-only / later):** explicit file import/export dialogs, profile rename, an "active profile" indicator / auto-load-last, weapon auto-detect (HUD CV) to auto-switch profiles (spec §13), config versioning/migration, and the Cyberpunk styling are noted, not silently dropped. Name validation is intentionally strict (alphanumeric + space/`-`/`_`) to keep profiles inside their directory.
