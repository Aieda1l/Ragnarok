# Phase 8B — Live Aim Tuning Panel (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the aim system tunable live from the GUI: an "Aim" control panel whose edits produce a new immutable `AppConfig`, funnel through `ConfigHandle.swap`, and hot-reload the running worker's aim controller — the spec §13 "immutable snapshot swap".

**Architecture:** A **Qt-free pure binding layer** (`gui/tuning_model.py`) describes each tunable aim field (`FieldSpec`) and applies edits by re-validating a new frozen `AppConfig` (`set_field`/`apply_field`). A **reload coordinator** (`gui/live_config.py`) turns a swapped config into a freshly-rebuilt `AimController` and atomically rebinds it into the loop via a new `WorkerLoop.set_aim_controller` seam. A thin `TuningPanel(QWidget)` builds rows from the field specs, applies on commit, and emits `configChanged`. The actual SendInput-backed controller rebuild stays box-only (injected builder); every seam around it is CI-tested.

**Tech Stack:** Python 3.11+, pydantic v2 (`model_dump` + validating re-construction), PySide6 (`QWidget`/`QDoubleSpinBox`/`QComboBox`/`QCheckBox`), pytest + pytest-qt. No torch/GPU/network/SendInput in any test.

## Global Constraints

- **Rebuild-on-swap is the hot-reload model** (spec §13). A config swap rebuilds the whole `AimController` from the new frozen `AppConfig`, so *all* aim fields take effect at once. No field mutates a live controller in place.
- **`set_field` MUST re-validate.** pydantic v2 `model_copy(update=...)` bypasses validation; construct the sub-model via its class so out-of-range values raise `ValidationError`. Out-of-range edits must never reach `ConfigHandle`.
- **Reload is aim-scoped this slice.** Only the `aim` section is surfaced/reloaded; other sections are preserved untouched by `set_field`. Other tabs' live-reload arrive with those tabs.
- **CI-safety:** `tuning_model.py` and `live_config.py` import **zero Qt** and **zero SendInput/torch**. The controller builder is injected (real one in `app.py` is box-only). Widget tests use `qtbot`.
- **Backward-compatible seams:** `WorkerLoop.set_aim_controller` and `MainWindow(controls=...)` are additive; existing constructors/tests keep working.
- **The panel is the only writer to the handle** in this slice; the worker is the only reader (via the rebuilt controller). Single-writer/single-reader, GIL-atomic swap — no locks.
- TDD, one deliverable per task, commit per task. Test runner: `uv run --extra dev pytest`. Baseline: **538 passed**.

---

## File Structure

- **Create** `src/ragnarok/gui/tuning_model.py` — `FieldSpec`, `AIM_FIELDS`, `get_field`, `set_field`, `apply_field`. Qt-free, pure.
- **Create** `src/ragnarok/gui/live_config.py` — `AimReloader` (config-swap → rebuilt controller → `loop.set_aim_controller`). Qt-free; injected builder.
- **Create** `src/ragnarok/gui/tuning_panel.py` — `TuningPanel(QWidget)`.
- **Modify** `src/ragnarok/worker/loop.py` — add `set_aim_controller`.
- **Modify** `src/ragnarok/gui/main_window.py` — optional `controls` widget slot.
- **Modify** `src/ragnarok/app.py` — build `ConfigHandle`, `TuningPanel`, `AimReloader`; connect (box-only).
- **Create** tests: `tests/gui/test_tuning_model.py`, `tests/gui/test_live_config.py`, `tests/gui/test_tuning_panel.py`; extend `tests/worker/test_loop.py`, `tests/gui/test_main_window.py`.

---

### Task 1: `tuning_model.py` — field specs + validating get/set/apply

**Files:**
- Create: `src/ragnarok/gui/tuning_model.py`
- Test: `tests/gui/test_tuning_model.py`

**Interfaces:**
- Consumes: `config.schema.AppConfig`, `config.store.ConfigHandle`.
- Produces:
  - `FieldSpec(path, label, kind, minimum=None, maximum=None, step=None, choices=())` where `kind` ∈ `{"float","int","bool","choice"}` and `path` is `"section.field"`.
  - `AIM_FIELDS: tuple[FieldSpec, ...]`.
  - `get_field(cfg, path) -> value`.
  - `set_field(cfg, path, value) -> AppConfig` (re-validates; raises on invalid).
  - `apply_field(handle, path, value) -> AppConfig` (set + swap + return new cfg).

- [ ] **Step 1: Write the failing test** — `tests/gui/test_tuning_model.py`:

```python
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.tuning_model import (
    AIM_FIELDS, FieldSpec, get_field, set_field, apply_field)


def test_aim_fields_cover_key_knobs_and_are_wellformed():
    paths = {f.path for f in AIM_FIELDS}
    assert {"aim.enabled", "aim.kp", "aim.aimer", "aim.aim_fov_deg"} <= paths
    for f in AIM_FIELDS:
        assert f.path.startswith("aim.")
        assert f.kind in {"float", "int", "bool", "choice"}
        if f.kind == "choice":
            assert len(f.choices) >= 2


def test_get_and_set_roundtrip_preserves_other_sections():
    cfg = AppConfig()
    assert get_field(cfg, "aim.kp") == cfg.aim.kp
    new = set_field(cfg, "aim.kp", 0.8)
    assert new.aim.kp == 0.8
    assert cfg.aim.kp != 0.8                      # original untouched (frozen)
    assert new.detection == cfg.detection          # other sections preserved
    assert new.capture == cfg.capture


def test_set_field_revalidates_and_rejects_out_of_range():
    cfg = AppConfig()
    with pytest.raises(ValidationError):
        set_field(cfg, "aim.kp", 99.0)             # schema: kp <= 2.0
    with pytest.raises(ValidationError):
        set_field(cfg, "aim.switch_margin", 1.5)   # schema: < 1.0


def test_apply_field_swaps_handle():
    h = ConfigHandle(AppConfig())
    returned = apply_field(h, "aim.enabled", True)
    assert h.current.aim.enabled is True
    assert returned is h.current
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_tuning_model.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/tuning_model.py`:

```python
"""Pure binding layer for the live tuning panels (spec §13).

ZERO Qt: describes each tunable field (``FieldSpec``) and applies an edit by
building a NEW, RE-VALIDATED frozen ``AppConfig``. pydantic v2's
``model_copy(update=...)`` skips validation, so ``set_field`` reconstructs the
edited sub-model through its class — an out-of-range value raises
``ValidationError`` and never reaches the ``ConfigHandle``.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field

from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle


@dataclass(frozen=True)
class FieldSpec:
    path: str                     # "section.field"
    label: str
    kind: str                     # "float" | "int" | "bool" | "choice"
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()


# The "Aim" tab. Ranges mirror config.schema.AimConfig Field() constraints.
AIM_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("aim.enabled", "Aim enabled", "bool"),
    FieldSpec("aim.aimer", "Aimer", "choice",
              choices=("flick", "feedback", "hybrid", "predictive")),
    FieldSpec("aim.controller_mode", "PID mode", "choice", choices=("p", "pi", "pid")),
    FieldSpec("aim.kp", "Kp", "float", 0.01, 2.0, 0.01),
    FieldSpec("aim.ki", "Ki", "float", 0.0, 5.0, 0.01),
    FieldSpec("aim.kd", "Kd", "float", 0.0, 5.0, 0.01),
    FieldSpec("aim.kff", "Kff (feed-fwd)", "float", 0.0, 4.0, 0.05),
    FieldSpec("aim.max_step_px", "Max step (px)", "float", 1.0, 300.0, 1.0),
    FieldSpec("aim.ema_alpha", "EMA alpha", "float", 0.01, 1.0, 0.01),
    FieldSpec("aim.aim_fov_deg", "FOV acquire (deg)", "float", 0.1, 179.0, 0.5),
    FieldSpec("aim.retain_fov_deg", "FOV retain (deg)", "float", 0.1, 179.0, 0.5),
    FieldSpec("aim.dwell_ms", "Dwell (ms)", "float", 0.0, 2000.0, 10.0),
    FieldSpec("aim.switch_margin", "Switch margin", "float", 0.0, 0.99, 0.01),
    FieldSpec("aim.sensitivity", "Sensitivity (deg/count)", "float", 0.001, 1.0, 0.001),
    FieldSpec("aim.lead_ms", "Lead (ms)", "float", 0.0, 500.0, 5.0),
    FieldSpec("aim.head_frac", "Head fraction", "float", 0.0, 1.0, 0.01),
    FieldSpec("aim.aim_point", "Aim point", "choice", choices=("head", "body")),
    FieldSpec("aim.adaptive_lead", "Adaptive lead", "bool"),
)


def _split(path: str) -> tuple[str, str]:
    section, field = path.split(".", 1)
    return section, field


def get_field(cfg: AppConfig, path: str):
    section, field = _split(path)
    return getattr(getattr(cfg, section), field)


def set_field(cfg: AppConfig, path: str, value) -> AppConfig:
    """Return a NEW frozen AppConfig with ``path`` set to ``value``.

    Re-validates by reconstructing the sub-model through its class; an invalid
    value raises ``pydantic.ValidationError``.
    """
    section, field = _split(path)
    sub = getattr(cfg, section)
    new_sub = sub.__class__(**{**sub.model_dump(), field: value})   # validates
    return cfg.model_copy(update={section: new_sub})


def apply_field(handle: ConfigHandle, path: str, value) -> AppConfig:
    new_cfg = set_field(handle.current, path, value)
    handle.swap(new_cfg)
    return new_cfg
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_tuning_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tuning_model.py tests/gui/test_tuning_model.py
git commit -m "feat(tuning): pure FieldSpec binding layer with validating set_field"
```

---

### Task 2: `WorkerLoop.set_aim_controller` seam

**Files:**
- Modify: `src/ragnarok/worker/loop.py`
- Test: `tests/worker/test_loop.py`

**Interfaces:**
- Produces: `WorkerLoop.set_aim_controller(controller_or_none) -> None` (atomic rebind; next `tick()` uses it).

- [ ] **Step 1: Write the failing test** — append to `tests/worker/test_loop.py`:

```python
def test_set_aim_controller_hotswaps_and_can_disable():
    class _Aim:
        def __init__(self, tid):
            self.target_id = tid
            self.calls = 0
        def update(self, tracks, t_ns):
            self.calls += 1
    a1, a2 = _Aim(1), _Aim(2)
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub, aim_controller=a1)
    loop.tick()
    assert a1.calls == 1 and pub.latest().locked_target_id == 1
    loop.set_aim_controller(a2)
    loop.tick()
    assert a2.calls == 1 and pub.latest().locked_target_id == 2
    loop.set_aim_controller(None)                  # disable aim live
    loop.tick()
    assert pub.latest().locked_target_id is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/worker/test_loop.py::test_set_aim_controller_hotswaps_and_can_disable -q`
Expected: FAIL (`set_aim_controller` undefined).

- [ ] **Step 3: Implement** — in `src/ragnarok/worker/loop.py`, add a method to `WorkerLoop` (e.g. after `__init__`):

```python
    def set_aim_controller(self, controller) -> None:
        """Atomically hot-swap the aim controller (or None to disable aim).

        Single attribute rebind -> GIL-atomic; the tick loop reads self._aim
        once per iteration, so it always sees a whole controller or None.
        """
        self._aim = controller
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/worker/test_loop.py -q`
Expected: PASS (new + existing loop tests).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/worker/loop.py tests/worker/test_loop.py
git commit -m "feat(tuning): WorkerLoop.set_aim_controller hot-swap seam"
```

---

### Task 3: `live_config.py` — `AimReloader`

**Files:**
- Create: `src/ragnarok/gui/live_config.py`
- Test: `tests/gui/test_live_config.py`

**Interfaces:**
- Consumes: `WorkerLoop.set_aim_controller` (Task 2), an injected `build_aim(cfg, commanded_buffer) -> controller`, `AppConfig`.
- Produces: `AimReloader(loop, build_aim, commanded_buffer=None)` with `reload(cfg: AppConfig) -> None` — rebuilds + sets the controller when `cfg.aim.enabled`, else sets `None`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_live_config.py`:

```python
from ragnarok.config.schema import AppConfig
from ragnarok.gui.live_config import AimReloader


class _Loop:
    def __init__(self):
        self.controller = "sentinel"
    def set_aim_controller(self, c):
        self.controller = c


def test_reload_builds_and_sets_when_enabled():
    loop = _Loop()
    seen = {}
    def build(cfg, buf):
        seen["cfg"] = cfg
        seen["buf"] = buf
        return "CTRL"
    r = AimReloader(loop, build, commanded_buffer="BUF")
    cfg = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"enabled": True})})
    r.reload(cfg)
    assert loop.controller == "CTRL"
    assert seen["cfg"] is cfg and seen["buf"] == "BUF"


def test_reload_disables_without_building_when_aim_off():
    loop = _Loop()
    called = {"n": 0}
    def build(cfg, buf):
        called["n"] += 1
        return "CTRL"
    r = AimReloader(loop, build)
    r.reload(AppConfig())                           # aim.enabled defaults False
    assert loop.controller is None
    assert called["n"] == 0                          # no rebuild when disabled
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_live_config.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/live_config.py`:

```python
"""Config-swap -> live worker reload coordinator (spec §13, aim-scoped).

ZERO Qt / SendInput: turns a freshly-swapped ``AppConfig`` into a rebuilt aim
controller and atomically rebinds it into the running loop. The controller
builder is INJECTED so this is unit-testable without SendInput/torch; ``app.py``
passes the real (box-only) builder.

Rebuild-on-swap (not in-place mutation) is the model: a full rebuild makes every
aim field take effect at once and starts the controller cleanly disengaged.
"""
from __future__ import annotations


class AimReloader:
    def __init__(self, loop, build_aim, commanded_buffer=None) -> None:
        self._loop = loop
        self._build = build_aim
        self._buf = commanded_buffer

    def reload(self, cfg) -> None:
        if cfg.aim.enabled:
            self._loop.set_aim_controller(self._build(cfg, self._buf))
        else:
            self._loop.set_aim_controller(None)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_live_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/live_config.py tests/gui/test_live_config.py
git commit -m "feat(tuning): AimReloader rebuilds+hot-swaps controller on config change"
```

---

### Task 4: `MainWindow` optional `controls` slot

**Files:**
- Modify: `src/ragnarok/gui/main_window.py`
- Test: `tests/gui/test_main_window.py`

**Interfaces:**
- Produces: `MainWindow(publisher, controls: QWidget | None = None)` — when given, `controls` is added to the layout and exposed as `self.controls`.

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_main_window.py`:

```python
def test_main_window_embeds_controls_widget(qtbot):
    from PySide6.QtWidgets import QLabel
    from ragnarok.gui.main_window import MainWindow
    from ragnarok.telemetry.snapshot import SnapshotPublisher
    panel = QLabel("controls")
    win = MainWindow(SnapshotPublisher(), controls=panel)
    qtbot.addWidget(win)
    assert win.controls is panel
    assert panel.parent() is not None                # actually parented into the window
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_main_window.py::test_main_window_embeds_controls_widget -q`
Expected: FAIL (unexpected `controls` kwarg).

- [ ] **Step 3: Implement** — in `src/ragnarok/gui/main_window.py`, change the constructor signature and add the widget. Update `__init__` to accept `controls`:

```python
    def __init__(self, publisher: SnapshotPublisher, controls: QWidget | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Ragnarok")
        self._pub = publisher
        central = QWidget()
        layout = QVBoxLayout(central)
        self.preview_label = QLabel("no signal")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.stats_label = QLabel("--")
        layout.addWidget(self.preview_label)
        layout.addWidget(self.stats_label)
        self.controls = controls
        if controls is not None:
            layout.addWidget(controls)
        self.setCentralWidget(central)
```

(Leave the rest of the class — the `QTimer`/`refresh` — unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_main_window.py -q`
Expected: PASS (new + existing 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/main_window.py tests/gui/test_main_window.py
git commit -m "feat(tuning): MainWindow optional controls slot"
```

---

### Task 5: `TuningPanel` widget

**Files:**
- Create: `src/ragnarok/gui/tuning_panel.py`
- Test: `tests/gui/test_tuning_panel.py`

**Interfaces:**
- Consumes: `AIM_FIELDS`, `get_field`, `apply_field` (Task 1); `ConfigHandle`.
- Produces: `TuningPanel(handle, fields=AIM_FIELDS, *, on_save=None)` — a `QWidget` with:
  - `configChanged` Qt Signal (emits the new `AppConfig`).
  - `widget_for(path) -> QWidget`.
  - `_commit(path)` — read the widget, `apply_field`, emit `configChanged`.
  - a Save button calling `on_save(handle.current)` when provided.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_tuning_panel.py`:

```python
from PySide6.QtWidgets import QDoubleSpinBox, QCheckBox, QComboBox
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.tuning_panel import TuningPanel


def test_panel_builds_a_widget_per_field(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    assert isinstance(panel.widget_for("aim.kp"), QDoubleSpinBox)
    assert isinstance(panel.widget_for("aim.enabled"), QCheckBox)
    assert isinstance(panel.widget_for("aim.aimer"), QComboBox)
    # initialised from the handle's current config
    assert panel.widget_for("aim.kp").value() == AppConfig().aim.kp


def test_editing_a_field_swaps_handle_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("aim.kp").setValue(0.9)
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._commit("aim.kp")
    assert h.current.aim.kp == 0.9
    assert blocker.args[0].aim.kp == 0.9


def test_choice_and_bool_commit(qtbot):
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("aim.enabled").setChecked(True)
    panel._commit("aim.enabled")
    assert h.current.aim.enabled is True
    panel.widget_for("aim.aimer").setCurrentText("hybrid")
    panel._commit("aim.aimer")
    assert h.current.aim.aimer == "hybrid"


def test_save_button_invokes_callback(qtbot):
    h = ConfigHandle(AppConfig())
    saved = {}
    panel = TuningPanel(h, on_save=lambda cfg: saved.setdefault("cfg", cfg))
    qtbot.addWidget(panel)
    panel._save()
    assert saved["cfg"] is h.current
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_tuning_panel.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/tuning_panel.py`:

```python
"""Live aim tuning panel (spec §10.3 "Aim" tab, §13 snapshot-swap).

Thin Qt shell over the pure binding layer (``tuning_model``). Each field becomes
a labelled row; committing a row funnels through ``apply_field`` -> the
``ConfigHandle`` swaps -> ``configChanged`` fires so ``app.py`` can hot-reload
the worker. Full Cyberpunk styling is a later box-only pass.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QPushButton, QVBoxLayout,
    QWidget,
)

from ragnarok.gui.tuning_model import AIM_FIELDS, apply_field, get_field


class TuningPanel(QWidget):
    configChanged = Signal(object)          # emits the new AppConfig

    def __init__(self, handle, fields=AIM_FIELDS, *, on_save=None) -> None:
        super().__init__()
        self._handle = handle
        self._fields = tuple(fields)
        self._on_save = on_save
        self._widgets: dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        form = QFormLayout()
        cfg = handle.current
        for spec in self._fields:
            w = self._build_widget(spec, get_field(cfg, spec.path))
            self._widgets[spec.path] = w
            form.addRow(spec.label, w)
        root.addLayout(form)

        save = QPushButton("Save to config")
        save.clicked.connect(self._save)
        root.addWidget(save)

    # -- public ----------------------------------------------------------
    def widget_for(self, path: str) -> QWidget:
        return self._widgets[path]

    # -- construction ----------------------------------------------------
    def _build_widget(self, spec, value) -> QWidget:
        path = spec.path
        if spec.kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.stateChanged.connect(lambda _s, p=path: self._commit(p))
            return w
        if spec.kind == "choice":
            w = QComboBox()
            w.addItems(list(spec.choices))
            w.setCurrentText(str(value))
            w.currentIndexChanged.connect(lambda _i, p=path: self._commit(p))
            return w
        # float / int -> spin box
        w = QDoubleSpinBox()
        w.setDecimals(0 if spec.kind == "int" else 3)
        if spec.minimum is not None:
            w.setMinimum(spec.minimum)
        if spec.maximum is not None:
            w.setMaximum(spec.maximum)
        if spec.step is not None:
            w.setSingleStep(spec.step)
        w.setValue(float(value))
        w.editingFinished.connect(lambda p=path: self._commit(p))
        return w

    # -- commit ----------------------------------------------------------
    def _read(self, path: str):
        w = self._widgets[path]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.value()

    def _commit(self, path: str) -> None:
        new_cfg = apply_field(self._handle, path, self._read(path))
        self.configChanged.emit(new_cfg)

    def _save(self) -> None:
        if self._on_save is not None:
            self._on_save(self._handle.current)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_tuning_panel.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tuning_panel.py tests/gui/test_tuning_panel.py
git commit -m "feat(tuning): TuningPanel widget (rows from field specs, commit->swap)"
```

---

### Task 6: App wiring (box-only) + full suite

**Files:**
- Modify: `src/ragnarok/app.py`
- Test: full suite (no new unit test — this is box-only glue; the seams are covered by Tasks 1–5).

**Interfaces:**
- Consumes: `ConfigHandle`, `TuningPanel`, `AimReloader`, `_build_aim_controller`, `MainWindow(controls=...)`, `WorkerLoop.set_aim_controller`.

- [ ] **Step 1: Wire it** — in `src/ragnarok/app.py`:

Add imports near the other gui imports:

```python
from ragnarok.config.store import load_config, save_config, ConfigHandle
from ragnarok.gui.tuning_panel import TuningPanel
from ragnarok.gui.live_config import AimReloader
```

(The existing `from ragnarok.config.store import load_config` line becomes the combined import above.)

In `main()`, after the loop is built and before `window.show()`, introduce the handle, panel, and reloader. Replace:

```python
    worker = WorkerThread(loop)
    window = MainWindow(publisher)
    window.show()
    # Smart-lock FOV overlay: ...
    overlay = FovOverlay(publisher, lambda: cfg)
```

with:

```python
    handle = ConfigHandle(cfg)
    reloader = AimReloader(loop, _build_aim_controller,
                           commanded_buffer=cmd_buffer)

    def _on_config_changed(new_cfg):
        # rebuild the aim path from the swapped snapshot (spec §13)
        reloader.reload(new_cfg)

    panel = TuningPanel(handle, on_save=lambda c: save_config(c, _config_path()))
    panel.configChanged.connect(_on_config_changed)

    worker = WorkerThread(loop)
    window = MainWindow(publisher, controls=panel)
    window.show()
    # Overlay reads the live config so FOV-ring/aim-point edits show immediately.
    overlay = FovOverlay(publisher, lambda: handle.current)
```

(Keep the rest of `main()` — overlay resize/show, `worker.start()`, `aboutToQuit` — unchanged.)

- [ ] **Step 2: Verify the app imports cleanly (offscreen)**

Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 538 baseline + new tests, no regressions.

- [ ] **Step 4: Commit**

```bash
git add src/ragnarok/app.py
git commit -m "feat(tuning): wire live tuning panel + aim hot-reload into app"
```

---

## Self-Review

**Spec coverage (§10.3 Aim tab, §13 config/hot-reload):**
- In-GUI sliders for aim (aimer/PID/FOV/etc.) → `AIM_FIELDS` + `TuningPanel` (Tasks 1, 5). ✅
- Edits funnel through the immutable snapshot swap → `apply_field` → `ConfigHandle.swap` → `AimReloader.reload` → `set_aim_controller` (Tasks 1, 2, 3). ✅
- Validation on edit → `set_field` reconstructs the sub-model (raises on invalid) (Task 1). ✅
- Persistence to TOML → Save button → injected `save_config` (Tasks 5, 6). ✅
- Overlay reflects live edits (FOV ring / aim-point) → overlay reads `handle.current` (Task 6). ✅
- `QFileSystemWatcher` file-watch hot-reload → **explicitly deferred box-only** (this slice does in-GUI-slider hot-reload; the file-watch path reuses the same `AimReloader.reload` seam later).

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** `FieldSpec`/`AIM_FIELDS`/`get_field`/`set_field`/`apply_field` (Task 1) used by `TuningPanel` (Task 5) and `apply_field`+`ConfigHandle` in app (Task 6). `set_aim_controller` (Task 2) consumed by `AimReloader` (Task 3) and app (Task 6). `MainWindow(controls=...)` (Task 4) used in app (Task 6). `_build_aim_controller(cfg, commanded_buffer)` signature matches the existing `app.py` definition passed to `AimReloader`.

**Honest deferrals:** file-watch (`QFileSystemWatcher`) hot-reload, per-weapon/profile management, live-reload of non-aim sections (detection/tracking/etc. — arrive with their tabs), and the full Cyberpunk styling of the panel are box-only/later — noted here, not silently dropped. Rebuilding the whole `AimController` on each commit briefly resets IMM/lock/lead state (acceptable for user-driven tuning) and, on a live enabled→disabled edit, does not force-release a held trigger button on the *outgoing* controller — a box-only refinement to note when the panel is exercised on the real box.
