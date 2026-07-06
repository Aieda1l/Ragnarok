# Phase 8D — Settings Tabs + Broadened Hot-Reload (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the remaining config sections (Tracking, Friend/Foe, Trigger, Recoil, Motion) live-editable as GUI tabs using the existing generic `TuningPanel`, and broaden the hot-reload coordinator so a config swap rebuilds the tracker and classifier (not just the aim controller) — only the components that actually changed.

**Architecture:** Reuse the Phase 8B pieces unchanged: `FieldSpec` + `set_field`/`apply_field` (pure, validating) and the generic `TuningPanel(handle, fields=...)`. This slice adds five new field-spec tuples, two atomic loop seams (`set_tracker`/`set_classifier`), and a `WorkerReloader` that composes the existing `AimReloader` with tracker/classifier rebuilds — **change-gated** (compares the new frozen `AppConfig` sections against the previous so an aim-slider tweak doesn't needlessly reset tracking). The detector rebuild (torch/TRT) stays box-only.

**Tech Stack:** Python 3.11+, pydantic v2 (frozen-model equality for change detection; auto-coerces integral floats to `int`), PySide6 (`QTabWidget`, reused `TuningPanel`), `ragnarok.wiring.build_tracker/build_classifier`, pytest-qt. No torch/GPU/network/SendInput in any test.

## Global Constraints

- **Reuse the generic panel.** No new widget: each settings tab is `TuningPanel(handle, fields=SECTION_FIELDS)`. Field ranges mirror the pydantic `Field()` constraints in `config/schema.py` exactly.
- **Change-gated reload.** `WorkerReloader.reload(cfg)` rebuilds a component only if its config slice changed vs the previous swap: aim controller on `aim`/`trigger`/`recoil`/`motion` change (or `aim.enabled` toggle), tracker on `tracking` change or `aim.enabled` toggle (GMC buffer gating), classifier on `classification` change. Frozen pydantic models compare by value.
- **Detector reload is box-only.** No Detection tab this slice (a live detector swap reloads the torch/TRT model). Note it; do not wire it.
- **Aim-owned sections.** Trigger/Recoil/Motion take effect via the aim-controller rebuild, which (matching current architecture) only happens while `aim.enabled`. Editing them with aim disabled persists to config and applies when aim is next enabled.
- **Additive, backward-compatible seams:** `set_tracker`/`set_classifier` and `WorkerReloader` are new; `AimReloader` and existing loop/panel signatures are unchanged. Tracker/classifier are read once per tick, so an atomic rebind is race-free.
- **Friend/Foe color (palette + enemy_color) is deferred to the eyedropper wizard** — this tab exposes only the safe threshold/vote knobs (changing palette without a matching enemy_color key could break `resolve_enemy_profile`).
- TDD, one deliverable per task, commit per task. Runner: `uv run --extra dev pytest`. Baseline: **567 passed**.

---

## File Structure

- **Modify** `src/ragnarok/gui/tuning_model.py` — add `TRACKING_FIELDS`, `CLASSIFICATION_FIELDS`, `TRIGGER_FIELDS`, `RECOIL_FIELDS`, `MOTION_FIELDS`.
- **Modify** `src/ragnarok/worker/loop.py` — add `set_tracker` + `set_classifier`.
- **Modify** `src/ragnarok/gui/live_config.py` — add `WorkerReloader`.
- **Modify** `src/ragnarok/app.py` — build per-section tabs; swap `AimReloader` for `WorkerReloader` (box-only glue).
- **Create** tests: `tests/gui/test_settings_fields.py`; extend `tests/worker/test_loop.py`, `tests/gui/test_live_config.py`.

---

### Task 1: field-spec tuples for the remaining sections

**Files:**
- Modify: `src/ragnarok/gui/tuning_model.py`
- Test: `tests/gui/test_settings_fields.py`

**Interfaces:**
- Consumes: existing `FieldSpec`, `set_field` (Phase 8B).
- Produces: `TRACKING_FIELDS`, `CLASSIFICATION_FIELDS`, `TRIGGER_FIELDS`, `RECOIL_FIELDS`, `MOTION_FIELDS` — each a `tuple[FieldSpec, ...]`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_settings_fields.py`:

```python
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig
from ragnarok.gui.tuning_model import (
    set_field, TRACKING_FIELDS, CLASSIFICATION_FIELDS, TRIGGER_FIELDS,
    RECOIL_FIELDS, MOTION_FIELDS)

ALL = (("tracking", TRACKING_FIELDS), ("classification", CLASSIFICATION_FIELDS),
       ("trigger", TRIGGER_FIELDS), ("recoil", RECOIL_FIELDS), ("motion", MOTION_FIELDS))


def test_every_field_targets_its_section_and_is_wellformed():
    for section, fields in ALL:
        assert len(fields) >= 2
        for f in fields:
            assert f.path.startswith(section + ".")
            assert f.kind in {"float", "int", "bool", "choice"}
            if f.kind == "choice":
                assert len(f.choices) >= 2


def test_every_field_get_and_set_roundtrips_on_default_config():
    cfg = AppConfig()
    for _section, fields in ALL:
        for f in fields:
            cur = _sample_value(f, cfg)
            new = set_field(cfg, f.path, cur)               # re-validates; must not raise
            assert new is not cfg


def _sample_value(f, cfg):
    if f.kind == "bool":
        return True
    if f.kind == "choice":
        return f.choices[-1]
    # numeric: midpoint of the declared range (falls back to 1)
    if f.minimum is not None and f.maximum is not None:
        mid = (f.minimum + f.maximum) / 2.0
        return int(mid) if f.kind == "int" else mid
    return 1


def test_int_field_coerces_and_choice_rejects_bad_value():
    new = set_field(AppConfig(), "tracking.track_buffer", 45.0)   # spinbox float -> int
    assert new.tracking.track_buffer == 45 and isinstance(new.tracking.track_buffer, int)
    with pytest.raises(ValidationError):
        set_field(AppConfig(), "tracking.backend", "not_a_backend")


def test_signed_deg_per_count_accepts_negative():
    new = set_field(AppConfig(), "tracking.deg_per_count", -0.05)
    assert new.tracking.deg_per_count == -0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_settings_fields.py -q`
Expected: FAIL (the new field tuples are undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/tuning_model.py` (after `AIM_FIELDS`):

```python
# Ranges mirror config.schema Field() constraints. Detection is box-only (a live
# detector swap reloads the torch/TRT model) and has no tab here.
TRACKING_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("tracking.backend", "Tracker", "choice", choices=("botsort", "identity")),
    FieldSpec("tracking.track_high_thresh", "Track high thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.track_low_thresh", "Track low thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.new_track_thresh", "New-track thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.track_buffer", "Track buffer", "int", 1, 600, 1),
    FieldSpec("tracking.match_thresh", "Match thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.proximity_thresh", "Proximity thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("tracking.gmc", "GMC", "choice", choices=("off", "feedforward")),
    FieldSpec("tracking.deg_per_count", "deg/count (signed)", "float", -1.0, 1.0, 0.001),
    FieldSpec("tracking.tau_render_s", "τ_render (s)", "float", 0.0, 0.1, 0.001),
)

# Palette + enemy_color are the eyedropper wizard's domain (changing palette
# without a matching color key can break resolve_enemy_profile), so only the
# safe threshold/vote knobs are exposed here.
CLASSIFICATION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("classification.enabled", "Friend/Foe enabled", "bool"),
    FieldSpec("classification.frac_threshold", "Outline frac thresh", "float", 0.0, 1.0, 0.01),
    FieldSpec("classification.thickness", "Ring thickness (px)", "int", 1, 64, 1),
    FieldSpec("classification.vote_window", "Vote window", "int", 1, 120, 1),
    FieldSpec("classification.vote_min", "Vote min", "int", 1, 120, 1),
)

TRIGGER_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("trigger.enabled", "Trigger bot enabled", "bool"),
    FieldSpec("trigger.activation_delay_ms", "Activation delay (ms)", "float", 0.0, 2000.0, 5.0),
    FieldSpec("trigger.require_line_clear", "Require line clear", "bool"),
    FieldSpec("trigger.button", "Fire button", "choice", choices=("left", "right", "middle")),
)

RECOIL_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("recoil.enabled", "Recoil comp enabled", "bool"),
    FieldSpec("recoil.scale", "Recoil scale", "float", 0.0, 5.0, 0.1),
)

MOTION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("motion.shaper", "Motion shaper", "choice", choices=("none", "windmouse")),
    FieldSpec("motion.gravity", "WindMouse gravity", "float", 0.0, 50.0, 0.5),
    FieldSpec("motion.wind", "WindMouse wind", "float", 0.0, 20.0, 0.5),
    FieldSpec("motion.max_step", "WindMouse max step", "float", 1.0, 100.0, 1.0),
    FieldSpec("motion.target_area", "WindMouse target area", "float", 1.0, 100.0, 1.0),
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_settings_fields.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tuning_model.py tests/gui/test_settings_fields.py
git commit -m "feat(settings): field specs for tracking/classification/trigger/recoil/motion"
```

---

### Task 2: `WorkerLoop.set_tracker` + `set_classifier` seams

**Files:**
- Modify: `src/ragnarok/worker/loop.py`
- Test: `tests/worker/test_loop.py`

**Interfaces:**
- Produces: `WorkerLoop.set_tracker(tracker) -> None`, `WorkerLoop.set_classifier(classifier) -> None` (atomic rebind; next `tick()` uses them; `None` restores the default).

- [ ] **Step 1: Write the failing test** — append to `tests/worker/test_loop.py`:

```python
def test_set_tracker_and_classifier_hotswap():
    from ragnarok.core.types import Track, Tracks
    class _Trk:
        def __init__(self, tid): self.tid = tid
        def update(self, detections, frame=None):
            return Tracks(items=(Track(track_id=self.tid, xyxy=(0, 0, 10, 10),
                                       confidence=0.9, class_id=0),))
    class _Cls:
        def __init__(self, tag): self.tag = tag; self.seen = 0
        def classify(self, tracks, image): self.seen += 1; return tracks
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), StageProfiler(), pub, tracker=_Trk(1))
    loop.tick()
    assert pub.latest().tracks[0].track_id == 1
    loop.set_tracker(_Trk(9))
    loop.tick()
    assert pub.latest().tracks[0].track_id == 9
    cls = _Cls("x")
    loop.set_classifier(cls)
    loop.tick()
    assert cls.seen == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/worker/test_loop.py::test_set_tracker_and_classifier_hotswap -q`
Expected: FAIL (`set_tracker` undefined).

- [ ] **Step 3: Implement** — in `src/ragnarok/worker/loop.py`, next to `set_aim_controller`, add:

```python
    def set_tracker(self, tracker) -> None:
        """Atomically hot-swap the tracker (None restores the identity default).

        The tick reads self._tracker exactly once per iteration, so a rebind is
        race-free (a tick sees a whole tracker, never a partial one)."""
        self._tracker = tracker if tracker is not None else IdentityTracker()

    def set_classifier(self, classifier) -> None:
        """Atomically hot-swap the friend/foe classifier (None restores null)."""
        self._classifier = classifier if classifier is not None else NullClassifier()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/worker/test_loop.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/worker/loop.py tests/worker/test_loop.py
git commit -m "feat(settings): WorkerLoop.set_tracker + set_classifier hot-swap seams"
```

---

### Task 3: `WorkerReloader` — change-gated rebuild of aim/tracker/classifier

**Files:**
- Modify: `src/ragnarok/gui/live_config.py`
- Test: `tests/gui/test_live_config.py`

**Interfaces:**
- Consumes: `AimReloader` (Phase 8B), `WorkerLoop.set_tracker/set_classifier` (Task 2), injected `build_tracker(cfg, *, gmc_buffer=None)` / `build_classifier(cfg)`.
- Produces: `WorkerReloader(loop, *, aim_reloader, build_tracker, build_classifier, commanded_buffer=None, initial_cfg=None)` with `reload(cfg) -> None` — rebuilds only the components whose config slice changed vs the previous swap.

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_live_config.py`:

```python
from ragnarok.gui.live_config import WorkerReloader


class _FullLoop:
    def __init__(self):
        self.tracker = None
        self.classifier = None
    def set_tracker(self, t): self.tracker = t
    def set_classifier(self, c): self.classifier = c


class _RecordingAim:
    def __init__(self): self.reloads = 0
    def reload(self, cfg): self.reloads += 1


def _make(initial):
    loop = _FullLoop()
    aim = _RecordingAim()
    bt_calls, bc_calls = [], []
    def bt(cfg, *, gmc_buffer=None): bt_calls.append(gmc_buffer); return "T"
    def bc(cfg): bc_calls.append(cfg); return "C"
    r = WorkerReloader(loop, aim_reloader=aim, build_tracker=bt, build_classifier=bc,
                       commanded_buffer="BUF", initial_cfg=initial)
    return r, loop, aim, bt_calls, bc_calls


def test_first_reload_rebuilds_everything_when_no_initial():
    r, loop, aim, bt, bc = _make(None)
    r.reload(AppConfig())
    assert aim.reloads == 1 and loop.tracker == "T" and loop.classifier == "C"
    assert bt == [None]                                   # aim disabled -> no gmc buffer


def test_only_changed_section_rebuilds():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    # change only tracking.track_buffer
    new = base.model_copy(update={"tracking": base.tracking.model_copy(update={"track_buffer": 45})})
    r.reload(new)
    assert loop.tracker == "T" and len(bt) == 1           # tracker rebuilt
    assert aim.reloads == 0 and len(bc) == 0              # aim + classifier untouched


def test_aim_slider_change_does_not_reset_tracker():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"aim": base.aim.model_copy(update={"kp": 0.9})})
    r.reload(new)
    assert aim.reloads == 1                                # aim rebuilt
    assert len(bt) == 0 and len(bc) == 0                  # tracker/classifier untouched


def test_enabling_aim_reattaches_gmc_buffer_to_tracker():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"aim": base.aim.model_copy(update={"enabled": True})})
    r.reload(new)
    assert aim.reloads == 1                                # aim slice changed
    assert bt == ["BUF"]                                  # enabled -> gmc buffer fed
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_live_config.py -q`
Expected: FAIL (`WorkerReloader` undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/live_config.py`:

```python
class WorkerReloader:
    """Change-gated rebuild of the CI-safe worker components on a config swap
    (spec §13). Composes the aim-controller reload with tracker/classifier
    rebuilds, rebuilding only the components whose config slice actually changed
    (frozen pydantic models compare by value) so an aim-slider tweak doesn't
    needlessly reset tracking. Detector reload (torch/TRT) is box-only and NOT
    handled here.
    """

    def __init__(self, loop, *, aim_reloader, build_tracker, build_classifier,
                 commanded_buffer=None, initial_cfg=None) -> None:
        self._loop = loop
        self._aim = aim_reloader
        self._build_tracker = build_tracker
        self._build_classifier = build_classifier
        self._buf = commanded_buffer
        self._prev = initial_cfg

    def reload(self, cfg) -> None:
        prev = self._prev
        # aim controller depends on aim + trigger + recoil + motion
        if prev is None or (cfg.aim, cfg.trigger, cfg.recoil, cfg.motion) != (
                prev.aim, prev.trigger, prev.recoil, prev.motion):
            self._aim.reload(cfg)
        # tracker depends on the tracking slice (+ the aim.enabled GMC-buffer gate)
        if prev is None or cfg.tracking != prev.tracking or \
                cfg.aim.enabled != prev.aim.enabled:
            gmc = self._buf if cfg.aim.enabled else None
            self._loop.set_tracker(self._build_tracker(cfg, gmc_buffer=gmc))
        # classifier depends on the classification slice
        if prev is None or cfg.classification != prev.classification:
            self._loop.set_classifier(self._build_classifier(cfg))
        self._prev = cfg
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_live_config.py -q`
Expected: PASS (new + existing `AimReloader` tests).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/live_config.py tests/gui/test_live_config.py
git commit -m "feat(settings): change-gated WorkerReloader (aim + tracker + classifier)"
```

---

### Task 4: App wiring — settings tabs + broadened reload

**Files:**
- Modify: `src/ragnarok/app.py`
- Test: full suite (box-only glue; seams covered by Tasks 1–3 + Phases 8B/8C).

**Interfaces:**
- Consumes: `TuningPanel`, the five field tuples (Task 1), `WorkerReloader` (Task 3), `build_tracker`/`build_classifier` (already imported), `AimReloader`, `QTabWidget`.

- [ ] **Step 1: Wire it** — in `src/ragnarok/app.py`:

Extend the imports:

```python
from ragnarok.gui.tuning_panel import TuningPanel
from ragnarok.gui.tuning_model import (
    TRACKING_FIELDS, CLASSIFICATION_FIELDS, TRIGGER_FIELDS, RECOIL_FIELDS, MOTION_FIELDS)
from ragnarok.gui.diagnostics_panel import DiagnosticsPanel
from ragnarok.gui.live_config import AimReloader, WorkerReloader
```

Replace the reloader/panel/tabs block from Phase 8C:

```python
    handle = ConfigHandle(cfg)
    reloader = AimReloader(loop, _build_aim_controller, commanded_buffer=cmd_buffer)
    panel = TuningPanel(handle, on_save=lambda c: save_config(c, _config_path()))
    panel.configChanged.connect(reloader.reload)
    diagnostics = DiagnosticsPanel(handle)
    diagnostics.configChanged.connect(reloader.reload)

    tabs = QTabWidget()
    tabs.addTab(panel, "Aim")
    tabs.addTab(diagnostics, "Diagnostics")
```

with:

```python
    handle = ConfigHandle(cfg)
    aim_reloader = AimReloader(loop, _build_aim_controller, commanded_buffer=cmd_buffer)
    reloader = WorkerReloader(
        loop, aim_reloader=aim_reloader,
        build_tracker=build_tracker, build_classifier=build_classifier,
        commanded_buffer=cmd_buffer, initial_cfg=cfg)

    def _save(c):
        save_config(c, _config_path())

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

(Leave the rest of `main()` unchanged — `MainWindow(publisher, controls=tabs)`, the overlay reading `handle.current`, `worker.start()`, etc.)

- [ ] **Step 2: Verify the app imports cleanly (offscreen)**

Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 567 baseline + new tests, no regressions.

- [ ] **Step 4: Commit**

```bash
git add src/ragnarok/app.py
git commit -m "feat(settings): tracking/friendfoe/trigger/recoil/motion tabs + broadened hot-reload"
```

---

## Self-Review

**Spec coverage (§10.3 tabs, §13 hot-reload):**
- Tracking / Friend-Foe / Trigger / Recoil / Motion settings tabs → five field tuples + reused `TuningPanel` (Tasks 1, 4). ✅
- Edits funnel through the immutable snapshot swap and hot-reload the worker → `apply_field` → `configChanged` → `WorkerReloader.reload` → `set_tracker`/`set_classifier`/aim rebuild (Tasks 2, 3, 4). ✅
- Only-what-changed reload (no needless tracker reset on aim tuning) → change-gated `WorkerReloader` (Task 3). ✅
- Detection tab / detector live-reload → **box-only, explicitly deferred** (torch/TRT model swap).
- Friend/Foe palette + eyedropper → **deferred to the wizard**; this tab exposes safe threshold/vote knobs only.

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** the five `*_FIELDS` tuples (Task 1) feed the generic `TuningPanel(handle, fields=...)` (Phase 8B, used in Task 4). `set_tracker`/`set_classifier` (Task 2) are consumed by `WorkerReloader` (Task 3) and rebind to `IdentityTracker`/`NullClassifier` defaults consistent with the loop constructor. `WorkerReloader(loop, *, aim_reloader, build_tracker, build_classifier, commanded_buffer, initial_cfg)` matches `build_tracker(cfg, *, gmc_buffer=None)` / `build_classifier(cfg)` from `wiring.py`, and composes the existing `AimReloader.reload(cfg)`.

**Honest deferrals (box-only / later):** the Detection tab + live detector reload, the Friend/Foe eyedropper/palette + colorblind presets, Trigger/Recoil/Motion applying only while `aim.enabled` (current architecture — trigger lives inside the aim controller), per-weapon Profiles, Dashboard plots, and the full Cyberpunk styling are noted, not silently dropped.
