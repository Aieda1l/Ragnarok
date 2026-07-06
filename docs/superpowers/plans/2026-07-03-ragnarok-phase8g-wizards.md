# Phase 8G — Calibration Wizards (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Wizards" tab with a sensitivity / GMC calibration wizard — from a known calibration turn, compute `deg_per_count` (signed, for feed-forward GMC) and `sensitivity` (magnitude, for the px↔count conversion) and apply them through the immutable-snapshot swap. Reuse the tested Phase 5B solvers; keep the analysis CI-tested and only the live data *collection* box-only.

**Architecture:** A Qt-free `calibration_model` wraps the 5B pure solvers (`solve_deg_per_count`, `estimate_tau_render`) into `apply_*` helpers that build a re-validated `AppConfig` and swap the handle. A thin `CalibrationPanel(QWidget)` runs the manual-turn sensitivity wizard (numeric inputs the user reads off a known turn) and emits `configChanged` like the other panels. The τ_render *solver* is included in the model (CI-tested) but its optical-flow *collection* is box-only, so the panel exposes only the manual-turn wizard this slice (the Tracking tab already has a `tau_render_s` manual field).

**Tech Stack:** Python 3.11+, existing `tracking.calibration` (`solve_deg_per_count`, `estimate_tau_render`), pydantic v2 (validated sub-model construction), PySide6 (reused widget patterns), pytest-qt. No torch/GPU/capture in any test.

## Global Constraints

- **Reuse the 5B solvers** — `solve_deg_per_count`/`estimate_tau_render` are not re-implemented. `calibration_model` only orchestrates + applies.
- **Applied config must re-validate.** `sensitivity` is schema `gt=0`; build the edited sub-models through their class (like `set_field`) so a zero/negative result raises `ValidationError` rather than silently landing an invalid config via `model_copy(update=)`.
- **`deg_per_count` sign is preserved** (GMC needs direction); `sensitivity` takes its magnitude (`abs`).
- **`calibration_model` imports ZERO Qt.** The widget is the only Qt file. Widget tests use `qtbot`.
- **Live collection is box-only.** The manual-turn wizard takes numeric inputs (the user reads them off a real turn); the τ_render optical-flow collection is deferred box-only.
- **Additive:** funnel through `ConfigHandle.swap` + `configChanged` + the app's `_on_config_changed` (refresh all + guarded reload) exactly like the other tabs.
- TDD, one deliverable per task, commit per task. Runner: `uv run --extra dev pytest`. Baseline: **599 passed**.

---

## File Structure

- **Create** `src/ragnarok/gui/calibration_model.py` — `apply_sensitivity`, `apply_tau_render`.
- **Create** `src/ragnarok/gui/calibration_panel.py` — `CalibrationPanel(QWidget)`.
- **Modify** `src/ragnarok/app.py` — add the Wizards tab (box-only glue).
- **Create** tests: `tests/gui/test_calibration_model.py`, `tests/gui/test_calibration_panel.py`.

---

### Task 1: `calibration_model` — apply solvers to config

**Files:**
- Create: `src/ragnarok/gui/calibration_model.py`
- Test: `tests/gui/test_calibration_model.py`

**Interfaces:**
- Consumes: `tracking.calibration.solve_deg_per_count/estimate_tau_render`, `config.store.ConfigHandle`.
- Produces:
  - `apply_sensitivity(handle, *, total_counts, measured_deg) -> AppConfig` — sets `tracking.deg_per_count` (signed) + `aim.sensitivity` (magnitude); swaps.
  - `apply_tau_render(handle, *, commanded, measured, dt_s, max_lag_s=0.1) -> AppConfig` — sets `tracking.tau_render_s`; swaps.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_calibration_model.py`:

```python
import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.calibration_model import apply_sensitivity, apply_tau_render


def test_apply_sensitivity_sets_signed_deg_per_count_and_magnitude():
    h = ConfigHandle(AppConfig())
    new = apply_sensitivity(h, total_counts=1000.0, measured_deg=22.0)
    assert new.tracking.deg_per_count == pytest.approx(0.022)
    assert new.aim.sensitivity == pytest.approx(0.022)
    assert h.current is new


def test_apply_sensitivity_preserves_sign_but_sensitivity_is_magnitude():
    h = ConfigHandle(AppConfig())
    new = apply_sensitivity(h, total_counts=1000.0, measured_deg=-22.0)   # inverted turn
    assert new.tracking.deg_per_count == pytest.approx(-0.022)            # sign kept for GMC
    assert new.aim.sensitivity == pytest.approx(0.022)                    # magnitude for px<->count


def test_apply_sensitivity_zero_measure_raises_not_silent():
    h = ConfigHandle(AppConfig())
    with pytest.raises(ValidationError):                                  # sensitivity gt=0
        apply_sensitivity(h, total_counts=1000.0, measured_deg=0.0)


def test_apply_tau_render_sets_tracking_tau():
    h = ConfigHandle(AppConfig())
    # measured trails commanded by 2 samples at dt=1ms -> tau ~ 0.002 s
    commanded = [0, 0, 1, 0, 0, 0, 0, 0]
    measured = [0, 0, 0, 0, 1, 0, 0, 0]
    new = apply_tau_render(h, commanded=commanded, measured=measured, dt_s=0.001)
    assert new.tracking.tau_render_s == pytest.approx(0.002, abs=1e-9)
    assert h.current is new
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_calibration_model.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/calibration_model.py`:

```python
"""Calibration-wizard orchestration over the Phase 5B pure solvers (spec §11).

ZERO Qt: turns a known calibration turn (or a commanded/measured trace) into a
re-validated AppConfig via ConfigHandle.swap. Live data collection (the turn,
the optical-flow trace) is box-only; these helpers are the pure analysis+apply.
"""
from __future__ import annotations

from ragnarok.tracking.calibration import estimate_tau_render, solve_deg_per_count


def _swap(handle, updates):
    """Build a re-validated AppConfig with the given {section: {field: value}}
    updates and swap it in. Constructing each sub-model through its class
    re-validates (model_copy(update=) would not), so an invalid result raises."""
    cfg = handle.current
    section_updates = {}
    for section, fields in updates.items():
        sub = getattr(cfg, section)
        section_updates[section] = sub.__class__(**{**sub.model_dump(), **fields})
    new_cfg = cfg.model_copy(update=section_updates)
    handle.swap(new_cfg)
    return new_cfg


def apply_sensitivity(handle, *, total_counts: float, measured_deg: float):
    """From a known calibration turn: deg_per_count = measured_deg / total_counts.

    Sets tracking.deg_per_count (SIGNED, for the GMC back-projection) and
    aim.sensitivity (magnitude, for the px<->count conversion). Raises if the
    result is invalid (e.g. a zero measured turn -> sensitivity gt=0 violated).
    """
    dpc = solve_deg_per_count(total_counts, measured_deg)
    return _swap(handle, {
        "tracking": {"deg_per_count": dpc},
        "aim": {"sensitivity": abs(dpc)},
    })


def apply_tau_render(handle, *, commanded, measured, dt_s: float, max_lag_s: float = 0.1):
    """Set tracking.tau_render_s from a commanded/measured motion trace."""
    tau = estimate_tau_render(commanded, measured, dt_s, max_lag_s=max_lag_s)
    return _swap(handle, {"tracking": {"tau_render_s": tau}})
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_calibration_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/calibration_model.py tests/gui/test_calibration_model.py
git commit -m "feat(wizards): calibration_model apply_sensitivity + apply_tau_render"
```

---

### Task 2: `CalibrationPanel` widget

**Files:**
- Create: `src/ragnarok/gui/calibration_panel.py`
- Test: `tests/gui/test_calibration_panel.py`

**Interfaces:**
- Consumes: `apply_sensitivity` (Task 1), `ConfigHandle`.
- Produces: `CalibrationPanel(handle)` — a `QWidget` with `configChanged` Signal, `widget_for(key)` (`counts`/`degrees` spin boxes), `result_label`, and `_solve()`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_calibration_panel.py`:

```python
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.calibration_panel import CalibrationPanel


def test_solve_applies_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = CalibrationPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("counts").setValue(1000.0)
    panel.widget_for("degrees").setValue(22.0)
    with qtbot.waitSignal(panel.configChanged, timeout=1000) as blocker:
        panel._solve()
    assert abs(h.current.tracking.deg_per_count - 0.022) < 1e-9
    assert abs(h.current.aim.sensitivity - 0.022) < 1e-9
    assert blocker.args[0] is h.current
    assert "0.022" in panel.result_label.text()


def test_solve_zero_counts_is_safe_noop(qtbot):
    # solve_deg_per_count raises on zero counts; the panel must not crash / emit
    h = ConfigHandle(AppConfig())
    panel = CalibrationPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("counts").setValue(0.0)
    panel.widget_for("degrees").setValue(22.0)
    before = h.current
    with qtbot.assertNotEmitted(panel.configChanged):
        panel._solve()
    assert h.current is before                       # nothing applied
    assert "invalid" in panel.result_label.text().lower() or "—" in panel.result_label.text()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_calibration_panel.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/calibration_panel.py`:

```python
"""Calibration Wizards tab (spec §10.3, §11).

Sensitivity / GMC wizard: enter a known calibration turn (mouse counts moved +
degrees the view rotated) and apply deg_per_count + sensitivity. The τ_render
auto-collection (optical flow) is box-only; the Tracking tab has a manual
tau_render_s field meanwhile. Full Cyberpunk styling is a later box-only pass.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ragnarok.gui.calibration_model import apply_sensitivity


class CalibrationPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, handle) -> None:
        super().__init__()
        self._handle = handle
        self._spins: dict[str, QDoubleSpinBox] = {}

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "Sensitivity / GMC calibration: do a known in-game turn against a\n"
            "static reference, then enter the mouse counts moved and the degrees\n"
            "the view rotated (negative if inverted)."))
        form = QFormLayout()
        for key, label, lo, hi, step, default in (
            ("counts", "Mouse counts moved", -1e7, 1e7, 10.0, 1000.0),
            ("degrees", "Degrees rotated (signed)", -3600.0, 3600.0, 1.0, 360.0),
        ):
            w = QDoubleSpinBox()
            w.setDecimals(2)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(default)
            self._spins[key] = w
            form.addRow(label, w)
        root.addLayout(form)

        self.result_label = QLabel("deg/count —")
        root.addWidget(self.result_label)

        solve = QPushButton("Solve & apply")
        solve.clicked.connect(self._solve)
        root.addWidget(solve)

    def widget_for(self, key: str) -> QDoubleSpinBox:
        return self._spins[key]

    def _solve(self) -> None:
        counts = self._spins["counts"].value()
        degrees = self._spins["degrees"].value()
        try:
            new_cfg = apply_sensitivity(self._handle, total_counts=counts,
                                        measured_deg=degrees)
        except Exception as exc:  # noqa: BLE001 — bad input (zero counts / zero turn)
            self.result_label.setText(f"invalid calibration: {exc}")
            return
        self.result_label.setText(
            f"deg/count {new_cfg.tracking.deg_per_count:.4g}  "
            f"(sensitivity {new_cfg.aim.sensitivity:.4g})")
        self.configChanged.emit(new_cfg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_calibration_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/calibration_panel.py tests/gui/test_calibration_panel.py
git commit -m "feat(wizards): CalibrationPanel sensitivity/GMC wizard"
```

---

### Task 3: App wiring — Wizards tab

**Files:**
- Modify: `src/ragnarok/app.py`
- Test: full suite (box-only glue; seams covered by Tasks 1–2).

**Interfaces:**
- Consumes: `CalibrationPanel`, `_on_config_changed`.

- [ ] **Step 1: Wire it** — in `src/ragnarok/app.py`:

Add the import:

```python
from ragnarok.gui.calibration_panel import CalibrationPanel
```

After the Profiles tab is added, add the Wizards tab:

```python
    wizards = CalibrationPanel(handle)
    wizards.configChanged.connect(_on_config_changed)
    tabs.addTab(wizards, "Wizards")
```

- [ ] **Step 2: Verify the app imports cleanly (offscreen)**

Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 599 baseline + new tests, no regressions.

- [ ] **Step 4: Commit**

```bash
git add src/ragnarok/app.py
git commit -m "feat(wizards): Wizards tab (sensitivity/GMC calibration)"
```

---

## Self-Review

**Spec coverage (§10.3 Wizards, §11 calibration):**
- Sensitivity → pixels/deg calibration → `apply_sensitivity` (sets `sensitivity` + signed `deg_per_count`) + `CalibrationPanel` (Tasks 1, 2). ✅
- τ_render calibration → `apply_tau_render` solver applied to `tracking.tau_render_s` (Task 1); **live optical-flow collection deferred box-only** (Tracking tab has the manual field meanwhile).
- Applied config re-validates → `_swap` builds sub-models through their class (Task 1). ✅
- Funnels through the snapshot swap → `configChanged` → guarded reload (Task 3). ✅
- Eyedropper / recoil-on-wall / FOV / first-run wizards → **deferred** (eyedropper + FOV need live pixel/geometry capture; recoil-on-wall needs firing capture) — noted, not silently dropped.

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** `apply_sensitivity`/`apply_tau_render` (Task 1) consumed by `CalibrationPanel._solve` (Task 2) and app (Task 3). `_swap` mirrors the `set_field` validated-reconstruction pattern. `configChanged.emit(cfg)` matches the app's `_on_config_changed(new_cfg)` (guarded reload from 8F). Reuses `solve_deg_per_count`/`estimate_tau_render` with their existing signatures.

**Honest deferrals (box-only / later):** the τ_render optical-flow auto-collection, the outline-color eyedropper, the recoil-on-wall learner, the FOV calibration, and the first-run setup wizard all need live capture/cursor/firing and are box-only; only the manual-turn sensitivity/GMC wizard (numeric inputs) is in-GUI this slice. Cyberpunk styling remains the final visual pass. This is the last GUI slice with a substantial CI-testable core — the remaining §10.3 surfaces (eyedropper, Dashboard plots, Training, Detection reload, aesthetic pass) are predominantly box-only/visual.
