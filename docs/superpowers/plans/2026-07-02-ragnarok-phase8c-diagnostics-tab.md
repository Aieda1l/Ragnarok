# Phase 8C — Diagnostics Tab (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GUI "Diagnostics" tab that drives the Phase 5A machinery — simulate a step-response for the current controller, run relay + numeric PID auto-tune against a plant model, show the metrics/seeds, and apply seeds into the running config via the Phase 8B hot-swap seam.

**Architecture:** A **Qt-free orchestration model** (`gui/diagnostics_model.py`) wraps the tested 5A pure functions: `simulate_step` (build the configured aimer → `simulate_closed_loop` → metrics), `relay_tune`/`numeric_tune_from` (seed PID gains against a synthetic `AimPlant`), formatters, and `apply_tuned` (`apply_seeds` → `ConfigHandle.swap`). A thin `DiagnosticsPanel(QWidget)` exposes plant/run controls and an Apply button, emitting `configChanged` like the tuning panel so `app.py` hot-reloads the worker. All simulation is CI-safe; the real desktop/in-game closed-loop measurement (spec §11 modes a/b/c) stays box-only behind the existing `StepResponseRunner` move/sample seams.

**Tech Stack:** Python 3.11+, existing `ragnarok.diagnostics.*` (plant/metrics/relay/numeric/apply), `ragnarok.wiring.build_aimer`, scipy (already used by `numeric_tune`), PySide6, pytest-qt. No torch/GPU/network/SendInput in any test.

## Global Constraints

- **Simulation, not live measurement, is this slice's CI surface.** `simulate_step`/`relay_tune`/`numeric_tune_from` run against the synthetic `AimPlant`; the real cursor/detector/HIL samplers (spec §11) are box-only and reuse `StepResponseRunner`'s injected `move`/`sample`.
- **Auto-tune emits SEEDS, not final values** (spec §11). The panel never auto-applies; `apply_tuned` runs only on an explicit Apply click, and funnels through `apply_seeds` → `ConfigHandle.swap` → `configChanged` (reusing the Phase 8B reload path).
- **`diagnostics_model.py` imports ZERO Qt / SendInput / torch.** The widget is the only Qt file. Widget tests use `qtbot`.
- **Reuse, don't re-implement.** Metrics/relay/numeric/apply come from `ragnarok.diagnostics.*`; the aimer under test comes from `ragnarok.wiring.build_aimer(cfg)`. No duplicated control math.
- **Additive only:** no change to existing diagnostics/config/worker signatures; the app gains a tab host without breaking `MainWindow(controls=...)`.
- TDD, one deliverable per task, commit per task. Runner: `uv run --extra dev pytest`. Baseline: **551 passed**.

---

## File Structure

- **Create** `src/ragnarok/gui/diagnostics_model.py` — `PlantParams`, `simulate_step`, `relay_tune`, `numeric_tune_from`, `format_result`, `format_seeds`, `apply_tuned`. Qt-free.
- **Create** `src/ragnarok/gui/diagnostics_panel.py` — `DiagnosticsPanel(QWidget)`.
- **Modify** `src/ragnarok/app.py` — host the Aim + Diagnostics panels in a `QTabWidget`; wire `configChanged` → reload (box-only).
- **Create** tests: `tests/gui/test_diagnostics_model.py`, `tests/gui/test_diagnostics_panel.py`.

---

### Task 1: `diagnostics_model.py` — plant + step simulation + result formatting

**Files:**
- Create: `src/ragnarok/gui/diagnostics_model.py`
- Test: `tests/gui/test_diagnostics_model.py`

**Interfaces:**
- Consumes: `diagnostics.plant.AimPlant/simulate_closed_loop`, `diagnostics.results.StepResponseResult`, `diagnostics.metrics`, `wiring.build_aimer`, `config.schema.AppConfig`.
- Produces:
  - `PlantParams(gain=1.0, lag_tau_s=0.02, dead_time_s=0.0, dt_s=1/240)` with `.make() -> AimPlant`.
  - `simulate_step(cfg, params, *, setpoint=200.0, n_steps=240) -> StepResponseResult`.
  - `format_result(result) -> dict[str, str]` (keys `Rise`, `Overshoot`, `Settling`, `Dead time`; `None` → `"—"`).

- [ ] **Step 1: Write the failing test** — `tests/gui/test_diagnostics_model.py`:

```python
import numpy as np
from ragnarok.config.schema import AppConfig
from ragnarok.diagnostics.results import StepResponseResult
from ragnarok.gui.diagnostics_model import PlantParams, simulate_step, format_result


def test_plant_params_make_builds_plant():
    p = PlantParams(gain=1.0, lag_tau_s=0.01, dead_time_s=0.0, dt_s=1 / 240)
    plant = p.make()
    assert plant.position == 0.0
    plant.step(1.0)
    assert plant.position != 0.0                       # integrator moved


def test_simulate_step_returns_result_with_arrays_and_metrics():
    cfg = AppConfig()                                   # feedback P controller, kp 0.35
    res = simulate_step(cfg, PlantParams(), setpoint=200.0, n_steps=300)
    assert isinstance(res, StepResponseResult)
    assert res.t_s.shape == (300,) and res.y.shape == (300,)
    assert res.y_final == 200.0 and res.y0 == 0.0
    assert res.overshoot_pct >= 0.0
    assert res.y[-1] > 100.0                            # P controller drives toward setpoint


def test_format_result_handles_none_and_units():
    res = StepResponseResult(rise_s=0.042, overshoot_pct=3.2, settling_s=None,
                             dead_time_s=0.0, t_s=np.zeros(1), y=np.zeros(1),
                             y0=0.0, y_final=1.0)
    out = format_result(res)
    assert out["Settling"] == "—"                       # None -> em dash
    assert "42.0" in out["Rise"] and "ms" in out["Rise"]
    assert "3.2" in out["Overshoot"] and "%" in out["Overshoot"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_diagnostics_model.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/diagnostics_model.py`:

```python
"""Diagnostics-tab orchestration over the Phase 5A pure machinery (spec §11).

ZERO Qt / SendInput: builds the configured aimer, runs it against the synthetic
``AimPlant`` (``simulate_closed_loop``), and computes step-response metrics /
relay + numeric PID seeds. The real desktop/in-game/HIL samplers are box-only
and reuse ``diagnostics.runner.StepResponseRunner``. Seeds are applied only on an
explicit call (``apply_tuned``) via ``diagnostics.apply.apply_seeds`` -> swap.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragnarok.diagnostics import metrics
from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop
from ragnarok.diagnostics.results import StepResponseResult


@dataclass(frozen=True)
class PlantParams:
    gain: float = 1.0
    lag_tau_s: float = 0.02
    dead_time_s: float = 0.0
    dt_s: float = 1.0 / 240.0

    def make(self) -> AimPlant:
        return AimPlant(gain=self.gain, lag_tau_s=self.lag_tau_s,
                        dead_time_s=self.dead_time_s, dt_s=self.dt_s)


def _fmt_ms(v):
    return "—" if v is None else f"{v * 1000.0:.1f} ms"


def simulate_step(cfg, params: PlantParams, *, setpoint: float = 200.0,
                  n_steps: int = 240) -> StepResponseResult:
    """Closed-loop step response of the CURRENTLY-configured aimer vs a plant."""
    from ragnarok.wiring import build_aimer
    aimer = build_aimer(cfg)
    plant = params.make()
    t, m, _u = simulate_closed_loop(
        lambda e, dt: aimer.step((0.0, 0.0), (e, 0.0), dt)[0],
        plant, setpoint=setpoint, n_steps=n_steps, dt_s=params.dt_s,
    )
    return StepResponseResult(
        rise_s=metrics.rise_time(t, m, y0=0.0, y_final=setpoint),
        overshoot_pct=metrics.overshoot_pct(m, y0=0.0, y_final=setpoint),
        settling_s=metrics.settling_time(t, m, y0=0.0, y_final=setpoint),
        dead_time_s=metrics.dead_time(t, m, y0=0.0, y_final=setpoint),
        t_s=t, y=m, y0=0.0, y_final=setpoint,
    )


def format_result(result: StepResponseResult) -> dict[str, str]:
    return {
        "Rise": _fmt_ms(result.rise_s),
        "Overshoot": f"{result.overshoot_pct:.1f} %",
        "Settling": _fmt_ms(result.settling_s),
        "Dead time": _fmt_ms(result.dead_time_s),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_diagnostics_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/diagnostics_model.py tests/gui/test_diagnostics_model.py
git commit -m "feat(diag): plant params + step-response simulation + result formatting"
```

---

### Task 2: `diagnostics_model.py` — relay + numeric tune, seed formatting, apply

**Files:**
- Modify: `src/ragnarok/gui/diagnostics_model.py`
- Test: `tests/gui/test_diagnostics_model.py`

**Interfaces:**
- Consumes: `diagnostics.relay_experiment.run_relay_tune/RelayTuneResult`, `diagnostics.numeric_tune.numeric_tune/PidSeeds`, `diagnostics.apply.apply_seeds`, `config.store.ConfigHandle`.
- Produces:
  - `relay_tune(params, *, d=50.0, n_steps=3000, rule="low_overshoot") -> RelayTuneResult`.
  - `numeric_tune_from(cfg, params, *, setpoint=200.0, n_steps=240) -> PidSeeds`.
  - `format_seeds(seeds) -> dict[str, str]` (keys `Kp`, `Ki`, `Kd`).
  - `apply_tuned(handle, seeds, *, controller_mode="pid") -> AppConfig` (apply_seeds + swap).

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_diagnostics_model.py`:

```python
from ragnarok.config.store import ConfigHandle
from ragnarok.diagnostics.numeric_tune import PidSeeds
from ragnarok.diagnostics.relay_experiment import RelayTuneResult
from ragnarok.gui.diagnostics_model import (
    relay_tune, numeric_tune_from, format_seeds, apply_tuned)


def test_relay_tune_finds_a_limit_cycle():
    # integrator + lag + dead-time oscillates under relay feedback -> Ku/Tu > 0
    res = relay_tune(PlantParams(lag_tau_s=0.02, dead_time_s=0.01),
                     d=50.0, n_steps=4000)
    assert isinstance(res, RelayTuneResult)
    assert res.ku > 0.0 and res.tu > 0.0
    assert res.kp >= 0.0 and res.ki >= 0.0 and res.kd >= 0.0


def test_numeric_tune_from_returns_nonneg_seeds():
    cfg = AppConfig()
    seeds = numeric_tune_from(cfg, PlantParams(lag_tau_s=0.02), setpoint=100.0, n_steps=120)
    assert isinstance(seeds, PidSeeds)
    assert seeds.kp >= 0.0 and seeds.ki >= 0.0 and seeds.kd >= 0.0


def test_format_seeds_strings():
    out = format_seeds(PidSeeds(kp=0.6, ki=0.12, kd=0.03))
    assert out["Kp"].startswith("0.6")
    assert set(out) == {"Kp", "Ki", "Kd"}


def test_apply_tuned_swaps_handle_with_pid_mode():
    h = ConfigHandle(AppConfig())
    new = apply_tuned(h, PidSeeds(kp=0.5, ki=0.1, kd=0.02), controller_mode="pid")
    assert h.current is new
    assert new.aim.kp == 0.5 and new.aim.ki == 0.1 and new.aim.kd == 0.02
    assert new.aim.controller_mode == "pid"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_diagnostics_model.py -q`
Expected: FAIL (names undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/diagnostics_model.py`:

```python
def relay_tune(params: PlantParams, *, d: float = 50.0, n_steps: int = 3000,
               rule: str = "low_overshoot"):
    """Relay-feedback (Åström-Hägglund) auto-tune against the plant model."""
    from ragnarok.diagnostics.relay_experiment import run_relay_tune
    return run_relay_tune(params.make(), d=d, n_steps=n_steps, dt_s=params.dt_s,
                          rule=rule)


def numeric_tune_from(cfg, params: PlantParams, *, setpoint: float = 200.0,
                      n_steps: int = 240):
    """Nelder-Mead ITAE tune seeded from the current config's PID gains."""
    from ragnarok.diagnostics.numeric_tune import numeric_tune, PidSeeds
    seed = PidSeeds(kp=cfg.aim.kp, ki=cfg.aim.ki, kd=cfg.aim.kd)
    return numeric_tune(params.make, seed=seed, setpoint=setpoint,
                        n_steps=n_steps, dt_s=params.dt_s,
                        max_step_px=cfg.aim.max_step_px)


def format_seeds(seeds) -> dict[str, str]:
    return {"Kp": f"{seeds.kp:.4g}", "Ki": f"{seeds.ki:.4g}", "Kd": f"{seeds.kd:.4g}"}


def apply_tuned(handle, seeds, *, controller_mode: str = "pid"):
    """Apply auto-tune seeds into a NEW frozen AppConfig and swap the handle."""
    from ragnarok.diagnostics.apply import apply_seeds
    new_cfg = apply_seeds(handle.current, seeds, controller_mode=controller_mode)
    handle.swap(new_cfg)
    return new_cfg
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_diagnostics_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/diagnostics_model.py tests/gui/test_diagnostics_model.py
git commit -m "feat(diag): relay + numeric auto-tune wrappers, seed format, apply_tuned"
```

---

### Task 3: `DiagnosticsPanel` widget

**Files:**
- Create: `src/ragnarok/gui/diagnostics_panel.py`
- Test: `tests/gui/test_diagnostics_panel.py`

**Interfaces:**
- Consumes: `PlantParams`, `simulate_step`, `relay_tune`, `numeric_tune_from`, `format_result`, `format_seeds`, `apply_tuned` (Tasks 1–2); `ConfigHandle`.
- Produces: `DiagnosticsPanel(handle, *, controller_mode="pid", relay_steps=3000)` — a `QWidget` with:
  - `configChanged` Signal (emits new `AppConfig` on Apply).
  - `last_seeds` attribute (`PidSeeds | None`).
  - `_plant_params() -> PlantParams`, `_run_step()`, `_run_relay()`, `_run_numeric()`, `_apply()`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_diagnostics_panel.py`:

```python
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.diagnostics_panel import DiagnosticsPanel


def test_run_step_populates_metric_labels(qtbot):
    panel = DiagnosticsPanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel._run_step()
    text = panel.metrics_label.text()
    assert "Rise" in text and "Overshoot" in text          # formatted metrics shown


def test_run_relay_sets_seeds(qtbot):
    panel = DiagnosticsPanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel.widget_for("dead_time_s").setValue(10.0)          # ms -> ensures a limit cycle
    panel.widget_for("lag_tau_s").setValue(20.0)
    panel._run_relay()
    assert panel.last_seeds is not None
    assert "Kp" in panel.seeds_label.text()


def test_run_numeric_sets_seeds(qtbot):
    panel = DiagnosticsPanel(ConfigHandle(AppConfig()))
    qtbot.addWidget(panel)
    panel.widget_for("n_steps").setValue(120)
    panel._run_numeric()
    assert panel.last_seeds is not None


def test_apply_swaps_handle_and_emits(qtbot):
    h = ConfigHandle(AppConfig())
    panel = DiagnosticsPanel(h)
    qtbot.addWidget(panel)
    panel.widget_for("dead_time_s").setValue(10.0)
    panel.widget_for("lag_tau_s").setValue(20.0)
    panel._run_relay()
    with qtbot.waitSignal(panel.configChanged, timeout=2000) as blocker:
        panel._apply()
    assert h.current.aim.controller_mode == "pid"
    assert h.current.aim.kp == panel.last_seeds.kp
    assert blocker.args[0] is h.current


def test_apply_without_seeds_is_noop(qtbot):
    h = ConfigHandle(AppConfig())
    panel = DiagnosticsPanel(h)
    qtbot.addWidget(panel)
    before = h.current
    panel._apply()                                          # no run yet -> last_seeds None
    assert h.current is before
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_diagnostics_panel.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/diagnostics_panel.py`:

```python
"""Diagnostics tab: simulate step-response + relay/numeric PID auto-tune and
apply seeds (spec §10.3 Diagnostics, §11). Thin Qt shell over diagnostics_model;
all math is the tested Phase 5A machinery. Live desktop/in-game measurement and
a response plot are box-only follow-ups; auto-tune here runs on the synthetic
plant model. Long tuning runs execute synchronously (fast on the small sim);
moving them to a worker thread is a box-only refinement.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from ragnarok.gui.diagnostics_model import (
    PlantParams, apply_tuned, format_result, format_seeds, numeric_tune_from,
    relay_tune, simulate_step,
)


class DiagnosticsPanel(QWidget):
    configChanged = Signal(object)

    def __init__(self, handle, *, controller_mode: str = "pid",
                 relay_steps: int = 3000) -> None:
        super().__init__()
        self._handle = handle
        self._mode = controller_mode
        self._relay_steps = relay_steps
        self.last_seeds = None
        self._spins: dict[str, QDoubleSpinBox] = {}

        root = QVBoxLayout(self)
        form = QFormLayout()
        # (key, label, min, max, step, default) — plant model + run params
        for key, label, lo, hi, step, default in (
            ("gain", "Plant gain", 0.01, 10.0, 0.1, 1.0),
            ("lag_tau_s", "Actuator lag (ms)", 0.0, 200.0, 1.0, 20.0),
            ("dead_time_s", "Dead time (ms)", 0.0, 200.0, 1.0, 0.0),
            ("setpoint", "Step (px)", 1.0, 2000.0, 10.0, 200.0),
            ("n_steps", "Sim steps", 20.0, 4000.0, 20.0, 240.0),
        ):
            w = QDoubleSpinBox()
            w.setDecimals(0 if key in ("n_steps",) else 3)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(default)
            self._spins[key] = w
            form.addRow(label, w)
        root.addLayout(form)

        self.metrics_label = QLabel("Rise —  Overshoot —  Settling —  Dead time —")
        self.seeds_label = QLabel("Kp —  Ki —  Kd —")
        root.addWidget(self.metrics_label)
        root.addWidget(self.seeds_label)

        b_step = QPushButton("Run step-response (sim)")
        b_relay = QPushButton("Relay auto-tune")
        b_num = QPushButton("Numeric auto-tune")
        b_apply = QPushButton("Apply seeds")
        b_step.clicked.connect(self._run_step)
        b_relay.clicked.connect(self._run_relay)
        b_num.clicked.connect(self._run_numeric)
        b_apply.clicked.connect(self._apply)
        for b in (b_step, b_relay, b_num, b_apply):
            root.addWidget(b)

    # -- helpers ---------------------------------------------------------
    def widget_for(self, key: str) -> QDoubleSpinBox:
        return self._spins[key]

    def _plant_params(self) -> PlantParams:
        return PlantParams(
            gain=self._spins["gain"].value(),
            lag_tau_s=self._spins["lag_tau_s"].value() / 1000.0,   # ms -> s
            dead_time_s=self._spins["dead_time_s"].value() / 1000.0,
        )

    def _setpoint(self) -> float:
        return float(self._spins["setpoint"].value())

    def _n_steps(self) -> int:
        return int(self._spins["n_steps"].value())

    def _show_metrics(self, result) -> None:
        m = format_result(result)
        self.metrics_label.setText(
            f"Rise {m['Rise']}  Overshoot {m['Overshoot']}  "
            f"Settling {m['Settling']}  Dead time {m['Dead time']}")

    def _show_seeds(self, seeds) -> None:
        s = format_seeds(seeds)
        self.seeds_label.setText(f"Kp {s['Kp']}  Ki {s['Ki']}  Kd {s['Kd']}")

    # -- actions ---------------------------------------------------------
    def _run_step(self) -> None:
        res = simulate_step(self._handle.current, self._plant_params(),
                            setpoint=self._setpoint(), n_steps=self._n_steps())
        self._show_metrics(res)

    def _run_relay(self) -> None:
        res = relay_tune(self._plant_params(), n_steps=self._relay_steps)
        from ragnarok.diagnostics.numeric_tune import PidSeeds
        self.last_seeds = PidSeeds(kp=res.kp, ki=res.ki, kd=res.kd)
        self._show_seeds(self.last_seeds)

    def _run_numeric(self) -> None:
        self.last_seeds = numeric_tune_from(
            self._handle.current, self._plant_params(),
            setpoint=self._setpoint(), n_steps=self._n_steps())
        self._show_seeds(self.last_seeds)

    def _apply(self) -> None:
        if self.last_seeds is None:
            return
        new_cfg = apply_tuned(self._handle, self.last_seeds, controller_mode=self._mode)
        self.configChanged.emit(new_cfg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_diagnostics_panel.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/diagnostics_panel.py tests/gui/test_diagnostics_panel.py
git commit -m "feat(diag): DiagnosticsPanel widget (simulate/relay/numeric/apply)"
```

---

### Task 4: App wiring — tabbed control host

**Files:**
- Modify: `src/ragnarok/app.py`
- Test: full suite (box-only glue; seams covered by Tasks 1–3 + Phase 8B).

**Interfaces:**
- Consumes: `TuningPanel`, `DiagnosticsPanel`, `AimReloader`, `QTabWidget`, `MainWindow(controls=...)`.

- [ ] **Step 1: Wire it** — in `src/ragnarok/app.py`:

Add imports:

```python
from PySide6.QtWidgets import QApplication, QTabWidget
from ragnarok.gui.diagnostics_panel import DiagnosticsPanel
```

(Extend the existing `from PySide6.QtWidgets import QApplication` line to include `QTabWidget`.)

Replace the panel/window block from Phase 8B:

```python
    panel = TuningPanel(handle, on_save=lambda c: save_config(c, _config_path()))
    panel.configChanged.connect(reloader.reload)

    worker = WorkerThread(loop)
    window = MainWindow(publisher, controls=panel)
```

with a tabbed host:

```python
    panel = TuningPanel(handle, on_save=lambda c: save_config(c, _config_path()))
    panel.configChanged.connect(reloader.reload)
    diagnostics = DiagnosticsPanel(handle)
    diagnostics.configChanged.connect(reloader.reload)

    tabs = QTabWidget()
    tabs.addTab(panel, "Aim")
    tabs.addTab(diagnostics, "Diagnostics")

    worker = WorkerThread(loop)
    window = MainWindow(publisher, controls=tabs)
```

(Leave the rest of `main()` unchanged — the overlay reading `handle.current`, `worker.start()`, etc.)

- [ ] **Step 2: Verify the app imports cleanly (offscreen)**

Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 551 baseline + new tests, no regressions.

- [ ] **Step 4: Commit**

```bash
git add src/ragnarok/app.py
git commit -m "feat(diag): host Aim + Diagnostics panels in a tabbed control area"
```

---

## Self-Review

**Spec coverage (§10.3 Diagnostics tab, §11 diagnostics/auto-tune):**
- Step-response with rise/overshoot/settling/dead-time → `simulate_step` + `format_result` + `_run_step` (Tasks 1, 3). ✅ (against the plant model; real cursor/detector/HIL modes box-only.)
- Relay-feedback (Åström-Hägglund) auto-tune → `relay_tune` → `run_relay_tune` (Task 2). ✅
- Numeric (Nelder-Mead ITAE) auto-tune → `numeric_tune_from` → `numeric_tune` (Task 2). ✅
- Seeds, not final; explicit apply → `apply_tuned` → `apply_seeds` → swap → `configChanged` → reload (Tasks 2, 3, 4). ✅
- Tab lives beside the Aim tab → `QTabWidget` host (Task 4). ✅

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** `PlantParams`/`simulate_step`/`format_result` (Task 1) and `relay_tune`/`numeric_tune_from`/`format_seeds`/`apply_tuned` (Task 2) consumed by `DiagnosticsPanel` (Task 3) and app (Task 4). `apply_tuned` mirrors the Phase 8B `apply_field` swap pattern and reuses `diagnostics.apply.apply_seeds` (kp/ki/kd + controller_mode). `configChanged.connect(reloader.reload)` matches the Phase 8B `AimReloader.reload(cfg)` signature. `StepResponseResult`/`RelayTuneResult`/`PidSeeds` are the existing 5A types.

**Honest deferrals (box-only / later):** the live desktop/in-game/HIL step-response measurement modes (reuse `StepResponseRunner`), a `pyqtgraph` response-curve plot, running long tunes off the GUI thread, and cross-panel value refresh (after Apply, the Aim tab's spinboxes still show pre-tune gains until reopened — the config + worker are correctly updated) are noted, not silently dropped. Auto-tune runs against a plant *model*; its seeds are only as good as the plant params the user enters (from a real characterization, box-only) — consistent with spec §11 "seeds, not final values."
