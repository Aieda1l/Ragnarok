# Phase 8F — Input Tab (driver/transport select) (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An "Input" tab to select the mouse driver (SendInput vs Arduino) and edit the Arduino transport/port/host/baud, hot-reloading the aim controller so the choice takes effect — with the driver-selection logic CI-tested and only the real device build box-only.

**Architecture:** Add a small `InputConfig` (driver selector) and a reusable `"text"` field kind so the existing generic `TuningPanel` can edit the Arduino string fields. A pure `build_mouse_driver(cfg, *, sendinput_factory, arduino_factory)` chooses the driver from config (injected factories keep it CI-safe; the real `SendInputMouseDriver`/`ArduinoDriver` builds stay box-only in `app.py`). The `WorkerReloader` aim gate is extended to `input`/`arduino` so a driver/transport edit rebuilds the controller. The hardware "test move" button is deferred box-only.

**Tech Stack:** Python 3.11+, pydantic v2, PySide6 (`QLineEdit` for text fields, reused `TuningPanel`), existing `aim.mouse.SendInputMouseDriver` / `aim.arduino.ArduinoDriver`+`build_arduino_transport`, pytest-qt. No torch/GPU/serial/socket/SendInput in any test.

## Global Constraints

- **`InputConfig` default is `sendinput`** — existing behavior (the app always built SendInput) is preserved; the field is additive.
- **`build_mouse_driver` is selection-only and import-light.** It calls injected factories; it must not import `SendInputMouseDriver`/`ArduinoDriver`/serial/socket. The real builds live in `app.py` factories (box-only).
- **The `"text"` field kind is additive.** Existing float/int/bool/choice widgets and their tests are unchanged; `TuningPanel._read`/`refresh`/`_build_widget` gain a `QLineEdit` branch.
- **Driver/transport changes rebuild the aim controller** (only while `aim.enabled`, matching current architecture): extend the `WorkerReloader` aim gate to include `cfg.input` and `cfg.arduino`.
- **Reuse.** `TuningPanel(handle, fields=INPUT_FIELDS)`; edits funnel through `apply_field`→`ConfigHandle.swap`→`configChanged`→`_on_config_changed` (refresh all + reload) exactly like the other tabs.
- **Hardware test-move is box-only** (deferred): sending a real move to SendInput/Arduino needs the device.
- TDD, one deliverable per task, commit per task. Runner: `uv run --extra dev pytest`. Baseline: **590 passed**.

---

## File Structure

- **Modify** `src/ragnarok/config/schema.py` — add `InputConfig`; nest as `AppConfig.input`.
- **Modify** `src/ragnarok/gui/tuning_panel.py` — `"text"` → `QLineEdit` (build/read/refresh).
- **Modify** `src/ragnarok/gui/tuning_model.py` — add `INPUT_FIELDS`.
- **Modify** `src/ragnarok/wiring.py` — add `build_mouse_driver`.
- **Modify** `src/ragnarok/gui/live_config.py` — extend the `WorkerReloader` aim gate.
- **Modify** `src/ragnarok/app.py` — build the driver via `build_mouse_driver`; add the Input tab (box-only glue).
- **Create** tests: `tests/config/test_input_config.py`, `tests/gui/test_input_fields.py`, `tests/aim/test_mouse_factory.py`; extend `tests/gui/test_tuning_panel.py`, `tests/gui/test_live_config.py`.

---

### Task 1: `InputConfig` schema

**Files:**
- Modify: `src/ragnarok/config/schema.py`
- Test: `tests/config/test_input_config.py`

**Interfaces:**
- Produces: `InputConfig(mouse_driver: Literal["sendinput","arduino"] = "sendinput")`; `AppConfig.input: InputConfig`.

- [ ] **Step 1: Write the failing test** — `tests/config/test_input_config.py`:

```python
from ragnarok.config.schema import AppConfig, InputConfig


def test_input_defaults_to_sendinput():
    assert AppConfig().input.mouse_driver == "sendinput"


def test_input_accepts_arduino_and_is_frozen():
    cfg = AppConfig().model_copy(update={"input": InputConfig(mouse_driver="arduino")})
    assert cfg.input.mouse_driver == "arduino"
    import pytest
    with pytest.raises(Exception):
        cfg.input.mouse_driver = "sendinput"       # frozen
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/config/test_input_config.py -q`
Expected: FAIL (`InputConfig` undefined).

- [ ] **Step 3: Implement** — in `src/ragnarok/config/schema.py`, add the model (near `ArduinoConfig`):

```python
class InputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    mouse_driver: Literal["sendinput", "arduino"] = "sendinput"
```

and add the field to `AppConfig` (after `arduino`):

```python
    input: InputConfig = InputConfig()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/config/test_input_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/config/schema.py tests/config/test_input_config.py
git commit -m "feat(input): InputConfig mouse_driver selector (default sendinput)"
```

---

### Task 2: `TuningPanel` `"text"` field kind (`QLineEdit`)

**Files:**
- Modify: `src/ragnarok/gui/tuning_panel.py`
- Test: `tests/gui/test_tuning_panel.py`

**Interfaces:**
- Produces: a `"text"` `FieldSpec.kind` renders a `QLineEdit`; `_read` returns its text; `refresh` repaints it (signal-blocked); `_commit` on `editingFinished`.

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_tuning_panel.py`:

```python
def test_text_field_edits_and_refreshes(qtbot):
    from PySide6.QtWidgets import QLineEdit
    from ragnarok.gui.tuning_model import FieldSpec
    h = ConfigHandle(AppConfig())
    panel = TuningPanel(h, fields=(FieldSpec("arduino.port", "Port", "text"),))
    qtbot.addWidget(panel)
    w = panel.widget_for("arduino.port")
    assert isinstance(w, QLineEdit) and w.text() == ""
    w.setText("COM3")
    panel._commit("arduino.port")
    assert h.current.arduino.port == "COM3"
    # refresh from a swapped config, signal-blocked (no re-commit)
    h.swap(AppConfig().model_copy(update={"arduino": AppConfig().arduino.model_copy(update={"port": "COM7"})}))
    with qtbot.assertNotEmitted(panel.configChanged):
        panel.refresh()
    assert panel.widget_for("arduino.port").text() == "COM7"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_tuning_panel.py::test_text_field_edits_and_refreshes -q`
Expected: FAIL (text field builds a spin box / `float("")` raises).

- [ ] **Step 3: Implement** — in `src/ragnarok/gui/tuning_panel.py`:

Add `QLineEdit` to the imports:

```python
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)
```

In `_build_widget`, add a text branch before the float/int spin box:

```python
        if spec.kind == "text":
            w = QLineEdit()
            w.setText("" if value is None else str(value))
            w.editingFinished.connect(lambda p=path: self._commit(p))
            return w
```

In `_read`, handle `QLineEdit`:

```python
    def _read(self, path: str):
        w = self._widgets[path]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        if isinstance(w, QLineEdit):
            return w.text()
        return w.value()
```

In `refresh`, handle `QLineEdit` inside the blocked block:

```python
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, QComboBox):
                    w.setCurrentText(str(value))
                elif isinstance(w, QLineEdit):
                    w.setText("" if value is None else str(value))
                else:
                    self._fit_range(w, value)
                    w.setValue(float(value))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_tuning_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tuning_panel.py tests/gui/test_tuning_panel.py
git commit -m "feat(input): TuningPanel text field kind (QLineEdit)"
```

---

### Task 3: `INPUT_FIELDS` + `build_mouse_driver` selection factory

**Files:**
- Modify: `src/ragnarok/gui/tuning_model.py`
- Modify: `src/ragnarok/wiring.py`
- Test: `tests/gui/test_input_fields.py`, `tests/aim/test_mouse_factory.py`

**Interfaces:**
- Produces:
  - `INPUT_FIELDS: tuple[FieldSpec, ...]` (driver + arduino transport/port/host/udp_port/baud).
  - `wiring.build_mouse_driver(cfg, *, sendinput_factory, arduino_factory)` — returns `arduino_factory(cfg)` when `cfg.input.mouse_driver == "arduino"`, else `sendinput_factory()`.

- [ ] **Step 1: Write the failing tests** — `tests/gui/test_input_fields.py`:

```python
from ragnarok.config.schema import AppConfig
from ragnarok.gui.tuning_model import set_field, INPUT_FIELDS


def test_input_fields_wellformed_and_target_input_or_arduino():
    assert len(INPUT_FIELDS) >= 3
    for f in INPUT_FIELDS:
        assert f.path.startswith("input.") or f.path.startswith("arduino.")
        assert f.kind in {"text", "int", "choice"}


def test_input_fields_set_roundtrip_including_text():
    cfg = AppConfig()
    assert set_field(cfg, "input.mouse_driver", "arduino").input.mouse_driver == "arduino"
    assert set_field(cfg, "arduino.port", "COM5").arduino.port == "COM5"
    assert set_field(cfg, "arduino.udp_port", 9000).arduino.udp_port == 9000
    assert set_field(cfg, "arduino.transport", "udp").arduino.transport == "udp"
```

and `tests/aim/test_mouse_factory.py`:

```python
from ragnarok.config.schema import AppConfig, InputConfig
from ragnarok.wiring import build_mouse_driver


def test_build_mouse_driver_selects_sendinput_by_default():
    calls = {"send": 0, "arduino": 0}
    d = build_mouse_driver(
        AppConfig(),
        sendinput_factory=lambda: (calls.__setitem__("send", calls["send"] + 1), "SEND")[1],
        arduino_factory=lambda cfg: (calls.__setitem__("arduino", calls["arduino"] + 1), "ARD")[1])
    assert d == "SEND" and calls == {"send": 1, "arduino": 0}


def test_build_mouse_driver_selects_arduino_and_passes_cfg():
    cfg = AppConfig().model_copy(update={"input": InputConfig(mouse_driver="arduino")})
    seen = {}
    d = build_mouse_driver(
        cfg,
        sendinput_factory=lambda: "SEND",
        arduino_factory=lambda c: (seen.__setitem__("cfg", c), "ARD")[1])
    assert d == "ARD" and seen["cfg"] is cfg
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/gui/test_input_fields.py tests/aim/test_mouse_factory.py -q`
Expected: FAIL (`INPUT_FIELDS` / `build_mouse_driver` undefined).

- [ ] **Step 3: Implement**

In `src/ragnarok/gui/tuning_model.py`, append after `MOTION_FIELDS`:

```python
# Input: which mouse driver + the Arduino transport settings. The hardware
# test-move is box-only; here we only bind config.
INPUT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("input.mouse_driver", "Mouse driver", "choice",
              choices=("sendinput", "arduino")),
    FieldSpec("arduino.transport", "Arduino transport", "choice",
              choices=("serial", "udp")),
    FieldSpec("arduino.port", "Serial port (COM/tty)", "text"),
    FieldSpec("arduino.baud", "Baud", "int", 1200, 2000000, 100),
    FieldSpec("arduino.host", "UDP host", "text"),
    FieldSpec("arduino.udp_port", "UDP port", "int", 0, 65535, 1),
)
```

In `src/ragnarok/wiring.py`, add:

```python
def build_mouse_driver(cfg, *, sendinput_factory, arduino_factory):
    """Select the mouse driver from config. Selection-only + import-light: the
    real SendInput/Arduino builds live in the injected factories (box-only)."""
    if cfg.input.mouse_driver == "arduino":
        return arduino_factory(cfg)
    return sendinput_factory()
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --extra dev pytest tests/gui/test_input_fields.py tests/aim/test_mouse_factory.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tuning_model.py src/ragnarok/wiring.py tests/gui/test_input_fields.py tests/aim/test_mouse_factory.py
git commit -m "feat(input): INPUT_FIELDS + build_mouse_driver selection factory"
```

---

### Task 4: Reloader gate + app wiring

**Files:**
- Modify: `src/ragnarok/gui/live_config.py`
- Modify: `src/ragnarok/app.py`
- Test: `tests/gui/test_live_config.py`; full suite.

**Interfaces:**
- Consumes: `build_mouse_driver`, `INPUT_FIELDS`, `SendInputMouseDriver`, `ArduinoDriver`/`build_arduino_transport`.
- Produces: `WorkerReloader` aim gate also fires on `input`/`arduino` change; app builds the driver via `build_mouse_driver` and adds the Input tab.

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_live_config.py`:

```python
def test_driver_change_rebuilds_aim_controller():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"input": base.input.model_copy(update={"mouse_driver": "arduino"})})
    r.reload(new)
    assert aim.reloads == 1                                # input change -> aim rebuild
    assert len(bt) == 0 and len(bc) == 0                  # tracker/classifier untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_live_config.py::test_driver_change_rebuilds_aim_controller -q`
Expected: FAIL (`aim.reloads == 0` — the gate ignores `input`).

- [ ] **Step 3: Implement**

In `src/ragnarok/gui/live_config.py`, extend the aim gate in `WorkerReloader.reload`:

```python
        # aim controller depends on aim + trigger + recoil + motion + input driver + arduino
        if prev is None or (cfg.aim, cfg.trigger, cfg.recoil, cfg.motion,
                            cfg.input, cfg.arduino) != (
                prev.aim, prev.trigger, prev.recoil, prev.motion,
                prev.input, prev.arduino):
            self._aim.reload(cfg)
```

In `src/ragnarok/app.py`:

Add imports for the Input tab fields and driver builders:

```python
from ragnarok.gui.tuning_model import (
    TRACKING_FIELDS, CLASSIFICATION_FIELDS, TRIGGER_FIELDS, RECOIL_FIELDS,
    MOTION_FIELDS, INPUT_FIELDS)
from ragnarok.wiring import build_tracker, build_classifier, build_mouse_driver
```

(Extend the existing `from ragnarok.wiring import build_tracker, build_classifier` line.)

In `_build_aim_controller`, replace the SendInput-only build:

```python
    mouse = SendInputMouseDriver()
    mouse.connect()
```

with the selectable build:

```python
    def _sendinput():
        m = SendInputMouseDriver()
        m.connect()
        return m

    def _arduino(c):
        from ragnarok.aim.arduino import ArduinoDriver, build_arduino_transport
        d = ArduinoDriver(transport=build_arduino_transport(c))
        d.connect()
        return d

    mouse = build_mouse_driver(cfg, sendinput_factory=_sendinput, arduino_factory=_arduino)
```

(The existing `from ragnarok.aim.mouse import SendInputMouseDriver, MouseButton` import inside `_build_aim_controller` stays.)

In `main()`, add the Input tab to the settings-tab loop tuple:

```python
    for fields, title in ((TRACKING_FIELDS, "Tracking"),
                          (CLASSIFICATION_FIELDS, "Friend/Foe"),
                          (TRIGGER_FIELDS, "Trigger"),
                          (RECOIL_FIELDS, "Recoil"),
                          (MOTION_FIELDS, "Motion"),
                          (INPUT_FIELDS, "Input")):
```

- [ ] **Step 4: Run to verify the gate test + app import**

Run: `uv run --extra dev pytest tests/gui/test_live_config.py -q`
Expected: PASS.
Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Run the full suite**

Run: `uv run --extra dev pytest -q`
Expected: PASS — 590 baseline + new tests, no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/ragnarok/gui/live_config.py src/ragnarok/app.py
git commit -m "feat(input): Input tab + build_mouse_driver wiring + reloader gate for input/arduino"
```

---

## Self-Review

**Spec coverage (§10.3 Input tab):**
- Driver select (SendInput/Arduino) → `InputConfig.mouse_driver` + `INPUT_FIELDS` + `build_mouse_driver` (Tasks 1, 3, 4). ✅
- Transport / Arduino port / IP → `arduino.transport`/`port`/`host`/`udp_port`/`baud` via `INPUT_FIELDS` + the `"text"` widget kind (Tasks 2, 3). ✅
- Driver change takes effect → reloader gate on `input`/`arduino` rebuilds the controller via `build_mouse_driver` (Task 4). ✅
- "Test" (send a probe move) → **box-only, deferred** (needs the real device).

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** `InputConfig`/`AppConfig.input` (Task 1) read by `INPUT_FIELDS` + `build_mouse_driver` (Task 3) + the reloader gate (Task 4). The `"text"` kind (Task 2) is used by `arduino.port`/`host` in `INPUT_FIELDS`. `build_mouse_driver(cfg, *, sendinput_factory, arduino_factory)` matches the app factories; `build_arduino_transport(cfg)` reads `cfg.arduino` (existing signature). `set_field` handles the new `input.`/`arduino.` paths generically.

**Honest deferrals (box-only / later):** the hardware test-move button, live serial/UDP connect diagnostics, VID/PID spoofing, and per-transport validation UI are box-only; the driver rebuild (like all Trigger/Recoil/Motion edits) applies only while `aim.enabled`; a corrupt/invalid port string only fails at real connect time (box-only). Cyberpunk styling remains the final visual pass.
