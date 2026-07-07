# Latency-in-GUI + GUI Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the wall-based latency measurement (`scripts/measure_latency.py`) into the GUI as a one-click button, and collapse the 14-tab GUI into ~7 logically grouped tabs, removing redundant surfaces.

**Architecture:** The measurement needs the worker's *single* capturer + a SendInput driver, so it runs **inside the worker loop** on a thread-safe request flag (the loop blocks for ~2.5 s, publishes the result via a new telemetry field); the GUI's Calibrate tab fires the request after a countdown and auto-applies `deadtime_ms` + `tau_render_s`. The reorg stacks existing `TuningPanel`s under grouped top-level tabs via a small helper — no panel logic changes.

**Tech Stack:** PySide6 (offscreen-testable via `pytest-qt`/`qtbot`), numpy, OpenCV (`cv2.phaseCorrelate`), pydantic v2 config, `uv run --extra dev pytest`.

## Global Constraints

- Test runner: `uv run --extra dev pytest`. Full suite is currently **706 passing** — keep it green.
- GUI model layer stays **Qt-free + unit-tested**; Qt widgets are thin + offscreen-tested (`QT_QPA_PLATFORM=offscreen`). Box-only pieces (SendInput, bettercam, real frames) are `# pragma: no cover` with injected seams for tests.
- Config changes flow through `ConfigHandle.swap` + `configChanged` → `apply_config_change` (save-first) — never write `config.toml` directly from a panel.
- Cross-thread: worker writes, GUI reads; single-attribute rebinds only (GIL-atomic), mirroring `SnapshotPublisher`/`set_aim_controller`.
- Reuse existing tested cores: `aim/latency.py::estimate_lag`, `recoil/wall_learner.py::measure_shift`. Do NOT duplicate them.
- Each task ends green + committed. Branch: `phase9h-latency-gui-reorg` off `main`.

---

## File Structure

- Create `src/ragnarok/aim/latency_measure.py` — Qt-free `WallLatencyMeasurer` (injected capturer/mouse/shift-fn/clock; testable orchestration of the capture-command-estimate loop).
- Modify `src/ragnarok/telemetry/snapshot.py` — add `latency_ms: float | None` result field.
- Modify `src/ragnarok/worker/loop.py` — measure-request seam + in-tick measurement run.
- Modify `src/ragnarok/gui/counts_panel.py` — "Measure latency" button, countdown, result readout, auto-apply.
- Modify `src/ragnarok/app.py` — give the loop a measure-mouse; grouped tab construction.
- Create `src/ragnarok/gui/tab_groups.py` — `grouped_tab(sections)` helper (stack titled panels in a scroll area).
- Modify `src/ragnarok/gui/tuning_model.py` — no field changes (grouping is in app.py); keep existing `*_FIELDS`.
- Delete the `Profiles` top-level tab wiring (keep `ProfilesPanel` import/export controls, re-homed under "Interface").
- Tests: `tests/aim/test_latency_measure.py`, `tests/gui/test_latency_button.py`, `tests/gui/test_tab_groups.py`, and updates to `tests/gui/test_app_tabs.py` (or create it).

---

## Phase A — Latency measurement in the GUI

### Task 1: Testable measurement orchestrator

**Files:**
- Create: `src/ragnarok/aim/latency_measure.py`
- Test: `tests/aim/test_latency_measure.py`

**Interfaces:**
- Consumes: `ragnarok.aim.latency.estimate_lag(commanded, observed, dt_s, max_lag_frames) -> float | None`.
- Produces: `WallLatencyMeasurer(capturer, mouse, *, duration_s=2.5, amp=40.0, freq_hz=3.0, shift_fn=None, clock=None).run() -> float | None`. `capturer` has `.grab() -> Frame|None`; `mouse` has `.move_relative(dx, dy)`; `shift_fn(prev_gray, cur_gray) -> (dx, dy)` defaults to `recoil.wall_learner.measure_shift`; `clock() -> float` seconds, defaults to `time.perf_counter`.

- [ ] **Step 1: Write the failing test** — a fake capturer emits frames whose scene shift lags the commands; a simple injected `shift_fn` returns the last commanded dx delayed by 3 frames; assert the measured lag ≈ 3·dt.

```python
# tests/aim/test_latency_measure.py
import numpy as np
from ragnarok.aim.latency_measure import WallLatencyMeasurer
from ragnarok.core.types import Frame


class _FakeCap:
    def __init__(self, n):
        self._n = n
        self._i = 0
    def grab(self):
        if self._i >= self._n:
            return None
        self._i += 1
        return Frame(image=np.zeros((16, 16, 3), np.uint8), t_capture_ns=0, region=(0, 0, 16, 16))


class _FakeMouse:
    def __init__(self):
        self.cmds = []
    def move_relative(self, dx, dy):
        self.cmds.append(dx)


def test_measurer_recovers_injected_lag():
    mouse = _FakeMouse()
    # observed scene shift = -(commanded 3 frames ago); measurer records (commanded, observed)
    lag_frames = 3
    def shift_fn(prev, cur):
        i = len(mouse.cmds)                      # number of commands sent so far
        src = mouse.cmds[i - 1 - lag_frames] if i - 1 - lag_frames >= 0 else 0.0
        return (-src, 0.0)                       # scene moves opposite the view
    t = {"s": 0.0}
    def clock():
        t["s"] += 0.01
        return t["s"]
    m = WallLatencyMeasurer(_FakeCap(60), mouse, duration_s=0.55, amp=40.0,
                            freq_hz=3.0, shift_fn=shift_fn, clock=clock)
    lag = m.run()
    assert lag is not None
    assert abs(lag - lag_frames * 0.01) < 1e-2   # ≈ 0.03 s


def test_measurer_returns_none_on_too_few_frames():
    assert WallLatencyMeasurer(_FakeCap(2), _FakeMouse(), duration_s=0.02,
                               clock=lambda: 0.0).run() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/aim/test_latency_measure.py -q`
Expected: FAIL — `ModuleNotFoundError: ragnarok.aim.latency_measure`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/aim/latency_measure.py
"""Qt-free wall latency measurement orchestrator (box-only capture injected).

Oscillates the view via `mouse` while sampling the scene's optical flow via
`shift_fn`, then cross-correlates (aim.latency.estimate_lag) to the round-trip
latency. All I/O is injected so the loop is unit-testable with fakes; the real
run uses the worker's capturer + a SendInput driver + cv2 phase correlation.
"""
from __future__ import annotations

import math
import time

import cv2

from ragnarok.aim.latency import estimate_lag
from ragnarok.recoil.wall_learner import measure_shift


class WallLatencyMeasurer:
    def __init__(self, capturer, mouse, *, duration_s: float = 2.5, amp: float = 40.0,
                 freq_hz: float = 3.0, shift_fn=None, clock=None) -> None:
        self._cap = capturer
        self._mouse = mouse
        self._dur = duration_s
        self._amp = amp
        self._freq = freq_hz
        self._shift = shift_fn or measure_shift
        self._clock = clock or time.perf_counter

    def run(self) -> float | None:
        prev_gray = None
        prev_pos = 0.0
        commanded, observed, times = [], [], []
        t0 = self._clock()
        while self._clock() - t0 < self._dur:
            frame = self._cap.grab()
            if frame is None:
                continue
            t = self._clock() - t0
            gray = self._to_gray(frame.image)
            if prev_gray is not None:
                dx, _ = self._shift(prev_gray, gray)
                observed.append(dx)
                times.append(t)
                pos = self._amp * math.sin(2.0 * math.pi * self._freq * t)
                commanded.append(pos - prev_pos)
                prev_pos = pos
                self._mouse.move_relative(commanded[-1], 0.0)
            prev_gray = gray
        n = min(len(commanded), len(observed))
        if n < 10 or len(times) < 2:
            return None
        dt = (times[-1] - times[0]) / (len(times) - 1)
        return estimate_lag(commanded[:n], observed[:n], dt, max_lag_frames=int(0.25 / dt))

    @staticmethod
    def _to_gray(img):
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/aim/test_latency_measure.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Refactor `scripts/measure_latency.py` to use it (DRY) + commit**

Replace the inline capture loop in `scripts/measure_latency.py`'s `main()` with:
```python
from ragnarok.aim.latency_measure import WallLatencyMeasurer
# ... after cap.start() + mouse.connect() + countdown:
lag = WallLatencyMeasurer(cap, mouse, duration_s=dur).run()
cap.stop()
if lag is None:
    print("could not estimate (low optical-flow signal) — use a more textured wall")
    return
```
Run: `uv run --extra dev python -c "import ast; ast.parse(open('scripts/measure_latency.py').read())"` (Expected: no output = OK).

```bash
git add src/ragnarok/aim/latency_measure.py tests/aim/test_latency_measure.py scripts/measure_latency.py
git commit -m "feat(aim): testable WallLatencyMeasurer; script reuses it"
```

---

### Task 2: Worker measure-request seam + telemetry result

**Files:**
- Modify: `src/ragnarok/telemetry/snapshot.py` (add field)
- Modify: `src/ragnarok/worker/loop.py` (request seam + in-tick run)
- Test: `tests/worker/test_measure_request.py`

**Interfaces:**
- Consumes: `WallLatencyMeasurer` (Task 1).
- Produces: `WorkerLoop.set_measure_mouse(mouse)`, `WorkerLoop.request_latency_measure(duration_s=2.5)`, and a result surfaced as `TelemetrySnapshot.latency_ms: float | None`. `request_latency_measure` is a single-attribute rebind (GIL-atomic); `tick()` consumes it once.

- [ ] **Step 1: Write the failing test** — inject a fake capturer + fake mouse via `set_measure_mouse`, request a measure, run one `tick()`, assert the published snapshot carries a `latency_ms` and the request is cleared.

```python
# tests/worker/test_measure_request.py
import numpy as np
from ragnarok.worker.loop import WorkerLoop
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.core.types import Frame


class _Cap:
    def grab(self):
        return Frame(image=np.zeros((16, 16, 3), np.uint8), t_capture_ns=0, region=(0, 0, 16, 16))
    def stop(self):
        pass


class _Det:
    def detect(self, frame):
        from ragnarok.core.types import Detections
        return Detections(items=())


class _Mouse:
    def move_relative(self, dx, dy):
        pass


class _Prof:
    def record(self, *a):
        pass
    def percentiles(self, *a):
        return (0.0, 0.0)


def test_measure_request_runs_and_publishes(monkeypatch):
    pub = SnapshotPublisher()
    loop = WorkerLoop(_Cap(), _Det(), _Prof(), pub)
    loop.set_measure_mouse(_Mouse())
    # force the measurer to return a fixed lag without a real capture loop
    monkeypatch.setattr("ragnarok.worker.loop.WallLatencyMeasurer.run", lambda self: 0.037)
    loop.request_latency_measure(duration_s=0.1)
    loop.tick()
    snap = pub.latest()
    assert snap is not None and snap.latency_ms == 0.037
    loop.tick()                                   # request consumed -> no re-run
    assert pub.latest().latency_ms is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/worker/test_measure_request.py -q`
Expected: FAIL — `AttributeError: 'WorkerLoop' object has no attribute 'set_measure_mouse'`.

- [ ] **Step 3: Write minimal implementation**

In `src/ragnarok/telemetry/snapshot.py`, add to `TelemetrySnapshot` (after `roi_region`):
```python
    latency_ms: float | None = None          # last measured round-trip latency (Calibrate)
```

In `src/ragnarok/worker/loop.py`:
- import at top: `from ragnarok.aim.latency_measure import WallLatencyMeasurer`
- in `__init__` add: `self._measure_mouse = None`, `self._measure_req = None`, `self._measure_ms = None`
- add methods:
```python
    def set_measure_mouse(self, mouse) -> None:
        self._measure_mouse = mouse

    def request_latency_measure(self, duration_s: float = 2.5) -> None:
        self._measure_req = float(duration_s)    # GIL-atomic rebind
```
- at the TOP of `tick()` (before `self._cap.grab()`), consume the request:
```python
        req = self._measure_req
        if req is not None:
            self._measure_req = None
            self._measure_ms = None
            if self._measure_mouse is not None:
                lag = WallLatencyMeasurer(self._cap, self._measure_mouse, duration_s=req).run()
                self._measure_ms = round(lag * 1000.0, 1) if lag is not None else None
```
- in the `self._pub.publish(TelemetrySnapshot(...))` call, pass `latency_ms=self._measure_ms` and then, immediately after publishing, clear it: `self._measure_ms = None` so it appears in exactly one snapshot.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/worker/test_measure_request.py -q`
Expected: PASS. Then `uv run --extra dev pytest tests/worker tests/telemetry -q` (Expected: PASS, no regressions).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/telemetry/snapshot.py src/ragnarok/worker/loop.py tests/worker/test_measure_request.py
git commit -m "feat(worker): in-tick latency measure-request + telemetry result"
```

---

### Task 3: Calibrate-tab "Measure latency" button + auto-apply

**Files:**
- Modify: `src/ragnarok/gui/counts_panel.py`
- Modify: `src/ragnarok/app.py` (give loop a measure-mouse; pass loop + publisher into the panel)
- Test: `tests/gui/test_latency_button.py`

**Interfaces:**
- Consumes: `WorkerLoop.request_latency_measure`, `TelemetrySnapshot.latency_ms`, `calibration_model`-style swap.
- Produces: `CountsCalibratePanel(handle, *, loop=None, publisher=None, ...)`; new methods `_start_latency_measure()` (countdown then `loop.request_latency_measure`) and `apply_latency_ms(ms)` (swap `aim.deadtime_ms` + `tracking.tau_render_s`, emit `configChanged`). A `QTimer` polls `publisher.latest().latency_ms` and calls `apply_latency_ms` once when it appears.

- [ ] **Step 1: Write the failing test** — construct the panel with a fake loop + a fake publisher whose snapshot exposes `latency_ms`; call `apply_latency_ms(37.0)`; assert `aim.deadtime_ms == 37.0` and `tracking.tau_render_s == 0.037` and `configChanged` fired.

```python
# tests/gui/test_latency_button.py
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.counts_panel import CountsCalibratePanel


class _Loop:
    def __init__(self):
        self.requested = None
    def request_latency_measure(self, duration_s=2.5):
        self.requested = duration_s


def test_apply_latency_sets_deadtime_and_tau(qtbot):
    h = ConfigHandle(AppConfig())
    loop = _Loop()
    panel = CountsCalibratePanel(h, loop=loop, publisher=None)
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.configChanged, timeout=1000):
        panel.apply_latency_ms(37.0)
    assert h.current.aim.deadtime_ms == 37.0
    assert abs(h.current.tracking.tau_render_s - 0.037) < 1e-9


def test_start_measure_requests_on_loop(qtbot):
    loop = _Loop()
    panel = CountsCalibratePanel(ConfigHandle(AppConfig()), loop=loop, publisher=None)
    qtbot.addWidget(panel)
    panel._request_now()                          # skip the countdown in tests
    assert loop.requested is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_latency_button.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'loop'`.

- [ ] **Step 3: Write minimal implementation**

In `src/ragnarok/gui/counts_panel.py`:
- extend `__init__(self, handle, *, reset_provider=None, apply_provider=None, loop=None, publisher=None)`; store `self._loop = loop`, `self._publisher = publisher`.
- add a "Measure latency (aim at wall)" `QPushButton` + a `self.latency_result = QLabel("")` to the layout.
- add methods (import `apply_sensitivity`-style swap from `calibration_model` is not needed; write the swap inline mirroring `_apply_360`):
```python
    def _request_now(self):
        if self._loop is not None:
            self._loop.request_latency_measure(2.5)
            self.latency_result.setText("measuring… aim at a flat wall")
            if self._publisher is not None:
                self._poll = QTimer(self)
                self._poll.setInterval(200)
                self._poll.timeout.connect(self._check_latency)
                self._poll.start()

    def _check_latency(self):
        snap = self._publisher.latest() if self._publisher is not None else None
        if snap is not None and snap.latency_ms is not None:
            self._poll.stop()
            self.apply_latency_ms(snap.latency_ms)

    def apply_latency_ms(self, ms: float):
        cfg = self._handle.current
        new = cfg.model_copy(update={
            "aim": cfg.aim.model_copy(update={"deadtime_ms": float(ms)}),
            "tracking": cfg.tracking.model_copy(update={"tau_render_s": float(ms) / 1000.0})})
        self._handle.swap(new)
        self.latency_result.setText(f"latency = {ms:g} ms  (deadtime + tau_render set)")
        self.configChanged.emit(new)
```
- the button's `clicked` connects to a `_start_latency_measure` that does a 3→2→1 `QTimer` countdown updating a label then calls `_request_now()` (so the user can alt-tab into the game). Countdown is box-only UX; `_request_now`/`apply_latency_ms` are the tested seams.
- Add `from PySide6.QtCore import QTimer` if not already imported (it is).

In `src/ragnarok/app.py`: after building the loop and the SendInput factory, create a measure mouse and pass it + the panel wiring:
```python
    loop.set_measure_mouse(_build_mouse(cfg))
    sens_cal = CountsCalibratePanel(handle, loop=loop, publisher=publisher)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_latency_button.py tests/gui/test_counts_panel.py tests/gui/test_counts_hotkeys.py -q`
Expected: PASS. Then `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app"` (Expected: no error).

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/counts_panel.py src/ragnarok/app.py tests/gui/test_latency_button.py
git commit -m "feat(gui): Calibrate-tab Measure-latency button -> deadtime + tau_render"
```

---

## Phase B — GUI reorganization (14 tabs → 7)

**Target layout** (each grouped tab stacks the named `TuningPanel` sections in one scroll area):
1. **Dashboard** (unchanged) — telemetry.
2. **Aim** — `AIM_FIELDS`.
3. **Targeting** — Detection · Tracking · Friend/Foe sections.
4. **Fire** — Trigger section + `RecoilPanel`.
5. **Calibrate** — Sensitivity 360° + Measure-latency (the `CountsCalibratePanel`).
6. **Interface** — Keybinds · Overlay · Input sections + `ProfilesPanel` (import/export only).
7. **Advanced** — Diagnostics (PID auto-tune) + Motion (WindMouse) sections.

Removed as top-level: standalone **Detection/Tracking/Friend-Foe/Trigger/Motion/Keybinds/Overlay/Input/Profiles** tabs (re-homed above). Per-weapon Profiles framing is dropped (user configures the active weapon manually); `ProfilesPanel`'s Import/Export stays.

### Task 4: `grouped_tab` helper

**Files:**
- Create: `src/ragnarok/gui/tab_groups.py`
- Test: `tests/gui/test_tab_groups.py`

**Interfaces:**
- Produces: `grouped_tab(widgets: list[QWidget]) -> QWidget` — a scrollable container stacking `widgets` vertically (each already carries its own `#header` title). Returns a `QWidget` ready for `tabs.addTab(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/gui/test_tab_groups.py
from PySide6.QtWidgets import QLabel, QScrollArea
from ragnarok.gui.tab_groups import grouped_tab


def test_grouped_tab_stacks_all_widgets(qtbot):
    a, b = QLabel("A"), QLabel("B")
    tab = grouped_tab([a, b])
    qtbot.addWidget(tab)
    assert isinstance(tab, QScrollArea)
    inner = tab.widget()
    kids = inner.findChildren(QLabel)
    assert a in kids and b in kids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_tab_groups.py -q`
Expected: FAIL — `ModuleNotFoundError: ragnarok.gui.tab_groups`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ragnarok/gui/tab_groups.py
"""Stack several titled setting panels into one scrollable top-level tab."""
from __future__ import annotations

from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget


def grouped_tab(widgets: list[QWidget]) -> QScrollArea:
    inner = QWidget()
    lay = QVBoxLayout(inner)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch(1)
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    return area
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_tab_groups.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/tab_groups.py tests/gui/test_tab_groups.py
git commit -m "feat(gui): grouped_tab helper (stack titled panels in one tab)"
```

---

### Task 5: Restructure `app.py` tabs into the 7-tab layout

**Files:**
- Modify: `src/ragnarok/app.py` (tab construction block, ~lines 198-236)
- Test: `tests/gui/test_app_tabs.py` (create)

**Interfaces:**
- Consumes: `grouped_tab` (Task 4), existing `TuningPanel(handle, fields=, on_save=, title=)`, `RecoilPanel`, `DiagnosticsPanel`, `CalibrationPanel`-free (removed), `ProfilesPanel`, `CountsCalibratePanel`.
- Produces: `build_tabs(...) -> tuple[QTabWidget, list[TuningPanel]]` extracted from `main()` so it's offscreen-testable; the 7 tab titles are exactly `["Dashboard", "Aim", "Targeting", "Fire", "Calibrate", "Interface", "Advanced"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/gui/test_app_tabs.py
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.app import build_tabs
from ragnarok.telemetry.snapshot import SnapshotPublisher


def test_seven_grouped_tabs(qtbot):
    tabs, panels = build_tabs(ConfigHandle(AppConfig()), SnapshotPublisher(),
                              loop=None, on_save=lambda c: None, on_changed=lambda c: None)
    qtbot.addWidget(tabs)
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == ["Dashboard", "Aim", "Targeting", "Fire",
                      "Calibrate", "Interface", "Advanced"]
    assert "Profiles" not in titles and "Motion" not in titles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_app_tabs.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_tabs'`.

- [ ] **Step 3: Write minimal implementation** — extract a `build_tabs(handle, publisher, *, loop, on_save, on_changed)` function that constructs every panel (connecting each panel's `configChanged` to `on_changed`, and each `TuningPanel` with `on_save=on_save`), collects the `TuningPanel`s into a list, and assembles the 7 tabs:

```python
def build_tabs(handle, publisher, *, loop, on_save, on_changed):
    tuning = []
    def _tp(fields, title):
        p = TuningPanel(handle, fields=fields, on_save=on_save, title=title)
        p.configChanged.connect(on_changed)
        tuning.append(p)
        return p
    tabs = QTabWidget()
    tabs.addTab(_scroll(DashboardPanel(publisher)), "Dashboard")
    aim = TuningPanel(handle, on_save=on_save, title="Aim")
    aim.configChanged.connect(on_changed); tuning.append(aim)
    tabs.addTab(_scroll(aim), "Aim")
    tabs.addTab(grouped_tab([_tp(DETECTION_FIELDS, "Detection"),
                             _tp(TRACKING_FIELDS, "Tracking"),
                             _tp(CLASSIFICATION_FIELDS, "Friend/Foe")]), "Targeting")
    recoil = RecoilPanel(handle); recoil.configChanged.connect(on_changed)
    tabs.addTab(grouped_tab([_tp(TRIGGER_FIELDS, "Trigger"), recoil]), "Fire")
    sens = CountsCalibratePanel(handle, loop=loop, publisher=publisher)
    sens.configChanged.connect(on_changed)
    tabs.addTab(_scroll(sens), "Calibrate")
    profiles = ProfilesPanel(_profile_store(), handle); profiles.configChanged.connect(on_changed)
    tabs.addTab(grouped_tab([_tp(KEYBIND_FIELDS, "Keybinds"), _tp(OVERLAY_FIELDS, "Overlay"),
                             _tp(INPUT_FIELDS, "Input"), profiles]), "Interface")
    diag = DiagnosticsPanel(handle); diag.configChanged.connect(on_changed)
    tabs.addTab(grouped_tab([diag, _tp(MOTION_FIELDS, "Motion")]), "Advanced")
    return tabs, tuning
```
Then in `main()`, replace the inline tab block with `tabs, tuning_panels = build_tabs(handle, publisher, loop=loop, on_save=_save, on_changed=_on_config_changed)` (keep the `_on_config_changed` closure referencing `tuning_panels`). Fix the `DiagnosticsPanel`/`ProfilesPanel`/`_profile_store` constructor calls to match their real signatures (check current `main()`), and delete the now-dead standalone `addTab` lines.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_app_tabs.py -q` then `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app"`
Expected: PASS + clean import.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/app.py tests/gui/test_app_tabs.py
git commit -m "refactor(gui): 14 tabs -> 7 grouped tabs (Targeting/Fire/Interface/Advanced)"
```

---

### Task 6: Full-suite green + docs

**Files:**
- Modify: `.superpowers/sdd/progress.md` (append), `README`/help text if it enumerates tabs (grep first).

- [ ] **Step 1: Run the whole suite + app import**

Run: `uv run --extra dev pytest` (Expected: all pass, ≥ 706 + new) and `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`.

- [ ] **Step 2: Grep for stale tab references + fix**

Run: `grep -rn "Wizards\|\"Profiles\"\|\"Motion\"\|\"Detection\"" src/ docs/ README* 2>/dev/null`
Fix any user-facing text that still names removed tabs.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: suite green + docs after latency-GUI + tab reorg"
```

---

## Self-Review

**1. Spec coverage.**
- "Incorporate measure latency in the GUI" → Tasks 1–3 (testable measurer, worker request seam + telemetry result, Calibrate-tab button + auto-apply). ✅
- "Reorganize / remove redundant GUI clutter" → Tasks 4–5 (grouped_tab helper, 14→7 tabs, Profiles-per-weapon + standalone field tabs collapsed). ✅

**2. Placeholder scan.** No "TBD"/"add error handling"/"similar to Task N". Task 5 references checking the real `DiagnosticsPanel`/`ProfilesPanel` constructor signatures in the current `main()` — that is a concrete verify step, not a placeholder (their exact args must be read at implementation time; the panels already exist and are wired in the current `app.py`).

**3. Type consistency.**
- `WallLatencyMeasurer(...).run() -> float | None` used identically in Task 1, the script refactor, and Task 2's worker.
- `TelemetrySnapshot.latency_ms` added in Task 2, read in Task 3's `_check_latency`.
- `WorkerLoop.request_latency_measure(duration_s)` / `set_measure_mouse(mouse)` defined in Task 2, called in Task 3 (`_request_now`) and `app.py`.
- `grouped_tab(list[QWidget]) -> QScrollArea` defined in Task 4, used in Task 5.
- `build_tabs(handle, publisher, *, loop, on_save, on_changed) -> (QTabWidget, list)` defined + tested in Task 5.

**Known risk to flag during execution:** the measurement blocks the worker tick for ~2.5 s and moves the view — it only works with the **game focused**, so the GUI countdown must give the user time to alt-tab in; and it needs `deg_per_count`/sensitivity already calibrated for the commanded amplitude to be meaningful. Surface this in the Calibrate-tab label copy.
