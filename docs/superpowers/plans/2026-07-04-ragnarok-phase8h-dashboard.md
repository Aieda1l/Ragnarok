# Phase 8H — Dashboard Telemetry Sparklines (function-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Dashboard" tab that plots rolling FPS and loop-latency (p50/p99) history as sparklines, driven by the telemetry publisher — with the history buffer and the plot geometry pure and CI-tested, and only the QPainter render box-only.

**Architecture:** Mirrors the overlay pattern: a Qt-free `dashboard_model` holds a `TelemetryHistory` ring buffer (ingests snapshots) and a pure `sparkline_points` that maps a series into polyline points inside a rect. A thin `DashboardPanel(QWidget)` pulls the latest snapshot on its own timer (dedup by `seq`), pushes to history, updates numeric labels, and paints the sparklines with `QPainter`. No new dependency (no pyqtgraph) — the sparkline geometry is computed here and drawn with QPainter, so it stays offscreen-testable.

**Tech Stack:** Python 3.11+, `collections.deque`, PySide6 (`QPainter`/`QTimer`), reused `telemetry.snapshot.SnapshotPublisher`/`TelemetrySnapshot`, `gui.theme` palette, `core.clock.now_ns`, pytest-qt. No torch/GPU/network in any test.

## Global Constraints

- **`dashboard_model` imports ZERO Qt.** `TelemetryHistory` + `sparkline_points` are pure. The widget is the only Qt file; widget tests use `qtbot`.
- **Read-only telemetry view.** The Dashboard only *reads* `SnapshotPublisher.latest()`; it never touches config or the worker. No `configChanged`.
- **Dedup by `seq`.** The panel timer runs independent of the worker; it must not double-count the same snapshot (compare `snap.seq` to the last ingested).
- **No new runtime dependency.** Do not add pyqtgraph; render sparklines with `QPainter` over the pure geometry.
- **Palette reuse.** Use `gui.theme` colors (electric yellow for FPS, cyan for p50, alert-red for p99) — consistent with the overlay.
- **Additive:** new module + widget + one tab; existing signatures unchanged.
- TDD, one deliverable per task, commit per task. Runner: `uv run --extra dev pytest`. Baseline: **606 passed**.

---

## File Structure

- **Create** `src/ragnarok/gui/dashboard_model.py` — `TelemetryHistory`, `sparkline_points`.
- **Create** `src/ragnarok/gui/dashboard_panel.py` — `DashboardPanel(QWidget)`.
- **Modify** `src/ragnarok/app.py` — add the Dashboard tab (box-only glue).
- **Create** tests: `tests/gui/test_dashboard_model.py`, `tests/gui/test_dashboard_panel.py`.

---

### Task 1: `TelemetryHistory` ring buffer

**Files:**
- Create: `src/ragnarok/gui/dashboard_model.py`
- Test: `tests/gui/test_dashboard_model.py`

**Interfaces:**
- Consumes: `TelemetrySnapshot` (duck-typed: `.fps`, `.loop_ms_p50`, `.loop_ms_p99`).
- Produces: `TelemetryHistory(maxlen=240)` with `push(*, fps, p50, p99)`, `push_snapshot(snap)`, `series(key)->tuple[float,...]` (key in `{"fps","p50","p99"}`), `stats()->dict`, `__len__`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_dashboard_model.py`:

```python
from ragnarok.gui.dashboard_model import TelemetryHistory


class _Snap:
    def __init__(self, fps, p50, p99):
        self.fps, self.loop_ms_p50, self.loop_ms_p99 = fps, p50, p99


def test_history_records_series_and_stats():
    h = TelemetryHistory(maxlen=10)
    assert len(h) == 0 and h.series("fps") == ()
    h.push(fps=120.0, p50=5.0, p99=9.0)
    h.push_snapshot(_Snap(60.0, 8.0, 15.0))
    assert h.series("fps") == (120.0, 60.0)
    assert h.series("p50") == (5.0, 8.0) and h.series("p99") == (9.0, 15.0)
    assert h.stats() == {"fps": 60.0, "p50": 8.0, "p99": 15.0}     # latest values
    assert len(h) == 2


def test_history_is_bounded_ring_buffer():
    h = TelemetryHistory(maxlen=3)
    for i in range(5):
        h.push(fps=float(i), p50=0.0, p99=0.0)
    assert h.series("fps") == (2.0, 3.0, 4.0)                       # oldest dropped
    assert len(h) == 3


def test_empty_stats_is_zeroed():
    assert TelemetryHistory().stats() == {"fps": 0.0, "p50": 0.0, "p99": 0.0}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_dashboard_model.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/dashboard_model.py`:

```python
"""Pure telemetry-history + sparkline geometry for the Dashboard (spec §10.3).

ZERO Qt: a bounded ring buffer of FPS / loop-latency samples and a function that
maps a series into polyline points inside a rect. The QPainter render lives in
dashboard_panel; all the math is here so it is unit-testable without a display.
"""
from __future__ import annotations

from collections import deque


class TelemetryHistory:
    KEYS = ("fps", "p50", "p99")

    def __init__(self, maxlen: int = 240) -> None:
        self._series = {k: deque(maxlen=maxlen) for k in self.KEYS}

    def push(self, *, fps: float, p50: float, p99: float) -> None:
        self._series["fps"].append(float(fps))
        self._series["p50"].append(float(p50))
        self._series["p99"].append(float(p99))

    def push_snapshot(self, snap) -> None:
        self.push(fps=snap.fps, p50=snap.loop_ms_p50, p99=snap.loop_ms_p99)

    def series(self, key: str) -> tuple[float, ...]:
        return tuple(self._series[key])

    def stats(self) -> dict[str, float]:
        return {k: (self._series[k][-1] if self._series[k] else 0.0)
                for k in self.KEYS}

    def __len__(self) -> int:
        return len(self._series["fps"])
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_dashboard_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/dashboard_model.py tests/gui/test_dashboard_model.py
git commit -m "feat(dashboard): TelemetryHistory ring buffer"
```

---

### Task 2: `sparkline_points` geometry

**Files:**
- Modify: `src/ragnarok/gui/dashboard_model.py`
- Test: `tests/gui/test_dashboard_model.py`

**Interfaces:**
- Produces: `sparkline_points(values, *, x0, y0, w, h, y_min=None, y_max=None) -> tuple[tuple[float,float], ...]` — polyline points; higher value → higher on screen (smaller y); empty → `()`; single point → mid-height; flat series → mid-line.

- [ ] **Step 1: Write the failing test** — append to `tests/gui/test_dashboard_model.py`:

```python
from ragnarok.gui.dashboard_model import sparkline_points


def test_sparkline_empty_and_single():
    assert sparkline_points([], x0=0, y0=0, w=100, h=50) == ()
    pts = sparkline_points([7.0], x0=0, y0=0, w=100, h=50)
    assert pts == ((0.0, 25.0),)                              # single -> mid-height


def test_sparkline_maps_endpoints_and_inverts_y():
    # increasing series over the full width; min at bottom, max at top
    pts = sparkline_points([0.0, 5.0, 10.0], x0=10, y0=20, w=100, h=40)
    assert pts[0] == (10.0, 60.0)                             # min -> bottom (y0+h)
    assert pts[-1] == (110.0, 20.0)                           # max -> top (y0)
    assert pts[1] == (60.0, 40.0)                             # mid x, mid y


def test_sparkline_flat_series_is_midline():
    pts = sparkline_points([3.0, 3.0, 3.0], x0=0, y0=0, w=90, h=30)
    assert [p[1] for p in pts] == [15.0, 15.0, 15.0]          # flat -> mid-line


def test_sparkline_explicit_range():
    # with y_min=0,y_max=100 a value of 50 sits at mid-height regardless of data
    pts = sparkline_points([50.0, 50.0], x0=0, y0=0, w=10, h=100, y_min=0.0, y_max=100.0)
    assert pts[0] == (0.0, 50.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_dashboard_model.py -q`
Expected: FAIL (`sparkline_points` undefined).

- [ ] **Step 3: Implement** — append to `src/ragnarok/gui/dashboard_model.py`:

```python
def sparkline_points(values, *, x0: float, y0: float, w: float, h: float,
                     y_min: float | None = None, y_max: float | None = None):
    """Map ``values`` to polyline points inside the rect (x0,y0,w,h).

    Higher value -> higher on screen (smaller y). Empty -> (); single point ->
    mid-height; a flat series (or zero span) -> the mid-line.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return ()
    mid = y0 + h / 2.0
    if n == 1:
        return ((float(x0), mid),)
    lo = min(vals) if y_min is None else y_min
    hi = max(vals) if y_max is None else y_max
    span = hi - lo
    pts = []
    for i, v in enumerate(vals):
        x = x0 + (i / (n - 1)) * w
        if span <= 0.0:
            y = mid
        else:
            frac = (v - lo) / span
            y = y0 + (1.0 - frac) * h          # invert: bigger value -> smaller y
        pts.append((float(x), float(y)))
    return tuple(pts)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/gui/test_dashboard_model.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ragnarok/gui/dashboard_model.py tests/gui/test_dashboard_model.py
git commit -m "feat(dashboard): pure sparkline_points geometry"
```

---

### Task 3: `DashboardPanel` widget + app wiring

**Files:**
- Create: `src/ragnarok/gui/dashboard_panel.py`
- Modify: `src/ragnarok/app.py`
- Test: `tests/gui/test_dashboard_panel.py`; full suite.

**Interfaces:**
- Consumes: `SnapshotPublisher.latest()`, `TelemetryHistory`/`sparkline_points`, `gui.theme`, `core.clock.now_ns`.
- Produces: `DashboardPanel(publisher, *, interval_ms=200)` — a `QWidget` with `history` (a `TelemetryHistory`), a `_tick()` that ingests the latest snapshot (dedup by `seq`) and repaints, and `fps_label`/`lat_label`.

- [ ] **Step 1: Write the failing test** — `tests/gui/test_dashboard_panel.py`:

```python
from PySide6.QtGui import QImage
from ragnarok.telemetry.snapshot import TelemetrySnapshot, SnapshotPublisher
from ragnarok.gui.dashboard_panel import DashboardPanel


def _snap(seq, fps=120.0):
    return TelemetrySnapshot(fps=fps, loop_ms_p50=5.0, loop_ms_p99=9.0,
                             detection_count=0, preview=None, seq=seq)


def test_tick_ingests_and_dedups_by_seq(qtbot):
    pub = SnapshotPublisher()
    panel = DashboardPanel(pub)
    qtbot.addWidget(panel)
    pub.publish(_snap(1, fps=100.0))
    panel._tick()
    panel._tick()                                       # same seq -> not double-counted
    assert len(panel.history) == 1
    pub.publish(_snap(2, fps=60.0))
    panel._tick()
    assert panel.history.series("fps") == (100.0, 60.0)
    assert "60" in panel.fps_label.text()


def test_tick_no_snapshot_is_noop(qtbot):
    panel = DashboardPanel(SnapshotPublisher())
    qtbot.addWidget(panel)
    panel._tick()                                       # latest() is None -> no crash
    assert len(panel.history) == 0


def test_paints_without_error(qtbot):
    pub = SnapshotPublisher()
    panel = DashboardPanel(pub)
    qtbot.addWidget(panel)
    panel.resize(400, 200)
    for s in range(1, 6):
        pub.publish(_snap(s, fps=float(100 + s)))
        panel._tick()
    img = QImage(400, 200, QImage.Format_ARGB32)
    img.fill(0)
    panel.render(img)                                   # exercises paintEvent
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/gui/test_dashboard_panel.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** — create `src/ragnarok/gui/dashboard_panel.py`:

```python
"""Dashboard tab: rolling FPS + loop-latency sparklines (spec §10.3).

Read-only telemetry view. Pulls the latest snapshot on its own timer (dedup by
seq), pushes to a pure TelemetryHistory, and paints sparklines with QPainter over
the pure geometry (no pyqtgraph). Full Cyberpunk styling is a later box-only pass.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ragnarok.gui import theme
from ragnarok.gui.dashboard_model import TelemetryHistory, sparkline_points

_SERIES = (("fps", theme.ELECTRIC_YELLOW), ("p50", theme.CYAN), ("p99", theme.ALERT_RED))


class DashboardPanel(QWidget):
    def __init__(self, publisher, *, interval_ms: int = 200) -> None:
        super().__init__()
        self._pub = publisher
        self.history = TelemetryHistory()
        self._last_seq = None
        self.setMinimumHeight(160)

        layout = QVBoxLayout(self)
        self.fps_label = QLabel("FPS --")
        self.lat_label = QLabel("loop p50 -- ms  p99 -- ms")
        layout.addWidget(self.fps_label)
        layout.addWidget(self.lat_label)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        snap = self._pub.latest()
        if snap is None or snap.seq == self._last_seq:
            return
        self._last_seq = snap.seq
        self.history.push_snapshot(snap)
        s = self.history.stats()
        self.fps_label.setText(f"FPS {s['fps']:.1f}")
        self.lat_label.setText(f"loop p50 {s['p50']:.1f} ms  p99 {s['p99']:.1f} ms")
        self.update()

    def paintEvent(self, event) -> None:
        w = float(self.width())
        band = max(1.0, (self.height() - 8) / len(_SERIES))
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            for i, (key, color) in enumerate(_SERIES):
                pts = sparkline_points(self.history.series(key),
                                       x0=4.0, y0=4.0 + i * band, w=w - 8.0,
                                       h=band - 4.0)
                if len(pts) < 2:
                    continue
                p.setPen(QPen(QColor(color), 1))
                p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in pts]))
        finally:
            p.end()
```

- [ ] **Step 4: Run to verify the widget tests pass**

Run: `uv run --extra dev pytest tests/gui/test_dashboard_panel.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire into `app.py`** — add the import:

```python
from ragnarok.gui.dashboard_panel import DashboardPanel
```

and add the Dashboard as the FIRST tab (spec §10.3 lists Dashboard first). Right after `tabs = QTabWidget()`:

```python
    dashboard = DashboardPanel(publisher)
    tabs.addTab(dashboard, "Dashboard")
```

(placed before the `aim_panel` tab so Dashboard is index 0).

- [ ] **Step 6: Verify app import + full suite**

Run: `QT_QPA_PLATFORM=offscreen uv run --extra dev python -c "import ragnarok.app; print('ok')"`
Expected: prints `ok`.
Run: `uv run --extra dev pytest -q`
Expected: PASS — 606 baseline + new tests, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/ragnarok/gui/dashboard_panel.py src/ragnarok/app.py tests/gui/test_dashboard_panel.py
git commit -m "feat(dashboard): DashboardPanel sparklines + Dashboard tab"
```

---

## Self-Review

**Spec coverage (§10.3 Dashboard):**
- Live preview + p50/p99 latency & FPS graphs → `TelemetryHistory` + `sparkline_points` + `DashboardPanel` sparklines + numeric labels (Tasks 1, 2, 3). ✅ (The live *preview image* already lives in `MainWindow`; this adds the latency/FPS history graphs.)
- Decoupled from the hot loop → own timer, dedup by `seq`, read-only (Task 3). ✅

**Placeholder scan:** no TBD/placeholder logic; every step has literal code.

**Type consistency:** `TelemetryHistory`/`sparkline_points` (Tasks 1, 2) consumed by `DashboardPanel` (Task 3) and app. `push_snapshot` reads `.fps`/`.loop_ms_p50`/`.loop_ms_p99` — the real `TelemetrySnapshot` fields. `DashboardPanel(publisher, *, interval_ms)` takes the same `SnapshotPublisher` the overlay/main window use. Palette keys reuse `gui.theme`.

**Honest deferrals (box-only / later):** axis labels/gridlines/tooltips, a live-preview image on this tab (already in MainWindow), configurable window length, and the full Cyberpunk styling (glow, gridlines, gauges) are box-only/visual; the sparkline is a minimal QPainter polyline (no pyqtgraph dependency) this slice. This is a telemetry-only view — it never touches config or the worker.
