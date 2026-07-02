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
