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
