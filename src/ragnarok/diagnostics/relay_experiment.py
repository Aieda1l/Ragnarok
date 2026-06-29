"""Run a relay-feedback experiment against a plant to seed PID gains (spec §11).

CI runs this against the synthetic AimPlant; live desktop/in-game samplers are
thin box-only adapters over the same (error -> command) seam.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop
from ragnarok.diagnostics.relay import limit_cycle, ku_from_relay, zn_seed


class RelayController:
    def __init__(self, *, d: float, hysteresis: float = 0.0) -> None:
        self._d = d
        self._h = hysteresis
        self._out = d

    def step(self, error: float, dt: float) -> float:
        if error > self._h:
            self._out = self._d
        elif error < -self._h:
            self._out = -self._d
        # within the hysteresis band: hold previous output
        return self._out


@dataclass(frozen=True)
class RelayTuneResult:
    ku: float
    tu: float
    kp: float
    ki: float
    kd: float


def run_relay_tune(plant: AimPlant, *, d: float, n_steps: int, dt_s: float,
                   setpoint: float = 0.0, hysteresis: float = 0.0,
                   rule: str = "low_overshoot") -> RelayTuneResult:
    relay = RelayController(d=d, hysteresis=hysteresis)
    t_s, measured, _ = simulate_closed_loop(relay.step, plant, setpoint=setpoint,
                                            n_steps=n_steps, dt_s=dt_s)
    a, tu = limit_cycle(t_s, measured)
    ku = ku_from_relay(d, a) if a > 0.0 else 0.0
    seed = zn_seed(ku, tu, rule=rule) if (ku > 0.0 and tu > 0.0) else {"kp": 0.0, "ki": 0.0, "kd": 0.0}
    return RelayTuneResult(ku=ku, tu=tu, kp=seed["kp"], ki=seed["ki"], kd=seed["kd"])
