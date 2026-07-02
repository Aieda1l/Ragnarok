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
