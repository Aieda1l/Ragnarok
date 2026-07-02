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
