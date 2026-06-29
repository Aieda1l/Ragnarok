"""Tests for the relay experiment against the synthetic plant."""
from __future__ import annotations
from ragnarok.diagnostics.relay_experiment import RelayController, run_relay_tune, RelayTuneResult
from ragnarok.diagnostics.plant import AimPlant


def test_relay_controller_bangs():
    r = RelayController(d=2.0)
    assert r.step(5.0, 0.01) == 2.0       # positive error -> +d
    assert r.step(-5.0, 0.01) == -2.0     # negative error -> -d


def test_relay_tune_yields_positive_gains_on_lagged_plant():
    # An integrator with dead-time + lag sustains a limit cycle under relay.
    plant = AimPlant(dt_s=0.001, lag_tau_s=0.02, dead_time_s=0.01)
    res = run_relay_tune(plant, d=5.0, n_steps=4000, dt_s=0.001)
    assert isinstance(res, RelayTuneResult)
    assert res.tu > 0.0 and res.ku > 0.0
    assert res.kp > 0.0 and res.ki > 0.0
