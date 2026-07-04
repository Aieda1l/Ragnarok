import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.calibration_model import apply_sensitivity, apply_tau_render


def test_apply_sensitivity_sets_signed_deg_per_count_and_magnitude():
    h = ConfigHandle(AppConfig())
    new = apply_sensitivity(h, total_counts=1000.0, measured_deg=22.0)
    assert new.tracking.deg_per_count == pytest.approx(0.022)
    assert new.aim.sensitivity == pytest.approx(0.022)
    assert h.current is new


def test_apply_sensitivity_preserves_sign_but_sensitivity_is_magnitude():
    h = ConfigHandle(AppConfig())
    new = apply_sensitivity(h, total_counts=1000.0, measured_deg=-22.0)   # inverted turn
    assert new.tracking.deg_per_count == pytest.approx(-0.022)            # sign kept for GMC
    assert new.aim.sensitivity == pytest.approx(0.022)                    # magnitude for px<->count


def test_apply_sensitivity_zero_measure_raises_not_silent():
    h = ConfigHandle(AppConfig())
    with pytest.raises(ValidationError):                                  # sensitivity gt=0
        apply_sensitivity(h, total_counts=1000.0, measured_deg=0.0)


def test_apply_tau_render_sets_tracking_tau():
    h = ConfigHandle(AppConfig())
    # measured trails commanded by 2 samples at dt=1ms -> tau ~ 0.002 s
    commanded = [0, 0, 1, 0, 0, 0, 0, 0]
    measured = [0, 0, 0, 0, 1, 0, 0, 0]
    new = apply_tau_render(h, commanded=commanded, measured=measured, dt_s=0.001)
    assert new.tracking.tau_render_s == pytest.approx(0.002, abs=1e-9)
    assert h.current is new
