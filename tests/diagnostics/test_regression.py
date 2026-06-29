"""§15 step-response regression: the closed loop must stay well-behaved.

Runs BEFORE the Task 8 PID change to lock current P behaviour, and stays green
after (ki=kd=0 defaults reproduce P). Drives FeedbackAimer against the synthetic
integrator plant — fully CI-safe (no GPU/cursor/game).
"""
from __future__ import annotations
import numpy as np
from ragnarok.aim.aimers import FeedbackAimer
from ragnarok.config.schema import DiagnosticsConfig
from ragnarok.diagnostics.plant import AimPlant, simulate_closed_loop
from ragnarok.diagnostics.results import compute_step_result


def _drive(aimer, *, setpoint, dt, n):
    aimer.reset()
    plant = AimPlant(dt_s=dt)

    def ctrl(error, dt_):
        return aimer.step((0.0, 0.0), (error, 0.0), dt_)[0]   # 1-D: x-axis only

    t_s, measured, _ = simulate_closed_loop(ctrl, plant, setpoint=setpoint,
                                            n_steps=n, dt_s=dt)
    t_ns = (t_s * 1e9).astype(np.int64)
    return compute_step_result(t_ns, measured, y0=0.0, y_final=setpoint, hz=1.0 / dt)


def test_feedback_p_controller_is_well_behaved():
    cfg = DiagnosticsConfig()
    # Pure P (ema_alpha=1, kff=0), large max_step so the clamp doesn't slew-limit.
    aimer = FeedbackAimer(kp=0.3, max_step_px=1e9, ema_alpha=1.0)
    res = _drive(aimer, setpoint=100.0, dt=0.002, n=600)
    assert res.overshoot_pct <= cfg.reg_max_overshoot_pct   # P on integrator: ~0
    assert res.settling_s is not None                       # it settles
    assert res.rise_s is not None and res.rise_s > 0.0
