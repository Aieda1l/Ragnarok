"""Calibration-wizard orchestration over the Phase 5B pure solvers (spec §11).

ZERO Qt: turns a known calibration turn (or a commanded/measured trace) into a
re-validated AppConfig via ConfigHandle.swap. Live data collection (the turn,
the optical-flow trace) is box-only; these helpers are the pure analysis+apply.
"""
from __future__ import annotations

from ragnarok.tracking.calibration import estimate_tau_render, solve_deg_per_count


def _swap(handle, updates):
    """Build a re-validated AppConfig with the given {section: {field: value}}
    updates and swap it in. Constructing each sub-model through its class
    re-validates (model_copy(update=) would not), so an invalid result raises."""
    cfg = handle.current
    section_updates = {}
    for section, fields in updates.items():
        sub = getattr(cfg, section)
        section_updates[section] = sub.__class__(**{**sub.model_dump(), **fields})
    new_cfg = cfg.model_copy(update=section_updates)
    handle.swap(new_cfg)
    return new_cfg


def apply_sensitivity(handle, *, total_counts: float, measured_deg: float):
    """From a known calibration turn: deg_per_count = measured_deg / total_counts.

    Sets tracking.deg_per_count (SIGNED, for the GMC back-projection) and
    aim.sensitivity (magnitude, for the px<->count conversion). Raises if the
    result is invalid (e.g. a zero measured turn -> sensitivity gt=0 violated).
    """
    dpc = solve_deg_per_count(total_counts, measured_deg)
    return _swap(handle, {
        "tracking": {"deg_per_count": dpc},
        "aim": {"sensitivity": abs(dpc)},
    })


def apply_tau_render(handle, *, commanded, measured, dt_s: float, max_lag_s: float = 0.1):
    """Set tracking.tau_render_s from a commanded/measured motion trace."""
    tau = estimate_tau_render(commanded, measured, dt_s, max_lag_s=max_lag_s)
    return _swap(handle, {"tracking": {"tau_render_s": tau}})
