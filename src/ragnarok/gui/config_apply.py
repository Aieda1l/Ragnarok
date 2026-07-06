"""Persist + hot-apply a live config change (spec §13).

Extracted from app.py so the save-then-reload behavior is unit-testable. Every
non-TuningPanel config change (calibration, profile load, recoil, diagnostics
tuning) funnels through here. The SAVE happens FIRST so the user's change survives
even if the worker rebuild raises (previously these panels swapped the live handle
but never wrote to disk, so a calibration was lost on restart).
"""
from __future__ import annotations


def apply_config_change(new_cfg, *, save, refresh, reload) -> list:
    """Save ``new_cfg``, refresh dependent panels, then hot-reload the worker.

    ``save``/``reload`` exceptions are captured and returned (never raised) so a
    Qt slot can't abort the event loop. ``refresh`` is an iterable of zero-arg
    callables. Returns a list of ``(stage, exception)`` for any failures.
    """
    errors: list = []
    try:
        save(new_cfg)
    except Exception as exc:  # noqa: BLE001 — GUI must stay alive
        errors.append(("save", exc))
    for refresh_one in refresh:
        refresh_one()
    try:
        reload(new_cfg)
    except Exception as exc:  # noqa: BLE001 — GUI must stay alive
        errors.append(("reload", exc))
    return errors
