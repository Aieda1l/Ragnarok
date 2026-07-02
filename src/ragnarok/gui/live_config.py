"""Config-swap -> live worker reload coordinator (spec §13, aim-scoped).

ZERO Qt / SendInput: turns a freshly-swapped ``AppConfig`` into a rebuilt aim
controller and atomically rebinds it into the running loop. The controller
builder is INJECTED so this is unit-testable without SendInput/torch; ``app.py``
passes the real (box-only) builder.

Rebuild-on-swap (not in-place mutation) is the model: a full rebuild makes every
aim field take effect at once and starts the controller cleanly disengaged.
"""
from __future__ import annotations


class AimReloader:
    def __init__(self, loop, build_aim, commanded_buffer=None) -> None:
        self._loop = loop
        self._build = build_aim
        self._buf = commanded_buffer

    def reload(self, cfg) -> None:
        if cfg.aim.enabled:
            self._loop.set_aim_controller(self._build(cfg, self._buf))
        else:
            self._loop.set_aim_controller(None)
