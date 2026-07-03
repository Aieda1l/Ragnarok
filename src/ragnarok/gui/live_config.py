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


class WorkerReloader:
    """Change-gated rebuild of the CI-safe worker components on a config swap
    (spec §13). Composes the aim-controller reload with tracker/classifier
    rebuilds, rebuilding only the components whose config slice actually changed
    (frozen pydantic models compare by value) so an aim-slider tweak doesn't
    needlessly reset tracking. Detector reload (torch/TRT) is box-only and NOT
    handled here.
    """

    def __init__(self, loop, *, aim_reloader, build_tracker, build_classifier,
                 commanded_buffer=None, initial_cfg=None) -> None:
        self._loop = loop
        self._aim = aim_reloader
        self._build_tracker = build_tracker
        self._build_classifier = build_classifier
        self._buf = commanded_buffer
        self._prev = initial_cfg

    def reload(self, cfg) -> None:
        prev = self._prev
        # aim controller depends on aim + trigger + recoil + motion
        if prev is None or (cfg.aim, cfg.trigger, cfg.recoil, cfg.motion) != (
                prev.aim, prev.trigger, prev.recoil, prev.motion):
            self._aim.reload(cfg)
        # tracker depends on the tracking slice (+ the aim.enabled GMC-buffer gate)
        if prev is None or cfg.tracking != prev.tracking or \
                cfg.aim.enabled != prev.aim.enabled:
            gmc = self._buf if cfg.aim.enabled else None
            self._loop.set_tracker(self._build_tracker(cfg, gmc_buffer=gmc))
        # classifier depends on the classification slice
        if prev is None or cfg.classification != prev.classification:
            self._loop.set_classifier(self._build_classifier(cfg))
        self._prev = cfg
