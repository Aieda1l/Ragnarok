from ragnarok.config.schema import AppConfig
from ragnarok.gui.live_config import AimReloader


class _Loop:
    def __init__(self):
        self.controller = "sentinel"
    def set_aim_controller(self, c):
        self.controller = c


def test_reload_builds_and_sets_when_enabled():
    loop = _Loop()
    seen = {}
    def build(cfg, buf):
        seen["cfg"] = cfg
        seen["buf"] = buf
        return "CTRL"
    r = AimReloader(loop, build, commanded_buffer="BUF")
    cfg = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"enabled": True})})
    r.reload(cfg)
    assert loop.controller == "CTRL"
    assert seen["cfg"] is cfg and seen["buf"] == "BUF"


def test_reload_disables_without_building_when_aim_off():
    loop = _Loop()
    called = {"n": 0}
    def build(cfg, buf):
        called["n"] += 1
        return "CTRL"
    r = AimReloader(loop, build)
    r.reload(AppConfig())                           # aim.enabled defaults False
    assert loop.controller is None
    assert called["n"] == 0                          # no rebuild when disabled


def test_reload_builds_standalone_trigger_when_aim_off_trigger_on():
    loop = _Loop()
    called = {"n": 0}
    def build(cfg, buf):
        called["n"] += 1
        return "TRIG"
    r = AimReloader(loop, build)
    base = AppConfig()
    cfg = base.model_copy(update={"trigger": base.trigger.model_copy(update={"enabled": True})})
    r.reload(cfg)                                    # aim off + trigger on -> build standalone
    assert loop.controller == "TRIG"
    assert called["n"] == 1


def test_reload_releases_previous_controller_on_swap():
    """Phase 9P: swapping controllers releases the old one's held button (the shared
    mouse driver is NOT closed — it is owned by main() and reused)."""
    class _Rel:
        def __init__(self): self.released = 0
        def release(self): self.released += 1
    prev = _Rel()

    class _L:
        def __init__(self): self._aim = prev; self.controller = prev
        def set_aim_controller(self, c): self._aim = c; self.controller = c
    loop = _L()
    r = AimReloader(loop, build_aim=lambda cfg, buf: "NEW")
    cfg = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"enabled": True})})
    r.reload(cfg)
    assert loop.controller == "NEW"
    assert prev.released == 1


def test_reload_rolls_back_on_build_failure():
    import pytest

    class _L:
        def __init__(self): self.controller = "OLD"; self._aim = "OLD"
        def set_aim_controller(self, c): self.controller = c; self._aim = c
    loop = _L()

    def boom(cfg, buf):
        raise RuntimeError("arduino port busy")
    r = AimReloader(loop, build_aim=boom)
    cfg = AppConfig().model_copy(update={"aim": AppConfig().aim.model_copy(update={"enabled": True})})
    with pytest.raises(RuntimeError):
        r.reload(cfg)
    assert loop.controller == "OLD"        # build raised before the swap -> old kept


from ragnarok.gui.live_config import WorkerReloader


class _FullLoop:
    def __init__(self):
        self.tracker = None
        self.classifier = None
    def set_tracker(self, t): self.tracker = t
    def set_classifier(self, c): self.classifier = c


class _RecordingAim:
    def __init__(self): self.reloads = 0
    def reload(self, cfg): self.reloads += 1


def _make(initial):
    loop = _FullLoop()
    aim = _RecordingAim()
    bt_calls, bc_calls = [], []
    def bt(cfg, *, gmc_buffer=None): bt_calls.append(gmc_buffer); return "T"
    def bc(cfg): bc_calls.append(cfg); return "C"
    r = WorkerReloader(loop, aim_reloader=aim, build_tracker=bt, build_classifier=bc,
                       commanded_buffer="BUF", initial_cfg=initial)
    return r, loop, aim, bt_calls, bc_calls


def test_first_reload_rebuilds_everything_when_no_initial():
    r, loop, aim, bt, bc = _make(None)
    r.reload(AppConfig())
    assert aim.reloads == 1 and loop.tracker == "T" and loop.classifier == "C"
    assert bt == [None]                                   # aim disabled -> no gmc buffer


def test_only_changed_section_rebuilds():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"tracking": base.tracking.model_copy(update={"track_buffer": 45})})
    r.reload(new)
    assert loop.tracker == "T" and len(bt) == 1           # tracker rebuilt
    assert aim.reloads == 0 and len(bc) == 0              # aim + classifier untouched


def test_aim_slider_change_does_not_reset_tracker():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"aim": base.aim.model_copy(update={"kp": 0.9})})
    r.reload(new)
    assert aim.reloads == 1                                # aim rebuilt
    assert len(bt) == 0 and len(bc) == 0                  # tracker/classifier untouched


def test_enabling_aim_reattaches_gmc_buffer_to_tracker():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"aim": base.aim.model_copy(update={"enabled": True})})
    r.reload(new)
    assert aim.reloads == 1                                # aim slice changed
    assert bt == ["BUF"]                                  # enabled -> gmc buffer fed


def test_driver_change_rebuilds_aim_controller():
    base = AppConfig()
    r, loop, aim, bt, bc = _make(base)
    new = base.model_copy(update={"input": base.input.model_copy(update={"mouse_driver": "arduino"})})
    r.reload(new)
    assert aim.reloads == 1                                # input change -> aim rebuild
    assert len(bt) == 0 and len(bc) == 0                  # tracker/classifier untouched


def test_reload_build_failure_keeps_prev_stale_so_retry_rebuilds():
    import pytest
    base = AppConfig()
    loop = _FullLoop()
    class _BadAim:
        def __init__(self): self.calls = 0
        def reload(self, cfg):
            self.calls += 1
            raise RuntimeError("build failed (e.g. arduino empty port)")
    aim = _BadAim()
    r = WorkerReloader(loop, aim_reloader=aim, build_tracker=lambda c, *, gmc_buffer=None: "T",
                       build_classifier=lambda c: "C", initial_cfg=base)
    changed = base.model_copy(update={"input": base.input.model_copy(update={"mouse_driver": "arduino"})})
    with pytest.raises(RuntimeError):
        r.reload(changed)                                 # aim build raises -> propagates
    assert aim.calls == 1
    with pytest.raises(RuntimeError):
        r.reload(changed)                                 # _prev stayed stale -> retries (not skipped as "unchanged")
    assert aim.calls == 2
