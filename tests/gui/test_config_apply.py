from ragnarok.gui.config_apply import apply_config_change


def test_saves_first_then_refreshes_then_reloads():
    order = []
    saved = {}
    errs = apply_config_change(
        "CFG",
        save=lambda c: (order.append("save"), saved.update(cfg=c)),
        refresh=[lambda: order.append("refresh")],
        reload=lambda c: order.append("reload"))
    assert saved["cfg"] == "CFG"
    assert order == ["save", "refresh", "reload"]      # persist BEFORE reload
    assert errs == []


def test_calibration_persists_even_when_reload_fails():
    saved = {}

    def boom(_):
        raise RuntimeError("worker rebuild failed")

    errs = apply_config_change(
        "CFG", save=lambda c: saved.update(cfg=c), refresh=[], reload=boom)
    assert saved["cfg"] == "CFG"                        # saved despite reload error
    assert len(errs) == 1 and errs[0][0] == "reload"
    assert isinstance(errs[0][1], RuntimeError)


def test_save_failure_is_captured_not_raised():
    def boom(_):
        raise OSError("disk full")

    errs = apply_config_change(
        "CFG", save=boom, refresh=[], reload=lambda c: None)
    assert errs == [("save", errs[0][1])] and isinstance(errs[0][1], OSError)
