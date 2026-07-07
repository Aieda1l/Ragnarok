from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.counts_panel import CountsCalibratePanel


class _Loop:
    def __init__(self):
        self.requested = None

    def request_latency_measure(self, duration_s=2.5):
        self.requested = duration_s


def test_apply_latency_sets_deadtime_and_tau(qtbot):
    h = ConfigHandle(AppConfig())
    panel = CountsCalibratePanel(h, loop=_Loop(), publisher=None)
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.configChanged, timeout=1000):
        panel.apply_latency_ms(37.0)
    assert h.current.aim.deadtime_ms == 37.0
    assert abs(h.current.tracking.tau_render_s - 0.037) < 1e-9


def test_request_now_requests_on_loop(qtbot):
    loop = _Loop()
    panel = CountsCalibratePanel(ConfigHandle(AppConfig()), loop=loop, publisher=None)
    qtbot.addWidget(panel)
    panel._request_now()                        # skip the box-only countdown
    assert loop.requested == 2.5


def test_no_loop_is_a_safe_noop(qtbot):
    panel = CountsCalibratePanel(ConfigHandle(AppConfig()))    # loop/publisher default None
    qtbot.addWidget(panel)
    panel._request_now()                        # must not raise
