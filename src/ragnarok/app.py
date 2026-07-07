from __future__ import annotations
import sys
from pathlib import Path
import os
from PySide6.QtWidgets import QApplication, QTabWidget, QScrollArea
from ragnarok.config.store import load_config, save_config, ConfigHandle
from ragnarok.capture.factory import create_capturer
from ragnarok.detection.factory import create_detector
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.worker.loop import WorkerLoop
from ragnarok.wiring import build_tracker, build_classifier, build_mouse_driver
from ragnarok.gui.worker_thread import WorkerThread
from ragnarok.gui.main_window import MainWindow
from ragnarok.gui.overlay_window import FovOverlay
from ragnarok.gui.tuning_panel import TuningPanel
from ragnarok.gui.tuning_model import (
    TRACKING_FIELDS, CLASSIFICATION_FIELDS, TRIGGER_FIELDS, MOTION_FIELDS,
    INPUT_FIELDS, DETECTION_FIELDS, OVERLAY_FIELDS, KEYBIND_FIELDS)
from ragnarok.gui.diagnostics_panel import DiagnosticsPanel
from ragnarok.gui.profiles_panel import ProfilesPanel
from ragnarok.gui.counts_panel import CountsCalibratePanel
from ragnarok.gui.dashboard_panel import DashboardPanel
from ragnarok.gui.recoil_panel import RecoilPanel
from ragnarok.gui.chrome_frame import ChromeFrame
from ragnarok.gui.live_config import AimReloader, WorkerReloader
from ragnarok.gui.config_apply import apply_config_change
from ragnarok.config.profiles import ProfileStore

def _config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Ragnarok"
    return base / "config.toml"

def _profiles_dir() -> Path:
    return _config_path().parent / "profiles"

def _build_mouse(cfg):
    """Box-only mouse driver (SendInput or Arduino) from cfg.input."""
    from ragnarok.aim.mouse import SendInputMouseDriver

    def _sendinput():
        m = SendInputMouseDriver(compensate_ballistics=cfg.input.compensate_ballistics)
        m.connect()
        return m

    def _arduino(c):
        from ragnarok.aim.arduino import ArduinoDriver, build_arduino_transport
        d = ArduinoDriver(transport=build_arduino_transport(c))
        d.connect()
        return d

    return build_mouse_driver(cfg, sendinput_factory=_sendinput, arduino_factory=_arduino)


def _build_trigger_bot(cfg, mouse):
    """A safety-gated TriggerBot + its key provider (None if trigger disabled)."""
    from ragnarok.aim.keys import AsyncKeyStateProvider, make_aim_active
    from ragnarok.aim.mouse import MouseButton
    from ragnarok.trigger.bot import TriggerBot

    if not cfg.trigger.enabled:
        return None, None
    btn = {"left": MouseButton.LEFT, "right": MouseButton.RIGHT,
           "middle": MouseButton.MIDDLE}[cfg.trigger.button]
    trigger = TriggerBot(mouse=mouse,
                         activation_delay_s=cfg.trigger.activation_delay_ms / 1000.0,
                         button=btn)
    trigger_active = make_aim_active(
        AsyncKeyStateProvider(cfg.trigger.trigger_key), toggle=False)
    return trigger, trigger_active


def _build_fire_component(cfg, commanded_buffer):
    """The loop's per-tick fire/aim component: AimController when aim is enabled,
    else a standalone TriggerController when only the trigger is enabled, else None."""
    if cfg.aim.enabled:
        return _build_aim_controller(cfg, commanded_buffer)
    if cfg.trigger.enabled:
        return _build_trigger_controller(cfg)
    return None


def _build_trigger_controller(cfg):
    """Standalone trigger bot (fires without aim assist) for aim-disabled use."""
    from ragnarok.trigger.controller import TriggerController
    from ragnarok.wiring import build_recoil

    mouse = _build_mouse(cfg)
    trigger, trigger_active = _build_trigger_bot(cfg, mouse)
    return TriggerController(
        cfg.aim, trigger=trigger, trigger_active=trigger_active, mouse=mouse,
        roi_size=cfg.capture.roi_size, recoil=build_recoil(cfg))


def _build_aim_controller(cfg, commanded_buffer):
    """Build the AimController from cfg (Windows-only deps imported lazily).

    Wires the Phase 4 collaborators: selected aimer, motion shaper, velocity
    smoother, adaptive lead, recoil compensator, and a safety-gated trigger bot
    (with its own key provider), plus a commanded-motion buffer for the GMC.
    """
    from ragnarok.aim.keys import AsyncKeyStateProvider, make_aim_active
    from ragnarok.aim.fov import fov_deg_to_radius_px
    from ragnarok.aim.select import TargetSelector
    from ragnarok.aim.imm import IMMManager
    from ragnarok.aim.velocity import VelocitySmoother
    from ragnarok.aim.controller import AimController
    from ragnarok.latency.adaptive_lead import AdaptiveLead
    from ragnarok.wiring import build_aimer, build_shaper, build_recoil

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    retain_px = fov_deg_to_radius_px(a.retain_fov_deg, a.hfov_deg, a.screen_width_px)
    selector = TargetSelector(fov_px=fov_px, retain_fov_px=retain_px,
                              dwell_ms=a.dwell_ms, switch_margin=a.switch_margin,
                              head_frac=a.head_frac,
                              head_class_id=a.head_class_id if a.aim_point == "detected_head" else None)
    mouse = _build_mouse(cfg)
    is_active = make_aim_active(AsyncKeyStateProvider(a.aim_key), toggle=a.toggle)
    trigger, trigger_active = _build_trigger_bot(cfg, mouse)

    return AimController(
        a, selector=selector, imm_manager=IMMManager(),
        aimer=build_aimer(cfg), mouse=mouse, is_aim_active=is_active,
        roi_size=cfg.capture.roi_size,
        shaper=build_shaper(cfg),
        vel_smoother=VelocitySmoother(alpha=a.vel_smooth_alpha, max_px_s=a.vel_clamp_px_s),
        adaptive_lead=AdaptiveLead(alpha=a.lead_alpha, base_latency_s=a.lead_ms / 1000.0),
        recoil=build_recoil(cfg),
        trigger=trigger, trigger_active=trigger_active,
        commanded_buffer=commanded_buffer,
    )

def main() -> int:
    app = QApplication(sys.argv)
    from ragnarok.gui import theme
    theme.apply_theme(app)                 # Cyberpunk 2077 skin (spec §10.1)
    cfg = load_config(_config_path())
    publisher = SnapshotPublisher()
    if cfg.tracking.gmc == "feedforward":
        reasons = []
        if not cfg.aim.enabled:
            reasons.append("aim is disabled (no commanded-motion producer)")
        if cfg.tracking.deg_per_count == 0.0:
            reasons.append("deg_per_count is 0 (uncalibrated)")
        if cfg.tracking.backend != "botsort":
            reasons.append(f"tracking backend is {cfg.tracking.backend!r}, not 'botsort'")
        if reasons:
            import warnings
            warnings.warn("GMC 'feedforward' is enabled but inert: " + "; ".join(reasons))
    from ragnarok.tracking.egomotion import CommandedMotionBuffer
    cmd_buffer = CommandedMotionBuffer()
    aim_controller = _build_fire_component(cfg, cmd_buffer)
    detector = create_detector(cfg.detection)
    if cfg.dynamic_roi.enabled:                 # opt-in SEARCH/TRACK crop+upscale
        from ragnarok.detection.dynamic import DynamicRoiDetector
        from ragnarok.detection.roi import DynamicRoiPlanner
        detector = DynamicRoiDetector(detector, DynamicRoiPlanner(cfg.dynamic_roi),
                                      model_input_px=cfg.dynamic_roi.model_input_px)
    loop = WorkerLoop(
        create_capturer(cfg.capture), detector,
        StageProfiler(), publisher,
        # only feed GMC the buffer when aim is enabled (the buffer's only producer)
        tracker=build_tracker(cfg, gmc_buffer=cmd_buffer if cfg.aim.enabled else None),
        classifier=build_classifier(cfg),
        aim_controller=aim_controller,
    )
    loop.set_measure_mouse(_build_mouse(cfg))       # for the Calibrate-tab latency measure
    # Live config: the tuning panel edits funnel through ConfigHandle.swap and
    # rebuild the aim controller in-place (spec §13 immutable snapshot swap).
    handle = ConfigHandle(cfg)
    aim_reloader = AimReloader(loop, _build_fire_component, commanded_buffer=cmd_buffer)
    reloader = WorkerReloader(
        loop, aim_reloader=aim_reloader,
        build_tracker=build_tracker, build_classifier=build_classifier,
        commanded_buffer=cmd_buffer, initial_cfg=cfg)

    def _save(c):
        save_config(c, _config_path())

    tuning_panels: list[TuningPanel] = []

    def _on_config_changed(new_cfg):
        # Any panel edit / profile load / calibration: PERSIST to disk (so it
        # survives restart — the Calibrate/Profiles/Recoil/Diagnostics panels swap
        # the live handle but had no on_save), repaint the other tabs, then
        # hot-reload. A rebuild can raise on invalid config (e.g. arduino driver
        # with an empty port); apply_config_change captures save/reload failures
        # so they never escape into the Qt event loop (PySide6 aborts on an
        # unhandled slot exception) — WorkerReloader leaves _prev stale on failure.
        errors = apply_config_change(
            new_cfg, save=_save,
            refresh=[tp.refresh for tp in tuning_panels],
            reload=reloader.reload)
        for stage, exc in errors:
            import warnings
            warnings.warn(f"config {stage} failed (GUI kept alive): {exc}")

    def _scroll(widget):
        # keep the window compact: overflowing options scroll instead of
        # stretching the window taller.
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    tabs = QTabWidget()
    dashboard = DashboardPanel(publisher)
    tabs.addTab(_scroll(dashboard), "Dashboard")
    aim_panel = TuningPanel(handle, on_save=_save, title="Aim")
    aim_panel.configChanged.connect(_on_config_changed)
    tuning_panels.append(aim_panel)
    tabs.addTab(_scroll(aim_panel), "Aim")
    diagnostics = DiagnosticsPanel(handle)
    diagnostics.configChanged.connect(_on_config_changed)
    tabs.addTab(_scroll(diagnostics), "Diagnostics")
    for fields, title in ((DETECTION_FIELDS, "Detection"),
                          (TRACKING_FIELDS, "Tracking"),
                          (CLASSIFICATION_FIELDS, "Friend/Foe"),
                          (TRIGGER_FIELDS, "Trigger"),
                          (MOTION_FIELDS, "Motion"),
                          (KEYBIND_FIELDS, "Keybinds"),
                          (OVERLAY_FIELDS, "Overlay"),
                          (INPUT_FIELDS, "Input")):
        p = TuningPanel(handle, fields=fields, on_save=_save, title=title)
        p.configChanged.connect(_on_config_changed)
        tuning_panels.append(p)
        tabs.addTab(_scroll(p), title)
    recoil = RecoilPanel(handle)                 # dedicated spray-pattern editor
    recoil.configChanged.connect(_on_config_changed)
    tabs.addTab(_scroll(recoil), "Recoil")
    profiles = ProfilesPanel(ProfileStore(_profiles_dir()), handle)
    profiles.configChanged.connect(_on_config_changed)
    tabs.addTab(_scroll(profiles), "Profiles")
    sens_cal = CountsCalibratePanel(handle, loop=loop, publisher=publisher)
    sens_cal.configChanged.connect(_on_config_changed)
    tabs.addTab(_scroll(sens_cal), "Calibrate")

    worker = WorkerThread(loop)
    window = MainWindow(publisher, controls=ChromeFrame(tabs))   # Cyberpunk panel chrome
    window.resize(780, 820)                                       # compact; tabs scroll
    window.show()
    # Smart-lock FOV overlay: frameless/click-through, own timer, read-only.
    # Reads the LIVE config so FOV-ring / aim-point edits show immediately.
    # Sized ~16:9 to the configured screen width (box-only positioning refinement
    # per-monitor is deferred; see overlay_window docstring).
    overlay = FovOverlay(publisher, lambda: handle.current)
    overlay.resize(cfg.aim.screen_width_px, int(cfg.aim.screen_width_px * 9 / 16))
    overlay.show()
    worker.start()
    app.aboutToQuit.connect(worker.stop)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
