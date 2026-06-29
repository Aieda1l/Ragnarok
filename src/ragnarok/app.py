from __future__ import annotations
import sys
from pathlib import Path
import os
from PySide6.QtWidgets import QApplication
from ragnarok.config.store import load_config
from ragnarok.capture.factory import create_capturer
from ragnarok.detection.factory import create_detector
from ragnarok.latency.profiler import StageProfiler
from ragnarok.telemetry.snapshot import SnapshotPublisher
from ragnarok.worker.loop import WorkerLoop
from ragnarok.wiring import build_tracker, build_classifier
from ragnarok.gui.worker_thread import WorkerThread
from ragnarok.gui.main_window import MainWindow

def _config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Ragnarok"
    return base / "config.toml"

def _build_aim_controller(cfg, commanded_buffer):
    """Build the AimController from cfg (Windows-only deps imported lazily).

    Wires the Phase 4 collaborators: selected aimer, motion shaper, velocity
    smoother, adaptive lead, recoil compensator, and a safety-gated trigger bot
    (with its own key provider), plus a commanded-motion buffer for the GMC.
    """
    from ragnarok.aim.keys import AsyncKeyStateProvider, make_aim_active
    from ragnarok.aim.mouse import SendInputMouseDriver, MouseButton
    from ragnarok.aim.fov import fov_deg_to_radius_px
    from ragnarok.aim.select import TargetSelector
    from ragnarok.aim.imm import IMMManager
    from ragnarok.aim.velocity import VelocitySmoother
    from ragnarok.aim.controller import AimController
    from ragnarok.latency.adaptive_lead import AdaptiveLead
    from ragnarok.trigger.bot import TriggerBot
    from ragnarok.wiring import build_aimer, build_shaper, build_recoil

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    retain_px = fov_deg_to_radius_px(a.retain_fov_deg, a.hfov_deg, a.screen_width_px)
    selector = TargetSelector(fov_px=fov_px, retain_fov_px=retain_px,
                              dwell_ms=a.dwell_ms, switch_margin=a.switch_margin,
                              head_frac=a.head_frac)
    mouse = SendInputMouseDriver()
    mouse.connect()
    is_active = make_aim_active(AsyncKeyStateProvider(a.aim_key), toggle=a.toggle)

    trigger = None
    trigger_active = None
    if cfg.trigger.enabled:
        btn = {"left": MouseButton.LEFT, "right": MouseButton.RIGHT,
               "middle": MouseButton.MIDDLE}[cfg.trigger.button]
        trigger = TriggerBot(mouse=mouse,
                             activation_delay_s=cfg.trigger.activation_delay_ms / 1000.0,
                             button=btn)
        trigger_active = make_aim_active(
            AsyncKeyStateProvider(cfg.trigger.trigger_key), toggle=False)

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
    cfg = load_config(_config_path())
    publisher = SnapshotPublisher()
    from ragnarok.tracking.egomotion import CommandedMotionBuffer
    cmd_buffer = CommandedMotionBuffer()
    aim_controller = _build_aim_controller(cfg, cmd_buffer) if cfg.aim.enabled else None
    loop = WorkerLoop(
        create_capturer(cfg.capture), create_detector(cfg.detection),
        StageProfiler(), publisher,
        tracker=build_tracker(cfg, gmc_buffer=cmd_buffer),
        classifier=build_classifier(cfg),
        aim_controller=aim_controller,
    )
    worker = WorkerThread(loop)
    window = MainWindow(publisher)
    window.show()
    worker.start()
    app.aboutToQuit.connect(worker.stop)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
