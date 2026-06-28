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
from ragnarok.gui.worker_thread import WorkerThread
from ragnarok.gui.main_window import MainWindow

def _config_path() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home())) / "Ragnarok"
    return base / "config.toml"

def _build_aim_controller(cfg):
    """Build the AimController from cfg.aim (Windows-only deps imported lazily).

    NOTE: AimController is ENEMY-only; for it to engage live you must also wire a
    friend/foe classifier that labels enemies (HSVRingClassifier + an enemy-color
    config) — a small follow-up, since the worker currently defaults to
    NullClassifier/IdentityTracker.
    """
    from ragnarok.aim.mouse import SendInputMouseDriver
    from ragnarok.aim.keys import AsyncKeyStateProvider, make_aim_active
    from ragnarok.aim.fov import fov_deg_to_radius_px
    from ragnarok.aim.select import TargetSelector
    from ragnarok.aim.imm import IMMManager
    from ragnarok.aim.aimers import FlickAimer, FeedbackAimer
    from ragnarok.aim.controller import AimController

    a = cfg.aim
    fov_px = fov_deg_to_radius_px(a.aim_fov_deg, a.hfov_deg, a.screen_width_px)
    retain_px = fov_deg_to_radius_px(a.retain_fov_deg, a.hfov_deg, a.screen_width_px)
    selector = TargetSelector(fov_px=fov_px, retain_fov_px=retain_px,
                              dwell_ms=a.dwell_ms, switch_margin=a.switch_margin,
                              head_frac=a.head_frac)
    if a.aimer == "flick":
        aimer = FlickAimer(flick_speed_px_s=a.flick_speed_px_s)
    else:
        aimer = FeedbackAimer(kp=a.kp, max_step_px=a.max_step_px, ema_alpha=a.ema_alpha)
    mouse = SendInputMouseDriver()
    mouse.connect()
    is_active = make_aim_active(AsyncKeyStateProvider(a.aim_key), toggle=a.toggle)
    return AimController(a, selector=selector, imm_manager=IMMManager(), aimer=aimer,
                        mouse=mouse, is_aim_active=is_active, roi_size=cfg.capture.roi_size)

def main() -> int:
    app = QApplication(sys.argv)
    cfg = load_config(_config_path())
    publisher = SnapshotPublisher()
    aim_controller = _build_aim_controller(cfg) if cfg.aim.enabled else None
    loop = WorkerLoop(
        create_capturer(cfg.capture), create_detector(cfg.detection),
        StageProfiler(), publisher, aim_controller=aim_controller,
    )
    worker = WorkerThread(loop)
    window = MainWindow(publisher)
    window.show()
    worker.start()
    app.aboutToQuit.connect(worker.stop)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
