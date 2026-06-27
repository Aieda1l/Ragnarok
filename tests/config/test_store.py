from ragnarok.config.schema import AppConfig, CaptureConfig
from ragnarok.config.store import load_config, save_config, ConfigHandle

def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "config.toml"
    cfg = AppConfig(capture=CaptureConfig(roi_size=512, target_fps=240))
    save_config(cfg, p)
    loaded = load_config(p)
    assert loaded.capture.roi_size == 512
    assert loaded.capture.target_fps == 240

def test_load_missing_writes_defaults(tmp_path):
    p = tmp_path / "config.toml"
    cfg = load_config(p)
    assert p.exists()
    assert cfg.capture.roi_size == 384

def test_handle_atomic_swap():
    h = ConfigHandle(AppConfig())
    assert h.current.capture.roi_size == 384
    h.swap(AppConfig(capture=CaptureConfig(roi_size=256)))
    assert h.current.capture.roi_size == 256
