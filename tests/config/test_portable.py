import pytest

from ragnarok.config.schema import AppConfig
from ragnarok.config.portable import import_config, export_config


def test_export_then_import_roundtrips(tmp_path):
    cfg = AppConfig().model_copy(update={
        "aim": AppConfig().aim.model_copy(update={"aim_point": "detected_head",
                                                  "head_class_id": 1, "sensitivity": 3.5}),
        "recoil": AppConfig().recoil.model_copy(update={"pattern": ((0.0, 2.0), (0.0, 5.0)),
                                                        "scale": 1.25, "enabled": True}),
    })
    path = tmp_path / "sub" / "export.toml"      # parent dir created on write
    export_config(cfg, path)
    assert path.exists()
    assert import_config(path) == cfg             # frozen models compare by value


def test_import_missing_path_raises_not_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_config(tmp_path / "nope.toml")     # must NOT auto-create a default


def test_import_schema_invalid_raises(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[aim]\nsensitivity = -5.0\n", encoding="utf-8")  # violates ge=0
    with pytest.raises(Exception):
        import_config(bad)
