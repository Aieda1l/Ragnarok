import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig
from ragnarok.config.store import ConfigHandle
from ragnarok.gui.tuning_model import (
    AIM_FIELDS, FieldSpec, get_field, set_field, apply_field)


def test_aim_fields_cover_key_knobs_and_are_wellformed():
    paths = {f.path for f in AIM_FIELDS}
    assert {"aim.enabled", "aim.kp", "aim.aimer", "aim.aim_fov_deg"} <= paths
    for f in AIM_FIELDS:
        assert f.path.startswith("aim.")
        assert f.kind in {"float", "int", "bool", "choice"}
        if f.kind == "choice":
            assert len(f.choices) >= 2


def test_get_and_set_roundtrip_preserves_other_sections():
    cfg = AppConfig()
    assert get_field(cfg, "aim.kp") == cfg.aim.kp
    new = set_field(cfg, "aim.kp", 0.8)
    assert new.aim.kp == 0.8
    assert cfg.aim.kp != 0.8                      # original untouched (frozen)
    assert new.detection == cfg.detection          # other sections preserved
    assert new.capture == cfg.capture


def test_set_field_revalidates_and_rejects_out_of_range():
    cfg = AppConfig()
    with pytest.raises(ValidationError):
        set_field(cfg, "aim.kp", 99.0)             # schema: kp <= 2.0
    with pytest.raises(ValidationError):
        set_field(cfg, "aim.switch_margin", 1.5)   # schema: < 1.0


def test_apply_field_swaps_handle():
    h = ConfigHandle(AppConfig())
    returned = apply_field(h, "aim.enabled", True)
    assert h.current.aim.enabled is True
    assert returned is h.current
