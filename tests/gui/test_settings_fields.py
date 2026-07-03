import pytest
from pydantic import ValidationError
from ragnarok.config.schema import AppConfig
from ragnarok.gui.tuning_model import (
    set_field, TRACKING_FIELDS, CLASSIFICATION_FIELDS, TRIGGER_FIELDS,
    RECOIL_FIELDS, MOTION_FIELDS)

ALL = (("tracking", TRACKING_FIELDS), ("classification", CLASSIFICATION_FIELDS),
       ("trigger", TRIGGER_FIELDS), ("recoil", RECOIL_FIELDS), ("motion", MOTION_FIELDS))


def _sample_value(f, cfg):
    if f.kind == "bool":
        return True
    if f.kind == "choice":
        return f.choices[-1]
    # numeric: midpoint of the declared range (falls back to 1)
    if f.minimum is not None and f.maximum is not None:
        mid = (f.minimum + f.maximum) / 2.0
        return int(mid) if f.kind == "int" else mid
    return 1


def test_every_field_targets_its_section_and_is_wellformed():
    for section, fields in ALL:
        assert len(fields) >= 2
        for f in fields:
            assert f.path.startswith(section + ".")
            assert f.kind in {"float", "int", "bool", "choice"}
            if f.kind == "choice":
                assert len(f.choices) >= 2


def test_every_field_get_and_set_roundtrips_on_default_config():
    cfg = AppConfig()
    for _section, fields in ALL:
        for f in fields:
            cur = _sample_value(f, cfg)
            new = set_field(cfg, f.path, cur)               # re-validates; must not raise
            assert new is not cfg


def test_int_field_coerces_and_choice_rejects_bad_value():
    new = set_field(AppConfig(), "tracking.track_buffer", 45.0)   # spinbox float -> int
    assert new.tracking.track_buffer == 45 and isinstance(new.tracking.track_buffer, int)
    with pytest.raises(ValidationError):
        set_field(AppConfig(), "tracking.backend", "not_a_backend")


def test_signed_deg_per_count_accepts_negative():
    new = set_field(AppConfig(), "tracking.deg_per_count", -0.05)
    assert new.tracking.deg_per_count == -0.05
