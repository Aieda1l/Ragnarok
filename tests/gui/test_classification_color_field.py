from ragnarok.config.schema import AppConfig
from ragnarok.gui.tuning_model import CLASSIFICATION_FIELDS, set_field


def test_enemy_color_exposed_as_choice():
    fs = next(f for f in CLASSIFICATION_FIELDS if f.path == "classification.enemy_color")
    assert fs.kind == "choice"
    assert set(fs.choices) == {"red", "purple", "yellow"}   # default-palette keys


def test_set_enemy_color_updates_config():
    cfg = set_field(AppConfig(), "classification.enemy_color", "purple")
    assert cfg.classification.enemy_color == "purple"
