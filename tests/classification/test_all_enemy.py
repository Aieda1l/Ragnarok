import numpy as np

from ragnarok.core.types import Track, Tracks, Team
from ragnarok.classification.base import AllEnemyClassifier
from ragnarok.config.schema import AppConfig
from ragnarok.wiring import build_classifier


def _tracks():
    return Tracks(items=(
        Track(track_id=1, xyxy=(0.0, 0.0, 10.0, 10.0), confidence=0.9, class_id=0, team=Team.UNKNOWN),
        Track(track_id=2, xyxy=(5.0, 5.0, 9.0, 9.0), confidence=0.8, class_id=1, team=Team.TEAMMATE),
    ))


def test_all_enemy_stamps_every_track_enemy():
    out = AllEnemyClassifier().classify(_tracks(), np.zeros((20, 20, 3), np.uint8))
    assert [t.team for t in out] == [Team.ENEMY, Team.ENEMY]


def test_build_classifier_disabled_returns_all_enemy():
    cfg = AppConfig().model_copy(update={
        "classification": AppConfig().classification.model_copy(update={"enabled": False})})
    assert isinstance(build_classifier(cfg), AllEnemyClassifier)


def test_build_classifier_enabled_is_not_all_enemy():
    assert not isinstance(build_classifier(AppConfig()), AllEnemyClassifier)  # HSV classifier
