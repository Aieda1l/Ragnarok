import numpy as np
from ragnarok.core.types import Track, Team
from ragnarok.gui.overlay import draw_overlay, TEAM_BGR

def _track(team, track_id=1, xyxy=(4, 4, 40, 40)):
    return Track(track_id=track_id, xyxy=xyxy, confidence=0.9, class_id=0, team=team)

def test_draw_overlay_paints_enemy_color():
    img = np.zeros((64, 64, 3), np.uint8)
    out = draw_overlay(img, (_track(Team.ENEMY),), scale=1.0)
    enemy = np.array(TEAM_BGR[Team.ENEMY], dtype=np.uint8)
    matches = np.all(out == enemy, axis=2)
    assert matches.any()  # the enemy color appears on the drawn box

def test_team_colors_are_distinct():
    assert len({TEAM_BGR[Team.ENEMY], TEAM_BGR[Team.TEAMMATE], TEAM_BGR[Team.UNKNOWN]}) == 3

def test_draw_overlay_empty_tracks_is_noop():
    img = np.zeros((10, 10, 3), np.uint8)
    out = draw_overlay(img, (), scale=1.0)
    assert int(out.sum()) == 0

def test_draw_overlay_scale_applies():
    # box (0,0,60,60): at scale 0.5 it maps to (0,0,30,30), so row y=50 (below the
    # scaled box and away from the top label) is empty; unscaled it is painted.
    scaled = draw_overlay(np.zeros((64, 64, 3), np.uint8),
                          (_track(Team.ENEMY, xyxy=(0, 0, 60, 60)),), scale=0.5)
    assert int(scaled[50, :, :].sum()) == 0
    unscaled = draw_overlay(np.zeros((64, 64, 3), np.uint8),
                            (_track(Team.ENEMY, xyxy=(0, 0, 60, 60)),), scale=1.0)
    assert int(unscaled[50, :, :].sum()) > 0
