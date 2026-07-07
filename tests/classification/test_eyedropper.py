import numpy as np

from ragnarok.classification.eyedropper import hsv_band_from_bgr, profile_from_band
from ragnarok.classification.color import is_enemy_frame
from ragnarok.config.schema import AppConfig
from ragnarok.wiring import build_classifier
from ragnarok.classification.base import HSVRingClassifier


def test_hsv_band_centered_on_green():
    band = hsv_band_from_bgr((0, 255, 0), h_tol=10, s_tol=70, v_tol=70)  # pure green
    h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = band
    assert h_lo <= 60 <= h_hi                       # OpenCV green hue ≈ 60
    assert s_hi == 255 and v_hi == 255              # open to the bright end
    assert 0 <= h_lo and h_hi <= 179


def test_profile_from_band_matches_that_color():
    band = hsv_band_from_bgr((0, 255, 0))
    prof = profile_from_band(band)
    assert prof.name == "custom"
    green = np.zeros((60, 60, 3), np.uint8)
    green[:] = (0, 255, 0)
    # box inside the frame so its colour ring is fully green -> enemy frame
    assert is_enemy_frame(green, (20.0, 20.0, 40.0, 40.0), prof, frac_threshold=0.1, thickness=4)


def test_build_classifier_uses_custom_band_over_palette():
    band = hsv_band_from_bgr((0, 255, 0))
    cfg = AppConfig().model_copy(update={
        "classification": AppConfig().classification.model_copy(update={"custom_band": band})})
    clf = build_classifier(cfg)
    assert isinstance(clf, HSVRingClassifier)
    assert clf._profile.name == "custom"            # custom band beat palette=default/red
