"""Phase 2 detection overlay: draw track boxes + labels onto the preview image.

Colorblind-aware team colors (orange / blue / gray in BGR). The transparent
click-through in-game overlay is a later phase; Phase 2 draws on the preview
the MainWindow already shows."""
from __future__ import annotations
import cv2
from ragnarok.core.types import Team

# BGR, deuteranopia-safe (no red/green pairing)
TEAM_BGR = {
    Team.ENEMY: (0, 140, 255),     # orange
    Team.TEAMMATE: (255, 128, 0),  # blue
    Team.UNKNOWN: (160, 160, 160),  # gray
}


def draw_overlay(image, tracks, scale: float):
    """Draw each track's box + '#id team' label on `image` (BGR, in place).

    `scale` maps ROI/full-res track coords to the (downscaled) preview.
    """
    for tr in tracks:
        x1, y1, x2, y2 = (int(v * scale) for v in tr.xyxy)
        col = TEAM_BGR.get(tr.team, TEAM_BGR[Team.UNKNOWN])
        cv2.rectangle(image, (x1, y1), (x2, y2), col, 1)
        cv2.putText(image, f"#{tr.track_id} {tr.team.value}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return image
