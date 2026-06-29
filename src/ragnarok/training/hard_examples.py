"""Hard-example mining selection (spec §12 step 6).

Pure policy: pick the frames the detector did worst on (missed entirely, or
low max confidence) to push back to Roboflow for the next dataset version. The
actual push is the Roboflow client's job (Plan 6B).
"""
from __future__ import annotations


def select_hard_examples(records, *, conf_threshold: float) -> list:
    out = []
    for item_id, max_conf in records:
        if max_conf is None or max_conf < conf_threshold:
            out.append(item_id)
    return out
