# Vendored motion-only tracker

## Source

- **Repository:** https://github.com/NirAharon/BoT-SORT
- **Commit:** `251985436d6712aaf682aaaf5f71edb4987224bd` (2022-10-30)
- **License:** MIT (see upstream `LICENSE`)
- **Files adapted:** `tracker/bot_sort.py`, `tracker/kalman_filter.py`,
  `tracker/basetrack.py`, `tracker/matching.py`

## Layout here

| Vendored file        | Upstream origin                                   |
|----------------------|---------------------------------------------------|
| `basetrack.py`       | `tracker/basetrack.py` (verbatim)                 |
| `kalman_xywh.py`     | `tracker/kalman_filter.py` (verbatim logic)       |
| `ops.py`             | STrack box-conversion statics from `bot_sort.py`  |
| `iou.py`             | `matching.ious`/`iou_distance` reimplemented      |
| `strack.py`          | `bot_sort.py` `STrack` (ReID stripped)            |
| `botsort_core.py`    | `bot_sort.py` `BoTSORT.update` (ReID/CMC stripped)|

## Modifications

1. **ReID / appearance removed.** All FastReID / `with_reid` / `update_features`
   / feature-deque / `embedding_distance` / `fuse_motion` code is deleted. The
   tracker is motion-only. (`import torch`, `fast_reid` gone.)
2. **CMC / optical flow replaced.** The OpenCV global-motion `GMC.apply(img, dets)`
   is replaced by an **injected** `ego.estimate(frame) -> 2x3 affine`
   (`ragnarok.tracking.egomotion.EgoMotion`). `STrack.multi_gmc` is kept verbatim
   as the injection seam. (`import cv2` gone.)
3. **Linear assignment.** `lap.lapjv` replaced by
   `ragnarok.tracking.associate.linear_assignment` (scipy
   `linear_sum_assignment`). (`import lap` gone.)
4. **IoU.** `cython_bbox.bbox_overlaps` replaced by a pure-numpy `iou_batch`.
   (`import cython_bbox` gone.)
5. **Mahalanobis gate added.** A 2-DOF Mahalanobis gate
   (`ragnarok.tracking.gate.mahalanobis_gate`, threshold `CHI2_GATE_2DOF=5.9915`)
   is applied to the first (high-score) IoU association cost before solving.
6. **numpy 2.x fixes.** Removed deprecated `np.float` usages (now `float`).
7. **`mot20` flag dropped** — score fusion (`fuse_score`) always applied.

## Dependency invariant

The vendored modules import **numpy + scipy only** (scipy reached transitively
through `associate`/`gate`/`kalman_xywh`). No `torch`, `cv2`, `lap`,
`cython_bbox`, `matplotlib`, or FastReID. Enforced by the import grep in the
tracking test suite / Task 2 step 5.
