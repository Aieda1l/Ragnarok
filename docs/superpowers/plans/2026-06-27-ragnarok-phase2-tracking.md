# Ragnarok Phase 2 — Tracking + Friend/Foe + Overlay Implementation Plan

> **For agentic workers:** Implement task-by-task, TDD, one commit per task. Steps use checkbox (`- [ ]`) syntax. This plan is executed via a Workflow (ultracode) but is equally runnable inline.

**Goal:** Add multi-object tracking (stable IDs through occlusion/motion), HSV-outline friend/foe classification with a temporal vote, and a detection overlay on the preview — wired into the worker loop without breaking Phase 1.

**Architecture:** Detector → **Tracker** (vendored motion-only BoT-SORT, injectable ego-motion affine = identity in Phase 2, Mahalanobis 2-DOF gate on top of IoU) → **FriendFoeClassifier** (HSV outline-ring + ≥3-frame vote) → **overlay** drawn on the preview → telemetry (now carrying tracks). All new units sit behind ABCs with no-op defaults so existing tests/constructors are untouched. Tracker core is numpy+scipy only (no torch/opencv/weights) → unit tests stay GPU/display/weight-free.

**Tech Stack:** numpy, scipy (`linear_sum_assignment`, `linalg`), OpenCV (classifier/overlay only), PySide6 (unchanged), pytest.

## Global Constraints

- **Python 3.11+**; tests run headless/CI-safe via fakes (GUI tests under `QT_QPA_PLATFORM=offscreen`).
- **Do NOT pip-install `boxmot`** — env-hostile pins (numpy==1.26.4, torchvision<0.18) break Python 3.13. Vendor the motion core instead.
- **Vendor provenance:** base the vendored tracker on the **MIT-licensed** originals — `NirAharon/BoT-SORT` (`tracker/{bot_sort,byte_tracker,kalman_filter,matching,basetrack,gmc}.py`) and/or `FoundationVision/ByteTrack` — NOT AGPL `boxmot`. Add a `tracking/_vendor/VENDOR.md` noting source repo + commit/tag + the local modifications (ReID stripped, CMC→injected affine, lap→scipy, Mahalanobis gate added). Vendored core imports **numpy + scipy only**.
- **Linear assignment:** `scipy.optimize.linear_sum_assignment` (mask cost > thresh to +inf, solve, prune matches with cost > thresh). Add `scipy>=1.10` to `pyproject.toml` deps.
- **Mahalanobis gate constant:** `CHI2_GATE_2DOF = 5.9915` (chi-square 0.95 quantile, 2 DOF; gate on (x,y) center, `only_position=True`).
- **Ego-motion:** `IDENTITY_AFFINE = np.array([[1,0,0],[0,1,0]], np.float32)`; injected via an `EgoMotion.estimate(frame)->2x3` provider so Phase 3/4 feed-forward GMC is a drop-in.
- **Backward compatibility:** `TelemetrySnapshot` gains `tracks: tuple[Track, ...] = ()` appended LAST with a default; `WorkerLoop` gains `tracker`/`classifier` keyword args defaulting to no-ops. Phase 1's `MainWindow` and `WorkerLoop(cap, det, profiler, pub)` must keep working unchanged.
- **Friend/foe safety default:** insufficient-vote tracks are `Team.UNKNOWN` (never auto-target). Colorblind-aware colors throughout (no red/green pairing).
- TDD, DRY, YAGNI, one commit per task.

---

## File Structure (created/modified this phase)

```
src/ragnarok/core/types.py             # MODIFY: add Team, Track, Tracks
src/ragnarok/tracking/__init__.py      # new
src/ragnarok/tracking/_vendor/         # new: vendored MIT motion core (numpy+scipy only)
  __init__.py, VENDOR.md, basetrack.py, kalman_xywh.py, ops.py, iou.py, strack.py, botsort_core.py
src/ragnarok/tracking/associate.py     # new: scipy linear_assignment
src/ragnarok/tracking/gate.py          # new: mahalanobis_gate + CHI2_GATE_2DOF
src/ragnarok/tracking/egomotion.py     # new: EgoMotion ABC + IdentityEgoMotion
src/ragnarok/tracking/base.py          # new: Tracker ABC + IdentityTracker + IDENTITY_AFFINE
src/ragnarok/tracking/botsort.py       # new: BotSortTracker wrapper (Tracker impl)
src/ragnarok/classification/__init__.py# new
src/ragnarok/classification/color.py   # new: HSVBand, ColorProfile, ring_mask, color_match_fraction, palettes
src/ragnarok/classification/votes.py   # new: TrackVoteBook
src/ragnarok/classification/base.py    # new: FriendFoeClassifier ABC + NullClassifier + HSVRingClassifier
src/ragnarok/telemetry/snapshot.py     # MODIFY: append tracks=()
src/ragnarok/gui/overlay.py            # new: draw_overlay(image, tracks, scale) + TEAM_BGR
src/ragnarok/worker/loop.py            # MODIFY: detect->track->classify->overlay->publish
pyproject.toml                         # MODIFY: add scipy>=1.10
tests/...                              # mirrors
```

---

### Task 1: Core types — Team, Track, Tracks

**Files:** Modify `src/ragnarok/core/types.py`; Test `tests/core/test_track_types.py`.

**Interfaces — Produces:** `Team(str, Enum){UNKNOWN,ENEMY,TEAMMATE}`; frozen `Track(track_id:int, xyxy, confidence:float, class_id:int, team=Team.UNKNOWN, age=0, hits=0, time_since_update=0)` with `.center` and `Track.from_detection(det, track_id, *, team=..., age=0, hits=1, time_since_update=0)`; frozen `Tracks(items=())` with `__len__/__iter__/empty()`.

- [ ] **Step 1: Failing test** — assert `Team.ENEMY.value=="enemy"`; `Track.from_detection(Detection((0,0,10,20),0.9,0), 5).center==(5.0,10.0)` and `.track_id==5`, `.team==Team.UNKNOWN`; `Tracks.empty()` len 0; `Tracks((track,))` len 1.
- [ ] **Step 2: Run, see fail** (`ImportError: cannot import name 'Track'`).
- [ ] **Step 3: Implement** (append to `core/types.py`):

```python
from enum import Enum

class Team(str, Enum):
    UNKNOWN = "unknown"
    ENEMY = "enemy"
    TEAMMATE = "teammate"

@dataclass(frozen=True)
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    team: Team = Team.UNKNOWN
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def from_detection(cls, det: "Detection", track_id: int, *, team: "Team" = Team.UNKNOWN,
                       age: int = 0, hits: int = 1, time_since_update: int = 0) -> "Track":
        return cls(track_id=track_id, xyxy=det.xyxy, confidence=det.confidence,
                   class_id=det.class_id, team=team, age=age, hits=hits,
                   time_since_update=time_since_update)

@dataclass(frozen=True)
class Tracks:
    items: tuple[Track, ...] = ()
    def __len__(self) -> int: return len(self.items)
    def __iter__(self): return iter(self.items)
    @classmethod
    def empty(cls) -> "Tracks": return cls(items=())
```

- [ ] **Step 4: Run, see pass.** **Step 5: Commit** `feat(core): add Team/Track/Tracks types`.

---

### Task 2: Vendored motion-only tracker + gate + associate + wrapper

This is the largest task; it is one cohesive deliverable (Kalman + matching + STrack lifecycle + wrapper must work together). **Fetch the real MIT source** (`NirAharon/BoT-SORT` `tracker/` and/or `FoundationVision/ByteTrack`) and adapt — do not hand-write the Kalman/Hungarian from memory.

**Files:** Create `tracking/__init__.py`, `tracking/_vendor/*` (numpy+scipy only), `tracking/associate.py`, `tracking/gate.py`, `tracking/egomotion.py`, `tracking/base.py`, `tracking/botsort.py`; Tests `tests/tracking/test_gate.py`, `test_associate.py`, `test_botsort.py`.

**Interfaces — Produces:** `IDENTITY_AFFINE`; `EgoMotion` ABC + `IdentityEgoMotion.estimate(frame)->np.ndarray(2x3)`; `Tracker` ABC `update(detections: Detections, ego_affine=IDENTITY_AFFINE) -> Tracks`; `IdentityTracker` (no-op echo, for defaults/tests); `BotSortTracker(config..., ego: EgoMotion=IdentityEgoMotion())`; `CHI2_GATE_2DOF=5.9915`; `mahalanobis_gate(cost, track_mean_xy, track_S_xy, det_xy, thresh=CHI2_GATE_2DOF)`; `linear_assignment(cost, thresh) -> (matches Nx2, unmatched_rows, unmatched_cols)`.

**Adaptation rules (from prep):**
- Vendor: a `KalmanFilterXYWH` (predict/multi_predict/project/update/gating_distance + `chi2inv95`), `BaseTrack`/`TrackState`, box `ops` (xywh/xyxy/tlwh conversions), pure-numpy `iou_batch`, `STrack` (state = mean(8)/cov(8x8) xywh; shared Kalman classmethod), and the `BoTSORT`/ByteTrack two-stage `update`. **Strip** all ReID/appearance branches and the optical-flow CMC.
- `STrack.multi_gmc(stracks, H)` keep verbatim (the injection seam): `R8x8=np.kron(np.eye(4),H[:2,:2]); mean=R8x8@mean; mean[:2]+=H[:2,2]; cov=R8x8@cov@R8x8.T`. In `update`, replace `self.cmc.apply(...)` with `H = self.ego.estimate(frame)`.
- Replace `lap.lapjv` with `tracking/associate.linear_assignment` (scipy; mask>thresh→+inf, solve, prune).
- Add the Mahalanobis gate on the IoU cost before association via `tracking/gate.mahalanobis_gate` (compute `project()`→(mean_xy, S_xy), gate dets with d²>5.9915).
- The wrapper maps our `Detections` → the tracker's input array `[x1,y1,x2,y2,conf,cls]`, calls the core, and emits our `Tracks` (confirmed tracks → `Track(track_id, xyxy, confidence, class_id, age, hits, time_since_update)`, `team=UNKNOWN`).

- [ ] **Step 1 (gate, pure):** failing test for `mahalanobis_gate`: hand-built `track_mean_xy`, `track_S_xy` (e.g. identity*σ²), `det_xy`; assert a det at d²<5.9915 keeps finite cost and a far det is set to +inf. **Implement** `gate.py`:

```python
import numpy as np
CHI2_GATE_2DOF = 5.9915  # chi-square 0.95 quantile, 2 DOF (== chi2inv95[2])

def mahalanobis_gate(cost, track_mean_xy, track_S_xy, det_xy, thresh=CHI2_GATE_2DOF, gated=np.inf):
    for t in range(cost.shape[0]):
        L = np.linalg.cholesky(track_S_xy[t])
        d = (det_xy - track_mean_xy[t]).T
        z = np.linalg.solve(L, d)
        d2 = np.einsum('ij,ij->j', z, z)
        cost[t, d2 > thresh] = gated
    return cost
```
Run → pass. (Commit folded into this task's final commit.)

- [ ] **Step 2 (associate, pure):** failing test: a 2×2 cost with one clear match pair + a >thresh entry; assert correct matches/unmatched. **Implement** `associate.py`:

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost, thresh):
    if cost.size == 0:
        return (np.empty((0, 2), int), tuple(range(cost.shape[0])), tuple(range(cost.shape[1])))
    c = cost.copy(); c[c > thresh] = thresh + 1e-5
    rows, cols = linear_sum_assignment(c)
    matches, ur, uc = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, k in zip(rows, cols):
        if cost[r, k] <= thresh:
            matches.append((r, k)); ur.discard(r); uc.discard(k)
    return (np.array(matches).reshape(-1, 2), tuple(sorted(ur)), tuple(sorted(uc)))
```
Run → pass.

- [ ] **Step 3 (egomotion + base):** implement `egomotion.py` (`EgoMotion` ABC, `IdentityEgoMotion.estimate`→`np.eye(2,3,dtype=np.float32)`) and `base.py` (`IDENTITY_AFFINE`, `Tracker` ABC, `IdentityTracker` echoing detections→tracks with incrementing ids). Test `IdentityTracker.update(Detections((d,)))` returns one Track with matching xyxy.

- [ ] **Step 4 (vendor + wrapper):** fetch + vendor the MIT motion core into `_vendor/` (numpy+scipy only; add `VENDOR.md`), wire `lap→associate`, `cmc→ego`, add the gate in the association step, and implement `BotSortTracker(Tracker)` in `botsort.py`. **Failing test** `test_botsort.py` (CI-safe, synthetic): feed two frames of `Detections` for a box moving slightly; assert the same `track_id` persists across frames (ID stability) and that a confirmed track is emitted after the configured min hits. A second test: two well-separated detections get two distinct ids.

- [ ] **Step 5:** run `pytest tests/tracking -v` → all pass. Confirm no `torch`/`cv2`/`lap` import in `tracking/_vendor` or `tracking/*` (grep). **Commit** `feat(tracking): vendor motion-only BoT-SORT with scipy assignment + Mahalanobis gate + injectable ego-motion`.

---

### Task 3: Friend/foe HSV classifier

**Files:** Create `classification/__init__.py`, `classification/color.py`, `classification/votes.py`, `classification/base.py`; Tests `tests/classification/test_color.py`, `test_votes.py`, `test_classifier.py`.

**Interfaces — Produces:** `HSVBand`, `ColorProfile`, `DEFAULT_ENEMY_PROFILES{'red','purple','yellow'}`, `WONG_PROFILES{...}`, `ring_mask(shape_hw, xyxy, thickness=4)`, `color_match_fraction(img_bgr, xyxy, profile, thickness=4, open_ksize=3)`, `is_enemy_frame(img_bgr, xyxy, profile, frac_threshold=0.18, **kw)`; `TrackVoteBook(window=5, min_agree=3)` with `update(track_id, bool)->bool`, `label(track_id)->str`, `prune(live_ids)`; `FriendFoeClassifier` ABC, `NullClassifier`, `HSVRingClassifier(profile, vote_window=5, vote_min=3).classify(tracks, frame)->Tracks`.

Use the concrete code from the prep findings (color.py: `HSVBand`/`ColorProfile`/`ring_mask`/`color_match_fraction`/`is_enemy_frame`; the vivid `RED`(two-range wraparound)/`PURPLE`/`YELLOW` and `WONG_PROFILES` band tuples; votes.py: `TrackVoteBook`). `HSVRingClassifier.classify` runs `is_enemy_frame` per track → `TrackVoteBook.update` → `dataclasses.replace(track, team=Team.ENEMY if voted else Team.UNKNOWN)`; prune the votebook to live track ids each call.

- [ ] **Step 1 (color, TDD, synthetic images):** failing tests — a BGR-yellow ring on black classifies enemy with `YELLOW`; a desaturated grey box does NOT (S/V floor); red ring matches via the two-range wraparound. **Implement** `color.py`. Run → pass.
- [ ] **Step 2 (votes):** failing test — `TrackVoteBook(min_agree=3)`: not enemy after 2 Trues, enemy on the 3rd; `prune` drops dead ids. **Implement** `votes.py`. Run → pass.
- [ ] **Step 3 (classifier):** failing test — `HSVRingClassifier` over 3 frames of an enemy-colored track flips its `team` to `ENEMY` only on frame 3; a non-matching track stays `UNKNOWN`; `NullClassifier` returns tracks unchanged. **Implement** `base.py`. Run → pass.
- [ ] **Step 4: Commit** `feat(classification): add HSV outline friend/foe classifier with temporal vote`.

---

### Task 4: Telemetry snapshot extension + preview overlay

**Files:** Modify `telemetry/snapshot.py`; Create `gui/overlay.py`; Tests `tests/gui/test_overlay.py`, extend `tests/telemetry/test_snapshot.py`.

**Interfaces — Produces:** `TelemetrySnapshot(..., tracks: tuple[Track, ...] = ())` (append-only); `gui/overlay.draw_overlay(image, tracks, scale)->image` + `TEAM_BGR{ENEMY:(0,140,255), TEAMMATE:(255,128,0), UNKNOWN:(160,160,160)}` (colorblind-aware orange/blue/gray, BGR).

- [ ] **Step 1:** failing test — `TelemetrySnapshot(...legacy fields...)` still constructs (no `tracks`) and `.tracks==()`; constructing with `tracks=(track,)` carries it. **Implement** the append-only field. Run → pass. (Confirms Phase 1 `MainWindow`/snapshot consumers unaffected.)
- [ ] **Step 2:** failing test — `draw_overlay` on a small image with one `Track(team=ENEMY, xyxy=(4,4,20,20))`, scale 1.0, paints non-background pixels in `TEAM_BGR[ENEMY]` near the box edge; team color mapping correct. **Implement** `overlay.py`:

```python
import cv2
from ragnarok.core.types import Team
TEAM_BGR = {Team.ENEMY: (0, 140, 255), Team.TEAMMATE: (255, 128, 0), Team.UNKNOWN: (160, 160, 160)}

def draw_overlay(image, tracks, scale: float):
    for tr in tracks:
        x1, y1, x2, y2 = (int(v * scale) for v in tr.xyxy)
        col = TEAM_BGR[tr.team]
        cv2.rectangle(image, (x1, y1), (x2, y2), col, 1)
        cv2.putText(image, f"#{tr.track_id} {tr.team.value}", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    return image
```
Run → pass. **Step 3: Commit** `feat(telemetry,gui): carry tracks in snapshot and draw preview overlay`.

---

### Task 5: Worker integration

**Files:** Modify `worker/loop.py`, `pyproject.toml` (add `scipy>=1.10`); Modify/extend `tests/worker/test_loop.py`.

**Interfaces — Consumes** everything above. **Produces:** `WorkerLoop(capturer, detector, profiler, publisher, *, preview_max=320, tracker=None, classifier=None)` — defaults `IdentityTracker()` / `NullClassifier()`. `tick()`: detect → `tracker.update(dets, IDENTITY_AFFINE)` → `classifier.classify(tracks, frame)` → `draw_overlay(preview, tracks, scale)` → publish snapshot with `tracks=tuple(tracks)`. Adds profiler stages `"track"`, `"classify"`.

- [ ] **Step 1:** failing test — with a fake capturer+detector (one detection) and default tracker/classifier, `tick()` publishes a snapshot whose `tracks` has length ≥1 and whose `detection_count` is unchanged; profiler now has `"track"` and `"classify"` stages. Add a test injecting a fake tracker that returns a known `Track` and asserting it appears in the snapshot + preview is drawn. Confirm the **existing** Phase-1 worker tests still pass unchanged.
- [ ] **Step 2:** add `scipy>=1.10` to `pyproject.toml` deps; `pip install -e ".[dev]"`.
- [ ] **Step 3:** implement the `tick()` changes (from prep code). Run `pytest tests/worker -v` → pass.
- [ ] **Step 4:** run the FULL suite `QT_QPA_PLATFORM=offscreen python -m pytest -q` → all green (Phase 1 + Phase 2). **Commit** `feat(worker): wire tracking + friend/foe + overlay into the loop`.

---

## Self-Review

- **Spec coverage (§17.2):** tracking (Task 2) ✓, friend/foe HSV + vote (Task 3) ✓, detection overlay (Task 4) ✓, wired into loop (Task 5) ✓, Track type (Task 1) ✓. Mahalanobis gate + injectable ego-motion seam present for Phase 3/4 ✓. Full transparent in-game overlay + IMM/world-space are correctly deferred (Phase 3/4/8).
- **Placeholders:** none — concrete code or explicit fetch-and-adapt instructions with named source files for every step.
- **Type consistency:** `Track`/`Tracks`/`Team`, `Tracker.update(detections, ego_affine)->Tracks`, `FriendFoeClassifier.classify(tracks, frame)->Tracks`, `draw_overlay(image, tracks, scale)`, `TelemetrySnapshot(..., tracks=())`, `WorkerLoop(..., tracker=None, classifier=None)` — used in Tasks 4–5 exactly as defined in Tasks 1–3.
- **CI-safe:** tracker = numpy+scipy, classifier/overlay = numpy+cv2, all unit-testable with synthetic data; no torch/GPU/display/weights in the Phase 2 test path.
