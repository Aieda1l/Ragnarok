"""Target selection — pure scoring function + stateful dual-radius sticky selector.

Architecture
------------
``select_target`` is a **pure function**: given tracks, a crosshair, FOV radii,
and optional hysteresis parameters, it returns the best ENEMY ``track_id`` (or
``None``).  No state, fully deterministic — easy to unit-test.

``TargetSelector`` wraps ``select_target`` with **mutable lock state**:
dual-radius sticky FOV (acquire inner / retain outer), a **dwell timer**
(a challenger must outperform the lock for *dwell_ms* before a switch fires),
and a **switch margin** (challenger must beat the lock's cost by a fraction).
The clock is injected at construction so tests can drive time deterministically
via a ``FakeClock`` callable without sleeping.

Safety contract: ENEMY tracks only.  UNKNOWN and TEAMMATE are never selected.
"""
from __future__ import annotations

from dataclasses import dataclass

from ragnarok.core.clock import now_ns
from ragnarok.core.types import Team, Track, Tracks
from ragnarok.aim.fov import aim_point, dist_to


# ---------------------------------------------------------------------------
# Internal cost function
# ---------------------------------------------------------------------------


def _target_cost(
    tr: Track,
    crosshair: tuple[float, float],
    fov_px: float,
    head_frac: float,
    w_dist: float = 1.0,
    w_conf: float = 0.15,
) -> float:
    """Lower is better.

    Dominated by normalised distance-to-crosshair; confidence provides a
    small tie-breaker in favour of high-confidence detections.
    """
    d = dist_to(crosshair, aim_point(tr, head_frac))
    return w_dist * (d / fov_px) + w_conf * (1.0 - tr.confidence)


# ---------------------------------------------------------------------------
# Pure selection function
# ---------------------------------------------------------------------------


def select_target(
    tracks: Tracks,
    crosshair: tuple[float, float],
    fov_px: float,
    *,
    head_frac: float = 0.15,
    current_target_id: int | None = None,
    retain_fov_px: float | None = None,
    switch_margin: float = 0.0,
) -> int | None:
    """Select the best ENEMY ``track_id`` within *fov_px* of *crosshair*.

    Hysteresis (applied when *current_target_id* is provided):

    - The current target is kept if it remains within *retain_fov_px* (the
      outer, "sticky" radius), even if no better candidate is in *fov_px*.
    - A challenger must beat the current lock's cost by *switch_margin*
      (fraction, 0–1) to steal the lock.

    Ties are broken by ``track_id`` (ascending) for reproducible results.

    Parameters
    ----------
    tracks:
        All active tracks this frame.
    crosshair:
        ``(cx, cy)`` in ROI pixel coords.
    fov_px:
        Inner acquisition radius in pixels.
    head_frac:
        Head aim-point fraction (see ``fov.aim_point``).
    current_target_id:
        The currently locked ``track_id``, or ``None`` if no lock.
    retain_fov_px:
        Outer retention radius; defaults to *fov_px* if omitted.
    switch_margin:
        Fraction [0, 1) by which a challenger must beat the lock to steal.
        E.g. ``0.2`` means the challenger needs a 20% lower cost.
    """
    retain = retain_fov_px if retain_fov_px is not None else fov_px

    candidates: list[tuple[float, int]] = []   # (cost, track_id)
    cur_cost: float | None = None

    for tr in tracks:
        if tr.team is not Team.ENEMY:
            continue  # safety contract: ENEMY only
        d = dist_to(crosshair, aim_point(tr, head_frac))
        # Check if this track is the current lock and inside the retain radius
        if tr.track_id == current_target_id and d <= retain:
            cur_cost = _target_cost(tr, crosshair, fov_px, head_frac)
        # Candidate if inside the inner acquisition radius
        if d <= fov_px:
            cost = _target_cost(tr, crosshair, fov_px, head_frac)
            candidates.append((cost, tr.track_id))

    # Sort: primary by cost, secondary by track_id for stable tie-break
    candidates.sort()
    best_cost: float | None
    best_id: int | None
    if candidates:
        best_cost, best_id = candidates[0]
    else:
        best_cost, best_id = None, None

    if cur_cost is None:
        # No active lock (or lock outside retain radius) → take best in inner FOV
        return best_id

    if best_id is None or best_id == current_target_id:
        # Either no challenger in inner FOV, or current IS the best → keep it
        return current_target_id

    # Challenger must beat the current lock's cost by switch_margin to steal
    assert best_cost is not None  # implied by best_id is not None
    if best_cost < cur_cost * (1.0 - switch_margin):
        return best_id
    return current_target_id


# ---------------------------------------------------------------------------
# Stateful lock book
# ---------------------------------------------------------------------------


@dataclass
class _LockState:
    """Mutable state for one lock slot."""

    target_id: int | None = None
    challenger_id: int | None = None
    challenger_since_ns: int = 0


# ---------------------------------------------------------------------------
# Stateful TargetSelector
# ---------------------------------------------------------------------------


class TargetSelector:
    """Stateful ENEMY-only target selector.

    Combines dual-radius sticky FOV (inner acquire / outer retain),
    a dwell timer, and a switch margin for smooth, non-jittery locking.

    The clock is injected so tests can control time with a ``FakeClock``.

    Parameters
    ----------
    fov_px:
        Inner acquisition radius (px).  A new target must be within this
        cone to be locked.
    retain_fov_px:
        Outer retention radius (px).  The current lock is kept until the
        target exits this (larger) cone.
    dwell_ms:
        Minimum time (ms) a challenger must continuously outperform the
        lock before the switch is committed.
    switch_margin:
        A challenger must beat the lock's ``_target_cost`` by this
        fractional amount [0, 1) to qualify for a switch.
    head_frac:
        Head aim-point fraction passed through to ``aim_point``.
    clock:
        Zero-arg callable returning nanoseconds.  Defaults to
        ``core.clock.now_ns``.  Inject ``FakeClock()`` in tests.
    """

    def __init__(
        self,
        *,
        fov_px: float,
        retain_fov_px: float,
        dwell_ms: float = 120.0,
        switch_margin: float = 0.20,
        head_frac: float = 0.15,
        clock=now_ns,
    ) -> None:
        self._fov = fov_px
        self._retain = retain_fov_px
        self._dwell_ns = int(dwell_ms * 1_000_000)  # ms → ns
        self._margin = switch_margin
        self._head = head_frac
        self._clock = clock
        self._lk = _LockState()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def target_id(self) -> int | None:
        """Currently locked ``track_id``, or ``None`` if no lock is held."""
        return self._lk.target_id

    def reset(self) -> None:
        """Clear lock and dwell state.

        Call on aim-key re-trigger so stale locks from the previous aim
        session don't persist.
        """
        self._lk = _LockState()

    def select(self, tracks: Tracks, cx: float, cy: float) -> int | None:
        """Select the best ENEMY target this frame.

        Parameters
        ----------
        tracks:
            All active tracks from the current pipeline tick.
        cx, cy:
            Crosshair position in ROI pixel coords (typically ROI centre).

        Returns
        -------
        int | None
            The locked ``track_id``, or ``None`` if no target is selected.
        """
        crosshair = (cx, cy)
        enemy_ids = {t.track_id for t in tracks if t.team is Team.ENEMY}

        # ---- target death: drop lock if the target is gone ----
        if self._lk.target_id is not None and self._lk.target_id not in enemy_ids:
            self._lk = _LockState()

        # ---- pure selection: dual-radius stickiness + switch margin ----
        chosen = select_target(
            tracks,
            crosshair,
            self._fov,
            head_frac=self._head,
            current_target_id=self._lk.target_id,
            retain_fov_px=self._retain,
            switch_margin=self._margin,
        )

        now = self._clock()

        # ---- no current lock: acquire immediately ----
        if self._lk.target_id is None:
            self._lk = _LockState(target_id=chosen)
            return chosen

        # ---- current lock left the retention FOV (or no candidates) ----
        if chosen is None:
            self._lk = _LockState()
            return None

        # ---- same lock (or margin kept it): clear challenger state ----
        if chosen == self._lk.target_id:
            self._lk.challenger_id = None
            return self._lk.target_id

        # ---- new challenger cleared the margin: apply dwell timer ----
        if self._lk.challenger_id != chosen:
            # New challenger — start / restart dwell timer
            self._lk.challenger_id = chosen
            self._lk.challenger_since_ns = now
        elif now - self._lk.challenger_since_ns >= self._dwell_ns:
            # Challenger has held superiority for dwell_ms → commit switch
            self._lk = _LockState(target_id=chosen)

        return self._lk.target_id
