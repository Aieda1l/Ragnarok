import numpy as np
from ragnarok.aim.imm import TrackIMM, IMMManager

DT = 1 / 120.0

def test_cv_tracks_position_and_velocity():
    imm = TrackIMM(100.0, 100.0, DT)
    x, y, vx = 100.0, 100.0, 300.0  # px/s in x
    for _ in range(150):
        x += vx * DT
        imm.update(x, y, DT)
    px, py = imm.position()
    vxe, vye = imm.velocity()
    assert abs(px - x) < 6.0
    assert abs(py - y) < 6.0
    assert abs(vxe - vx) < 50.0 and abs(vye) < 25.0

def test_lead_extrapolates_forward():
    imm = TrackIMM(0.0, 0.0, DT)
    x, vx = 0.0, 600.0
    for _ in range(150):
        x += vx * DT
        imm.update(x, 0.0, DT)
    pos = imm.position()
    lead = imm.lead(0.1)  # 100 ms ahead
    assert lead[0] > pos[0] + 30.0  # leads in the direction of motion

def test_juke_raises_ca_mode():
    imm = TrackIMM(0.0, 0.0, DT)
    x, y, vx = 0.0, 0.0, 300.0
    for _ in range(60):                # steady CV
        x += vx * DT
        imm.update(x, y, DT)
    cv_prob_before = imm.mode_probs[0]
    vy = 0.0
    for _ in range(30):                # lateral juke (accel)
        vy += 6000.0 * DT
        y += vy * DT
        x += vx * DT
        imm.update(x, y, DT)
    assert imm.mode_probs[1] > 0.2     # CA mode probability rose
    assert imm.mode_probs[0] < cv_prob_before

def test_manager_prune_drops_dead_ids():
    m = IMMManager()
    m.update(1, 0.0, 0.0, DT)
    m.update(2, 5.0, 5.0, DT)
    m.prune({2})
    assert 1 not in m and 2 in m

def test_manager_lead_per_track():
    m = IMMManager()
    for _ in range(30):
        m.update(7, 10.0, 10.0, DT)
    assert isinstance(m.lead(7, 0.05), tuple)
