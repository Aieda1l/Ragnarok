from ragnarok.core.types import Track, Team
from ragnarok.aim.mouse import NullMouseDriver, MouseButton
from ragnarok.trigger.bot import TriggerBot


def _track():
    return Track(track_id=1, xyxy=(100.0, 100.0, 200.0, 300.0),
                 confidence=0.9, class_id=0, team=Team.ENEMY)


class _Clock:
    def __init__(self):
        self.t = 0
    def __call__(self):
        return self.t


def _bot(delay_s=0.1):
    clk = _Clock()
    mouse = NullMouseDriver()
    mouse.connect()
    bot = TriggerBot(mouse=mouse, activation_delay_s=delay_s, clock=clk)
    return bot, mouse, clk


def _gates(**over):
    g = dict(track=_track(), crosshair=(150.0, 150.0), occluded=False,
             enemy_confirmed=True, line_clear=True, active=True)
    g.update(over)
    return g


def test_fires_after_activation_delay():
    bot, mouse, clk = _bot(delay_s=0.1)
    assert bot.update(**_gates()) is False        # t=0: eligibility starts
    clk.t = 50_000_000                            # 50 ms < 100 ms
    assert bot.update(**_gates()) is False
    clk.t = 120_000_000                           # 120 ms >= 100 ms
    assert bot.update(**_gates()) is True         # NEW press -> shot
    assert (MouseButton.LEFT, True) in mouse.buttons


def test_no_fire_when_inactive():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(active=False)) is False
    assert mouse.buttons == []


def test_no_fire_when_occluded():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(occluded=True)) is False
    assert mouse.buttons == []


def test_no_fire_when_crosshair_outside_hitbox():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(crosshair=(10.0, 10.0))) is False


def test_no_fire_when_line_blocked():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(line_clear=False)) is False


def test_releases_when_gate_drops():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates()) is True          # press
    bot.update(**_gates(active=False))             # gate drops -> release
    assert (MouseButton.LEFT, False) in mouse.buttons


def test_single_press_held_not_repeated():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates()) is True           # first press
    clk.t = 200_000_000
    assert bot.update(**_gates()) is False          # still held, no new shot


def test_no_fire_when_enemy_not_confirmed():
    bot, mouse, clk = _bot(delay_s=0.0)
    assert bot.update(**_gates(enemy_confirmed=False)) is False
    assert mouse.buttons == []


def test_timer_resets_after_gate_drop():
    bot, mouse, clk = _bot(delay_s=0.1)
    clk.t = 80_000_000                     # 80 ms — below threshold
    assert bot.update(**_gates()) is False  # eligible, not yet fired
    bot.update(**_gates(active=False))      # gate drops — timer must reset
    clk.t = 140_000_000                    # +60 ms from drop — new window not yet satisfied
    assert bot.update(**_gates()) is False  # must NOT fire
    clk.t = 250_000_000                    # +170 ms from drop — full delay satisfied
    assert bot.update(**_gates()) is True   # now fires


def test_explicit_release():
    bot, mouse, clk = _bot(delay_s=0.0)
    bot.update(**_gates())                          # press
    bot.release()                                   # explicit release
    assert (MouseButton.LEFT, False) in mouse.buttons
    clk.t = 1_000_000_000
    assert bot.update(**_gates()) is True           # fires again after release
