# tests/test_emotion_baseline.py
from unittest.mock import MagicMock


def make_engine():
    from emotion_engine import EmotionEngine
    bus = MagicMock()
    bus.subscribe = MagicMock()
    persona = {"personality": {"neuroticism": 0.5}}
    return EmotionEngine(persona, bus)


def test_daily_baseline_defaults_to_zero():
    engine = make_engine()
    assert engine.daily_baseline == 0.0


def test_set_daily_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    assert engine.daily_baseline == 0.3


def test_passive_decay_valence_converges_toward_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.valence = 0.9
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.valence - 0.3) < 0.05


def test_passive_decay_valence_rises_toward_baseline_when_below():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.valence = -0.5
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.valence - 0.3) < 0.05


def test_passive_decay_arousal_decays_to_zero_regardless_of_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.arousal = 0.8
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.arousal) < 0.05


def test_passive_decay_baseline_zero_preserves_original_behaviour():
    """With baseline=0 (default), valence still decays toward zero."""
    engine = make_engine()
    engine.state.valence = 0.8
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.valence) < 0.05


def test_passive_decay_irritability_decays_to_zero_regardless_of_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.irritability = 0.9
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.irritability) < 0.05
