# tests/test_life_woke_up.py
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


def make_sim(hour: float):
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {"wake_up": [7, 8], "sleep": [23, 25]},
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "action": "刷手机", "location": "家", "detail": "看微博",
        "mood_impact": 0, "energy_change": -2, "notable": False, "shareable_thought": None
    })
    sim = LifeSimulator(persona, llm, bus)
    sim.is_sleeping = True
    sim.woke_up_today = False
    sim.physical.energy = 50
    return sim, bus


def test_life_woke_up_event_emitted():
    sim, bus = make_sim(7.5)
    emitted_events = []

    async def capture(event):
        emitted_events.append(event.name)

    bus.subscribe("life.woke_up", capture)

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 7, 30)
            # Force wake up: use side_effect so only the wake-check call returns 0.0
            # while the energy-calculation call returns its real value.
            # Call order in tick(): _passive_updates (2 calls), wake check (1 call).
            with patch("life_simulator.random.uniform",
                       side_effect=[0.5, 0.5, 0.0, 15.0, 2.0]):
                # side_effect list: passive energy, passive hunger, wake threshold, energy gain, sleep threshold
                await sim.tick()

    asyncio.run(run())
    assert "life.woke_up" in emitted_events


def test_life_woke_up_not_emitted_while_sleeping():
    sim, bus = make_sim(6.0)
    emitted_events = []

    async def capture(event):
        emitted_events.append(event.name)

    bus.subscribe("life.woke_up", capture)

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 6, 0)
            await sim.tick()

    asyncio.run(run())
    assert "life.woke_up" not in emitted_events
