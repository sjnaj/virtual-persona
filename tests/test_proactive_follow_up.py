# tests/test_proactive_follow_up.py
import asyncio
from unittest.mock import MagicMock
from event_bus import EventBus, Event
from datetime import datetime, timedelta


def test_inner_state_updated_adds_follow_up_triggers():
    from proactive_engine import ProactiveEngine
    bus = EventBus()
    persona = {"name": "测试", "personality": {"extraversion": 0.7}}
    engine = ProactiveEngine(persona, bus)

    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    event = Event("inner_state.updated", {
        "pending_thoughts": [
            {"content": "想问小明项目进度", "target_user_id": 999,
             "urgency": 0.7, "source": "conversation",
             "created_at": now_str, "expires_at": future},
        ]
    })
    asyncio.run(engine._on_inner_state_updated(event))

    assert any(
        t.get("type") == "follow_up" and t.get("content") == "想问小明项目进度"
        for t in engine.pending_triggers
    )


def test_inner_state_updated_ignores_zero_target():
    from proactive_engine import ProactiveEngine
    bus = EventBus()
    persona = {"name": "测试", "personality": {"extraversion": 0.7}}
    engine = ProactiveEngine(persona, bus)

    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    event = Event("inner_state.updated", {
        "pending_thoughts": [
            {"content": "随便想想", "target_user_id": 0,
             "urgency": 0.5, "source": "life_event",
             "created_at": now_str, "expires_at": future},
        ]
    })
    asyncio.run(engine._on_inner_state_updated(event))
    assert not any(t.get("type") == "follow_up" for t in engine.pending_triggers)


def test_inner_state_updated_deduplicates_triggers():
    from proactive_engine import ProactiveEngine
    bus = EventBus()
    persona = {"name": "测试", "personality": {"extraversion": 0.7}}
    engine = ProactiveEngine(persona, bus)

    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    thought = {"content": "想问小明", "target_user_id": 1,
               "urgency": 0.6, "source": "conversation",
               "created_at": now_str, "expires_at": future}
    event = Event("inner_state.updated", {"pending_thoughts": [thought]})
    asyncio.run(engine._on_inner_state_updated(event))
    asyncio.run(engine._on_inner_state_updated(event))  # same event twice
    follow_ups = [t for t in engine.pending_triggers if t.get("type") == "follow_up"]
    assert len(follow_ups) == 1   # not duplicated
