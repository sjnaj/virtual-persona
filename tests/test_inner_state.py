# tests/test_inner_state.py
import json
import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path


def test_inner_monologue_round_trips():
    from inner_state import InnerMonologue
    m = InnerMonologue(text="心里有点空", mood_tint=0.1, mood_reason="今天累了",
                       generated_at="2026-03-22T10:00:00")
    assert InnerMonologue.from_dict(m.to_dict()) == m


def test_pending_thought_round_trips():
    from inner_state import PendingThought
    t = PendingThought(content="想问小明", target_user_id=123, urgency=0.6,
                       source="conversation",
                       created_at="2026-03-22T10:00:00",
                       expires_at="2026-03-24T10:00:00")
    assert PendingThought.from_dict(t.to_dict()) == t


def test_pending_thought_is_expired():
    from inner_state import PendingThought
    past = (datetime.now() - timedelta(hours=1)).isoformat()
    t = PendingThought(content="x", target_user_id=0, urgency=0.5, source="life_event",
                       created_at=past, expires_at=past)
    assert t.is_expired() is True


def test_pending_thought_not_expired():
    from inner_state import PendingThought
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    t = PendingThought(content="x", target_user_id=0, urgency=0.5, source="life_event",
                       created_at=datetime.now().isoformat(), expires_at=future)
    assert t.is_expired() is False


def test_inner_state_manager_save_load(tmp_path):
    from inner_state import InnerStateManager, InnerMonologue, PendingThought
    from unittest.mock import MagicMock
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "inner_state.json"),
    )
    mgr.yesterday_final_valence = 0.3
    mgr.current_monologue = InnerMonologue(
        text="测试独白", mood_tint=0.2, mood_reason="测试原因",
        generated_at="2026-03-22T10:00:00"
    )
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    mgr.pending_thoughts = [
        PendingThought(content="待办", target_user_id=1, urgency=0.5,
                       source="conversation",
                       created_at="2026-03-22T10:00:00",
                       expires_at=future)
    ]
    mgr._save()

    mgr2 = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "inner_state.json"),
    )
    assert mgr2.yesterday_final_valence == 0.3
    assert mgr2.current_monologue.text == "测试独白"
    assert len(mgr2.pending_thoughts) == 1
    assert mgr2.pending_thoughts[0].content == "待办"


def test_inner_state_manager_load_missing_file(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "nonexistent.json"),
    )
    assert mgr.yesterday_final_valence == 0.0
    assert mgr.current_monologue is None
    assert mgr.pending_thoughts == []


def test_on_sleep_snapshots_valence(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock
    from event_bus import Event
    bus = MagicMock()
    bus.subscribe = MagicMock()
    emotion = MagicMock()
    emotion.state.valence = 0.42
    life_sim = MagicMock()
    life_sim.physical.sleep_debt = 0
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    asyncio.run(mgr._on_sleep(Event("life.sleeping", {})))
    assert mgr.yesterday_final_valence == pytest.approx(0.42)
    # verify persisted by loading with a fresh instance
    bus2 = MagicMock()
    bus2.subscribe = MagicMock()
    mgr2 = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus2,
        save_path=str(tmp_path / "s.json"),
    )
    assert mgr2.yesterday_final_valence == pytest.approx(0.42)


def test_on_wake_up_calls_set_daily_baseline(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock, call
    bus = MagicMock()
    bus.subscribe = MagicMock()
    emotion = MagicMock()
    emotion.state.valence = 0.3
    life_sim = MagicMock()
    life_sim.physical.sleep_debt = 0
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    mgr.yesterday_final_valence = 0.5
    asyncio.run(mgr.on_wake_up())
    assert emotion.set_daily_baseline.called
    called_val = emotion.set_daily_baseline.call_args[0][0]
    assert -1.0 <= called_val <= 1.0


def test_on_memory_consolidated_buffers_convos(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock
    from event_bus import Event
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    for i in range(4):
        asyncio.run(mgr._on_memory_consolidated(
            Event("memory.consolidated", {"chat_id": i, "convo_text": f"对话{i}"})
        ))
    assert len(mgr._recent_convos) == 3
    assert mgr._recent_convos[0] == "对话1"   # oldest dropped


def test_on_follow_up_fired_removes_thought(tmp_path):
    from inner_state import InnerStateManager, PendingThought
    from unittest.mock import MagicMock
    from event_bus import Event
    from datetime import timedelta
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    mgr.pending_thoughts = [
        PendingThought(content="想问小明", target_user_id=1, urgency=0.5,
                       source="conversation",
                       created_at=datetime.now().isoformat(),
                       expires_at=future),
        PendingThought(content="其他事情", target_user_id=2, urgency=0.3,
                       source="conversation",
                       created_at=datetime.now().isoformat(),
                       expires_at=future),
    ]
    asyncio.run(mgr._on_follow_up_fired(
        Event("proactive.follow_up_fired", {"thought_content": "想问小明"})
    ))
    assert len(mgr.pending_thoughts) == 1
    assert mgr.pending_thoughts[0].content == "其他事情"


def test_generate_monologue_skips_when_sleeping(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    life_sim = MagicMock()
    life_sim.is_sleeping = True
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=MagicMock(),
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    asyncio.run(mgr.generate_monologue())
    llm.call_json.assert_not_called()


def test_generate_monologue_on_llm_failure_preserves_previous(tmp_path):
    from inner_state import InnerStateManager, InnerMonologue
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    prev = InnerMonologue(text="上次的独白", mood_tint=0.1,
                          mood_reason="测试", generated_at="2026-01-01T00:00:00")
    mgr.current_monologue = prev
    asyncio.run(mgr.generate_monologue())
    assert mgr.current_monologue.text == "上次的独白"


def test_generate_monologue_updates_state(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "monologue": "今天有点累，但还好",
        "mood_tint": -0.1,
        "mood_reason": "没睡好",
        "new_pending_thoughts": [
            {"content": "想问张三项目进度", "target_user_id": 999, "urgency": 0.7, "source": "conversation"}
        ],
        "resolved_thoughts": [],
    })
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 60}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": -0.1, "arousal": 0.0}
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    emitted_events = []
    async def capture(event):
        emitted_events.append(event)
    bus.subscribe("inner_state.updated", capture)
    asyncio.run(mgr.generate_monologue())
    assert mgr.current_monologue.text == "今天有点累，但还好"
    assert len(mgr.pending_thoughts) == 1
    assert mgr.pending_thoughts[0].content == "想问张三项目进度"
    assert mgr.pending_thoughts[0].target_user_id == 999
    # verify inner_state.updated was emitted with the thought in payload
    assert len(emitted_events) == 1
    payload_thoughts = emitted_events[0].data["pending_thoughts"]
    assert any(t["content"] == "想问张三项目进度" for t in payload_thoughts)


def test_generate_monologue_removes_resolved_thoughts(tmp_path):
    from inner_state import InnerStateManager, PendingThought
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    from datetime import timedelta
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "monologue": "清爽",
        "mood_tint": 0.2,
        "mood_reason": "处理完了",
        "new_pending_thoughts": [],
        "resolved_thoughts": ["想问张三项目进度"],
    })
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": 0.2, "arousal": 0.1}
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    mgr.pending_thoughts = [
        PendingThought(content="想问张三项目进度", target_user_id=999, urgency=0.7,
                       source="conversation",
                       created_at=datetime.now().isoformat(), expires_at=future)
    ]
    asyncio.run(mgr.generate_monologue())
    assert len(mgr.pending_thoughts) == 0


def test_generate_monologue_cleans_expired_thoughts(tmp_path):
    from inner_state import InnerStateManager, PendingThought
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "monologue": "ok", "mood_tint": 0.0, "mood_reason": "",
        "new_pending_thoughts": [], "resolved_thoughts": [],
    })
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": 0.0, "arousal": 0.0}
    past = (datetime.now() - timedelta(hours=1)).isoformat()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    mgr.pending_thoughts = [
        PendingThought(content="已过期", target_user_id=1, urgency=0.5,
                       source="conversation", created_at=past, expires_at=past)
    ]
    asyncio.run(mgr.generate_monologue())
    assert len(mgr.pending_thoughts) == 0
