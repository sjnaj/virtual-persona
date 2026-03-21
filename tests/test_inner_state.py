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
