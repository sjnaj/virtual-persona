# tests/test_orchestrator_follow_up.py
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


def make_orchestrator(tmp_path):
    import yaml
    from orchestrator import Orchestrator
    cfg = yaml.safe_load(open("/Users/admin/Work/virtual-persona/config.yaml"))
    cfg["llm"]["expression"]["api_key"] = "test"
    cfg["llm"]["utility"]["api_key"] = "test"
    with patch("memory_hub.MemoryHub.__init__", lambda self, *a, **kw: None), \
         patch("vector_store.PersistentClient"):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = cfg
        orch.persona = cfg["persona"]
        orch.running = False
        from event_bus import EventBus
        orch.bus = EventBus()
        orch.inner_state = MagicMock()
        orch.proactive_callbacks = {}
        orch.relationships = MagicMock()
        orch.relationships.list_all.return_value = [{"user_id": 999}]
        orch.expression = MagicMock()
        orch.expression.compose_proactive = AsyncMock(return_value=[
            {"type": "text", "content": "嗯嗯", "delay": 1}
        ])
        orch.memory = MagicMock()
        orch.memory.add_message = MagicMock()
        orch.memory.get_status = MagicMock(return_value={})
        orch.emotion = MagicMock()
        orch.emotion.get_status = MagicMock(return_value={})
        orch.life_sim = MagicMock()
        orch.life_sim.get_status = MagicMock(return_value={})
        return orch


def test_follow_up_fired_emitted_on_success(tmp_path):
    orch = make_orchestrator(tmp_path)
    from datetime import datetime
    follow_up_trigger = {
        "type": "follow_up",
        "urgency": 0.8,
        "content": "想问小明进度",
        "target_user_id": 999,
        "source": "inner_state",
        "time": datetime.now(),
    }
    # patch evaluate() to return our trigger directly — avoids time-of-day guard
    orch.proactive = MagicMock()
    orch.proactive.evaluate.return_value = follow_up_trigger

    messages_sent = []
    async def fake_callback(msgs):
        messages_sent.extend(msgs)
    orch.proactive_callbacks[999] = fake_callback

    emitted = []
    async def capture(event):
        emitted.append(event)
    orch.bus.subscribe("proactive.follow_up_fired", capture)

    asyncio.run(orch.handle_proactive_trigger())

    assert len(messages_sent) > 0
    assert any(e.data["thought_content"] == "想问小明进度" for e in emitted)


def test_follow_up_skipped_when_target_not_registered(tmp_path):
    orch = make_orchestrator(tmp_path)
    from datetime import datetime
    orch.proactive = MagicMock()
    orch.proactive.evaluate.return_value = {
        "type": "follow_up",
        "urgency": 0.8,
        "content": "想问不在回调里的人",
        "target_user_id": 12345,   # not in proactive_callbacks
        "source": "inner_state",
        "time": datetime.now(),
    }

    emitted = []
    async def capture(event):
        emitted.append(event)
    orch.bus.subscribe("proactive.follow_up_fired", capture)

    asyncio.run(orch.handle_proactive_trigger())
    assert len(emitted) == 0   # thought not fired, not deleted
