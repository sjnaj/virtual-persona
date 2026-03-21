# tests/test_memory_consolidated_event.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


def test_consolidate_emits_memory_consolidated(tmp_path):
    from memory_hub import MemoryHub
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "facts": [{"fact": "他喜欢猫", "about_person": "用户", "confidence": 0.9, "source_privacy": "private"}],
        "episodes": [],
        "emotional_impressions": [],
    })
    persona = {"name": "测试"}
    hub = MemoryHub(persona, llm, bus, persist_dir=str(tmp_path / "chroma"))

    # Add 5 messages to chat 1 (minimum 4 needed to consolidate)
    for i in range(5):
        hub.add_message(1, "user", f"消息{i}", user_id=100, user_name="用户", context_type="private")

    emitted = []
    async def capture(event):
        emitted.append(event)
    bus.subscribe("memory.consolidated", capture)

    asyncio.run(hub.consolidate(chat_id=1))

    assert len(emitted) == 1
    assert emitted[0].data["chat_id"] == 1
    assert "消息" in emitted[0].data["convo_text"]
    assert len(emitted[0].data["convo_text"]) <= 1500
