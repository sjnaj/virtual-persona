# tests/test_orchestrator_queue.py
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call


def _make_orchestrator():
    """Minimal mocked Orchestrator — same pattern as test_orchestrator_media.py."""
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

        orch.relationships = MagicMock()
        orch.relationships.get_or_create = MagicMock(return_value=MagicMock())
        orch.relationships.update_after_conversation = AsyncMock()

        orch.chat_ctx = MagicMock()
        orch.chat_ctx.get_or_create = MagicMock(return_value=MagicMock())
        orch.chat_ctx.add_message = MagicMock()
        orch.chat_ctx.get_recent_context = MagicMock(return_value="")

        orch.memory = MagicMock()
        orch.memory.add_message = MagicMock()

        orch.emotion = MagicMock()
        orch.emotion.get_status = MagicMock(return_value={})

        orch.group_behavior = MagicMock()
        orch.proactive = MagicMock()
        orch.proactive.update_last_message_time = MagicMock()

        orch.expression = MagicMock()
        orch.expression.compose_reply = AsyncMock(return_value=[
            {"type": "text", "content": "嗯嗯", "delay": 1}
        ])

        orch.inner_state = MagicMock()
        orch.proactive_callbacks = {}
        return orch


def test_ingest_updates_relationship_and_context():
    """ingest_message updates relationship, chat context, and memory."""
    orch = _make_orchestrator()

    asyncio.run(orch.ingest_message(
        text="你好", user_id=1, user_name="小明",
        chat_id=100, chat_type="private",
    ))

    orch.relationships.get_or_create.assert_called_once_with(1, "小明")
    orch.chat_ctx.add_message.assert_called_once()
    orch.memory.add_message.assert_called_once()


def test_ingest_returns_direct_for_private():
    """ingest_message returns 'direct' for private chats."""
    orch = _make_orchestrator()

    result = asyncio.run(orch.ingest_message(
        text="hello", user_id=1, user_name="小明",
        chat_id=100, chat_type="private",
    ))

    assert result == "direct"


def test_ingest_returns_none_when_group_skips():
    """ingest_message returns None when group behavior engine says don't reply."""
    orch = _make_orchestrator()
    orch.group_behavior.should_respond = AsyncMock(return_value={
        "should_reply": False, "probability": 0.1, "reply_mode": "direct",
    })

    result = asyncio.run(orch.ingest_message(
        text="random group message", user_id=1, user_name="小明",
        chat_id=200, chat_type="group",
    ))

    assert result is None


def test_ingest_does_not_call_expression():
    """ingest_message must NOT call expression.compose_reply."""
    orch = _make_orchestrator()

    asyncio.run(orch.ingest_message(
        text="你好", user_id=1, user_name="小明",
        chat_id=100, chat_type="private",
    ))

    orch.expression.compose_reply.assert_not_called()


def test_generate_reply_calls_expression():
    """_generate_reply calls expression.compose_reply with the right args."""
    orch = _make_orchestrator()

    result = asyncio.run(orch._generate_reply(
        user_message="你好", user_id=1, chat_id=100,
        chat_type="private", reply_mode="direct",
    ))

    orch.expression.compose_reply.assert_called_once()
    kwargs = orch.expression.compose_reply.call_args.kwargs
    assert kwargs["user_message"] == "你好"
    assert kwargs["user_id"] == 1
    assert kwargs["chat_id"] == 100
    assert result == [{"type": "text", "content": "嗯嗯", "delay": 1}]


def test_generate_reply_does_not_record():
    """_generate_reply must NOT write to memory or chat_ctx."""
    orch = _make_orchestrator()

    asyncio.run(orch._generate_reply(
        user_message="你好", user_id=1, chat_id=100,
        chat_type="private", reply_mode="direct",
    ))

    # memory and chat_ctx should have zero calls (no ingest was done here)
    orch.memory.add_message.assert_not_called()
    orch.chat_ctx.add_message.assert_not_called()


def test_record_reply_writes_to_memory_and_context():
    """_record_reply stores bot messages in memory and chat_ctx."""
    orch = _make_orchestrator()
    messages = [
        {"type": "text", "content": "没事的", "delay": 1},
        {"type": "sticker", "file_id": "abc", "delay": 0},
    ]

    asyncio.run(orch._record_reply(
        messages=messages, user_id=1, chat_id=100, chat_type="private",
    ))

    # Only text messages are recorded
    assert orch.memory.add_message.call_count == 1
    assert orch.chat_ctx.add_message.call_count == 1
    orch.relationships.update_after_conversation.assert_awaited_once()


def test_handle_message_backward_compat():
    """handle_message still works end-to-end and returns messages."""
    orch = _make_orchestrator()

    result = asyncio.run(orch.handle_message(
        text="你好", user_id=1, user_name="小明",
        chat_id=100, chat_type="private",
    ))

    assert result is not None
    assert result[0]["type"] == "text"
    # expression was called once
    orch.expression.compose_reply.assert_called_once()
    # reply was recorded
    orch.memory.add_message.call_count >= 1
