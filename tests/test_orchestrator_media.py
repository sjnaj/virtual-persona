# tests/test_orchestrator_media.py
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


_TINY_JPEG = [{"mime_type": "image/jpeg", "base64": "abc123", "label": "photo"}]
_TINY_STICKER = [{"mime_type": "image/webp", "base64": "xyz789", "label": "sticker"}]


def _make_orchestrator():
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
        orch.relationships.get_social_context_for_prompt = MagicMock(return_value="")
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
            {"type": "text", "content": "好漂亮～", "delay": 1}
        ])

        orch.inner_state = MagicMock()
        orch.proactive_callbacks = {}
        return orch


def test_media_passed_to_expression():
    """handle_message forwards media list to expression.compose_reply."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="看这张图", user_id=1, user_name="小明",
        chat_id=1, chat_type="private", media=_TINY_JPEG,
    ))

    call_kwargs = orch.expression.compose_reply.call_args
    passed_media = call_kwargs.kwargs.get("media") if call_kwargs.kwargs else None
    if passed_media is None and call_kwargs.args:
        # positional fallback shouldn't happen, but just in case
        passed_media = None
    assert passed_media == _TINY_JPEG


def test_photo_label_appended_to_memory():
    """Photo media label '[图片]' is appended to stored message text."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="给你看", user_id=1, user_name="小明",
        chat_id=1, chat_type="private", media=_TINY_JPEG,
    ))

    stored_texts = []
    for call in orch.memory.add_message.call_args_list:
        if len(call.args) >= 3:
            stored_texts.append(call.args[2])
        elif call.kwargs.get("content"):
            stored_texts.append(call.kwargs["content"])

    assert any("[图片]" in t for t in stored_texts), f"No [图片] in stored texts: {stored_texts}"


def test_sticker_label_appended():
    """Static sticker label '[表情包]' is appended to stored message text."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="", user_id=1, user_name="小明",
        chat_id=1, chat_type="private", media=_TINY_STICKER,
    ))

    stored_texts = []
    for call in orch.memory.add_message.call_args_list:
        if len(call.args) >= 3:
            stored_texts.append(call.args[2])

    assert any("[表情包]" in t for t in stored_texts), f"No [表情包] in: {stored_texts}"


def test_no_media_passes_none():
    """handle_message with no media passes media=None to expression."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="你好", user_id=1, user_name="小明",
        chat_id=1, chat_type="private",
    ))

    call_kwargs = orch.expression.compose_reply.call_args
    passed_media = call_kwargs.kwargs.get("media") if call_kwargs.kwargs else "NOT_FOUND"
    assert passed_media is None
