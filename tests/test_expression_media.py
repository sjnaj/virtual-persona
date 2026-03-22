# tests/test_expression_media.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


_PHOTO = [{"mime_type": "image/jpeg", "base64": "abc123", "label": "photo"}]
_STICKER = [{"mime_type": "image/webp", "base64": "xyz789", "label": "sticker"}]


def _make_expression():
    import yaml
    from expression import ExpressionSynthesizer
    cfg = yaml.safe_load(open("/Users/admin/Work/virtual-persona/config.yaml"))
    persona = cfg["persona"]

    llm = MagicMock()
    llm.call = AsyncMock(return_value="纯文字回复")
    llm.call_vision = AsyncMock(return_value="看到图片啦")

    memory = MagicMock()
    memory.recall = AsyncMock(return_value={
        "certain": [], "vague": [], "feelings": [], "private_hints": []
    })

    emotion = MagicMock()
    emotion.get_expression_style = MagicMock(return_value={
        "tone": "开心", "message_length": "medium", "emoji_freq": "medium",
        "response_speed": "normal", "typo_rate": 0, "sticker_mood": "happy",
        "max_segments": 2,
    })
    emotion.get_status = MagicMock(return_value={"valence": 0.5, "arousal": 0.3, "attachment": 0.2})

    life_sim = MagicMock()
    life_sim.get_status = MagicMock(return_value={
        "current_action": "在家", "location": "上海",
        "physical": {"energy": 80, "hunger": 30},
    })

    sticker_engine = MagicMock()
    sticker_engine.should_use_sticker = AsyncMock(return_value=False)
    sticker_engine.select_sticker = AsyncMock(return_value=None)

    browser = MagicMock()
    browser.get_recent_interesting = MagicMock(return_value=[])

    rel_mgr = MagicMock()
    rel_mgr.get_social_context_for_prompt = MagicMock(return_value="")
    rel_mgr.get_group_social_map = MagicMock(return_value="")
    rel_mgr.get = MagicMock(return_value=None)

    chat_ctx = MagicMock()
    chat_ctx.get_recent_context = MagicMock(return_value="")
    chat_ctx.get_or_create = MagicMock(return_value=MagicMock(
        group_title="", group_members=[]
    ))

    expr = ExpressionSynthesizer(
        persona=persona, llm=llm, memory=memory, emotion=emotion,
        life_sim=life_sim, sticker_engine=sticker_engine, browser=browser,
        relationship_mgr=rel_mgr, chat_ctx_mgr=chat_ctx,
    )
    return expr


def test_call_vision_used_when_media_present():
    """compose_reply calls llm.call_vision (not llm.call) when media is provided."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="看这张", user_id=1, chat_id=1,
        chat_type="private", media=_PHOTO,
    ))

    expr.llm.call_vision.assert_called_once()
    expr.llm.call.assert_not_called()


def test_call_vision_used_when_no_media():
    """compose_reply calls llm.call_vision with empty images when no media (fallback to plain)."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="你好", user_id=1, chat_id=1, chat_type="private",
    ))

    expr.llm.call_vision.assert_called_once()
    call_args = expr.llm.call_vision.call_args
    images_arg = call_args.kwargs.get("images") if call_args.kwargs else call_args.args[1]
    assert images_arg == []


def test_system_prompt_forwarded():
    """call_vision receives system_prompt containing persona name."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="嗨", user_id=1, chat_id=1, chat_type="private", media=_PHOTO,
    ))

    call_kwargs = expr.llm.call_vision.call_args.kwargs
    sys_prompt = call_kwargs.get("system_prompt", "")
    assert "林小晴" in sys_prompt


def test_media_note_in_prompt_for_photo():
    """The prompt passed to call_vision mentions the photo label."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="帮我看看", user_id=1, chat_id=1, chat_type="private", media=_PHOTO,
    ))

    call_args = expr.llm.call_vision.call_args
    prompt_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "photo" in prompt_arg or "图片" in prompt_arg or "帮我看看" in prompt_arg


def test_sticker_label_in_prompt():
    """Sticker label appears in the prompt annotation."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="", user_id=1, chat_id=1, chat_type="private", media=_STICKER,
    ))

    call_args = expr.llm.call_vision.call_args
    prompt_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "sticker" in prompt_arg or "表情包" in prompt_arg
