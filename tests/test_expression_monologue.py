# tests/test_expression_monologue.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


def make_expression(monologue_text: str = ""):
    from expression import ExpressionSynthesizer
    persona = {
        "name": "测试", "background": "测试背景",
        "speaking_style": "随意",
    }
    llm = MagicMock()
    llm.call = AsyncMock(return_value="测试回复")
    memory = MagicMock()
    memory.recall = AsyncMock(return_value={
        "certain": [], "vague": [], "feelings": [], "private_hints": []
    })
    emotion = MagicMock()
    emotion.get_expression_style.return_value = {
        "tone": "正常友好", "message_length": "正常", "emoji_freq": "中",
        "typo_rate": 0.0, "response_speed": "normal", "sticker_mood": "neutral"
    }
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    life_sim = MagicMock()
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司",
        "physical": {"energy": 80}
    }
    stickers = MagicMock()
    stickers.should_use_sticker = AsyncMock(return_value=False)
    browser = MagicMock()
    browser.get_recent_interesting.return_value = []
    rel = MagicMock()
    rel.get_social_context_for_prompt.return_value = "陌生人"
    rel.get.return_value = None
    chat_ctx = MagicMock()
    chat_ctx.get_recent_context.return_value = ""
    inner_state = MagicMock()
    inner_state.get_monologue_text.return_value = monologue_text

    synth = ExpressionSynthesizer(
        persona, llm, memory, emotion, life_sim,
        stickers, browser, rel, chat_ctx,
        inner_state=inner_state,
    )
    return synth, llm


def test_monologue_injected_into_prompt():
    synth, llm = make_expression(monologue_text="有点想回家，工作太烦了")
    asyncio.run(synth.compose_reply(
        user_message="你好", user_id=1, chat_id=1
    ))
    prompt_arg = llm.call.call_args[0][0]
    assert "你心里在转的" in prompt_arg
    assert "有点想回家，工作太烦了" in prompt_arg


def test_no_monologue_section_when_empty():
    synth, llm = make_expression(monologue_text="")
    asyncio.run(synth.compose_reply(
        user_message="你好", user_id=1, chat_id=1
    ))
    prompt_arg = llm.call.call_args[0][0]
    assert "你心里在转的" not in prompt_arg


def test_no_monologue_section_when_inner_state_none():
    from expression import ExpressionSynthesizer
    persona = {
        "name": "测试", "background": "测试背景", "speaking_style": "随意",
    }
    llm = MagicMock()
    llm.call = AsyncMock(return_value="回复")
    memory = MagicMock()
    memory.recall = AsyncMock(return_value={
        "certain": [], "vague": [], "feelings": [], "private_hints": []
    })
    emotion = MagicMock()
    emotion.get_expression_style.return_value = {
        "tone": "正常友好", "message_length": "正常", "emoji_freq": "中",
        "typo_rate": 0.0, "response_speed": "normal", "sticker_mood": "neutral"
    }
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    life_sim = MagicMock()
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    stickers = MagicMock()
    stickers.should_use_sticker = AsyncMock(return_value=False)
    browser = MagicMock()
    browser.get_recent_interesting.return_value = []
    rel = MagicMock()
    rel.get_social_context_for_prompt.return_value = "陌生人"
    rel.get.return_value = None
    chat_ctx = MagicMock()
    chat_ctx.get_recent_context.return_value = ""
    synth = ExpressionSynthesizer(
        persona, llm, memory, emotion, life_sim,
        stickers, browser, rel, chat_ctx,
        inner_state=None,
    )
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt_arg = llm.call.call_args[0][0]
    assert "你心里在转的" not in prompt_arg
