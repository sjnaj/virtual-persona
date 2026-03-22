# tests/test_expression_persona_traits.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


def make_expression_with_traits():
    from expression import ExpressionSynthesizer
    persona = {
        "name": "测试",
        "background": "测试背景",
        "speaking_style": "随意",
        "age": 24,
        "mbti": "ENFP",
        "interests": ["猫", "穿搭", "甜品"],
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
        "typo_rate": 0.0, "response_speed": "normal", "sticker_mood": "neutral",
    }
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    life_sim = MagicMock()
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80},
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
    )
    return synth, llm


def test_trait_section_heading_in_prompt():
    synth, llm = make_expression_with_traits()
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    assert "你的特质" in prompt


def test_age_in_prompt():
    synth, llm = make_expression_with_traits()
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    assert "24岁" in prompt


def test_mbti_in_prompt():
    synth, llm = make_expression_with_traits()
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    assert "ENFP" in prompt


def test_mbti_description_in_prompt():
    synth, llm = make_expression_with_traits()
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    # ENFP should include its full Chinese description string
    assert "热情开放、充满好奇心" in prompt


def test_interests_in_prompt():
    synth, llm = make_expression_with_traits()
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    assert "穿搭" in prompt
    assert "甜品" in prompt


def test_trait_section_appears_between_background_and_current_state():
    synth, llm = make_expression_with_traits()
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    # 你的特质 must come after 你是谁 and before 此刻状态
    idx_who = prompt.find("你是谁")
    idx_traits = prompt.find("你的特质")
    idx_state = prompt.find("此刻状态")
    assert idx_who < idx_traits < idx_state


def test_missing_traits_graceful():
    """Persona without age/mbti/interests should not raise and must not emit broken labels."""
    from expression import ExpressionSynthesizer
    persona = {
        "name": "测试", "background": "背景", "speaking_style": "随意",
        # no age, mbti, interests
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
        "typo_rate": 0.0, "response_speed": "normal", "sticker_mood": "neutral",
    }
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    life_sim = MagicMock()
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80},
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
    )
    # Should not raise
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt = llm.call.call_args[0][0]
    assert "年龄：岁" not in prompt       # no broken empty age label
    assert "MBTI：\n" not in prompt       # no empty MBTI line
    assert "## 你的特质" not in prompt    # entire section suppressed when all fields absent
