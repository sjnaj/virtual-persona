# Activate Unused Persona Config Fields — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `age`, `mbti`, `interests`, and `daily_patterns.work_start/work_end/lunch` from `config.yaml` into the prompts that drive character expression and life simulation.

**Architecture:** Two files are modified. `expression.py` gains a `## 你的特质` section injected into the LLM prompt inside `_compose()`. `life_simulator.py` gains an interests line in `tick()`'s prompt and a dynamic schedule annotation appended by `_get_activity_hints()` for weekdays.

**Tech Stack:** Python 3.11+, pytest, unittest.mock

---

## File Map

| File | Change |
|------|--------|
| `expression.py` | Add `trait_section` variable in `_compose()`; splice into prompt f-string |
| `life_simulator.py` | Add `interests_str` in `tick()`; insert into prompt f-string; refactor `_get_activity_hints()` to append schedule annotation for weekdays |
| `tests/test_expression_persona_traits.py` | New test file for Change 1 |
| `tests/test_life_simulator_enriched.py` | Add new tests for Changes 2 & 3 (existing file) |

---

## Task 1: expression.py — inject age, MBTI, interests into prompt

**Files:**
- Modify: `expression.py:184-224` (`_compose()` — prompt construction block)
- Create: `tests/test_expression_persona_traits.py`

---

- [ ] **Step 1: Write the failing tests**

Create `tests/test_expression_persona_traits.py`:

```python
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
    assert "猫" in prompt
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


def test_missing_traits_graceful(monkeypatch):
    """Persona without age/mbti/interests should not raise."""
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_expression_persona_traits.py -v
```

Expected: 6 tests FAIL ("你的特质 not in prompt" etc). `test_missing_traits_graceful` already passes before implementation since the current code doesn't raise on missing optional fields — that is expected and correct.

- [ ] **Step 3: Implement — add `trait_section` and splice into prompt**

In `expression.py`, inside `_compose()`, add before the `prompt = f"""...` line (around line 184):

```python
# ---- 人设特质 ----
interests_str = "、".join(self.persona.get("interests", []))
mbti = self.persona.get("mbti", "")
_mbti_desc = {
    "ENFP": "热情开放、充满好奇心、喜欢分享、容易被新事物吸引",
    "INFP": "理想主义、善解人意、内敛但情感深厚",
    "ENFJ": "热情有感染力、善于倾听、天生领导者",
    "INTJ": "独立理性、目标明确、不喜欢废话",
}
mbti_desc = _mbti_desc.get(mbti, "")
trait_section = (
    f"## 你的特质\n"
    f"年龄：{self.persona.get('age', '')}岁\n"
    f"MBTI：{mbti}{'（' + mbti_desc + '）' if mbti_desc else ''}\n"
    f"兴趣：{interests_str}\n"
)
```

Then modify the prompt f-string. Change:

```python
        prompt = f"""你是{self.persona['name']}，一个真实的人在用手机聊天。

## 你是谁
{self.persona['background']}

## 此刻状态
```

to:

```python
        prompt = f"""你是{self.persona['name']}，一个真实的人在用手机聊天。

## 你是谁
{self.persona['background']}

{trait_section}
## 此刻状态
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_expression_persona_traits.py -v
```

Expected: 7 tests PASS (all, including the graceful-degradation test that already passed)

- [ ] **Step 5: Confirm existing expression tests still pass**

```bash
python -m pytest tests/test_expression_monologue.py -v
```

Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add expression.py tests/test_expression_persona_traits.py
git commit -m "feat: inject age/mbti/interests into expression prompt"
```

---

## Task 2: life_simulator.py — add interests to tick prompt

**Files:**
- Modify: `life_simulator.py:143-169` (`tick()` — prompt construction)
- Modify: `tests/test_life_simulator_enriched.py` (add new tests at end of file)

---

- [ ] **Step 1: Write the failing test**

Append to `tests/test_life_simulator_enriched.py`:

```python
# ── Task: interests in tick prompt ──────────────────────────────────────────

def _make_tick_sim_with_interests():
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    from unittest.mock import patch
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {
            "wake_up": [7, 8], "sleep": [23, 25],
            "work_start": [9, 10], "work_end": [18, 19], "lunch": [11, 13],
        },
        "interests": ["猫", "穿搭", "甜品"],
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = LifeSimulator(persona, llm, bus)
    sim.is_sleeping = False
    sim.woke_up_today = True
    sim.physical.energy = 60.0
    sim.daily_weather = {
        "condition": "晴", "temp": 25,
        "date": datetime.now().date().isoformat(),
    }
    default_result = {
        "action": "做设计", "location": "公司", "detail": "改稿",
        "mood_impact": 0, "energy_change": -2, "notable": False,
        "shareable_thought": None,
    }
    sim.llm.call_json = AsyncMock(return_value=default_result)
    return sim


def test_tick_prompt_contains_interests():
    sim = _make_tick_sim_with_interests()
    captured = []
    original = sim.llm.call_json
    async def capture(prompt, **kwargs):
        captured.append(prompt)
        return await original(prompt, **kwargs)
    sim.llm.call_json = capture

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 15, 0)
            await sim.tick()
    asyncio.run(run())

    assert captured, "call_json never called"
    prompt = captured[0]
    assert "她的兴趣爱好" in prompt
    # Use "穿搭" (not "猫" — "猫" appears in yearago_str regardless of this change)
    assert "穿搭" in prompt
    assert "甜品" in prompt
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
python -m pytest tests/test_life_simulator_enriched.py::test_tick_prompt_contains_interests -v
```

Expected: FAIL — "她的兴趣爱好 not in prompt"

- [ ] **Step 3: Implement — add interests_str and insert into tick prompt**

In `life_simulator.py`, inside `tick()`, add before the `prompt = f"""...` line (around line 143):

```python
interests_str = "、".join(self.persona.get("interests", []))
```

Then in the prompt f-string, change:

```python
她的职业：{self.persona.get('occupation', '')}
她的性格：外向{self.persona['personality']['extraversion']:.1f} 自律{self.persona['personality']['conscientiousness']:.1f}
```

to:

```python
她的职业：{self.persona.get('occupation', '')}
她的兴趣爱好：{interests_str}
她的性格：外向{self.persona['personality']['extraversion']:.1f} 自律{self.persona['personality']['conscientiousness']:.1f}
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest tests/test_life_simulator_enriched.py::test_tick_prompt_contains_interests -v
```

Expected: PASS

- [ ] **Step 5: Commit**

> ⚠️ Task 3 also modifies these same two files. Only stage the `tick()` change here (`interests_str` variable + one prompt line). Do NOT stage `_get_activity_hints()` changes yet — those belong to Task 3.

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py
git commit -m "feat: add interests to life_simulator tick prompt"
```

---

## Task 3: life_simulator.py — dynamic schedule hint in `_get_activity_hints()`

**Files:**
- Modify: `life_simulator.py:337-350` (`_get_activity_hints()`)
- Modify: `tests/test_life_simulator_enriched.py` (add new tests at end)

---

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_life_simulator_enriched.py`:

```python
# ── Task: dynamic schedule hint ─────────────────────────────────────────────

def _make_sim_with_schedule():
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    from unittest.mock import patch
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {
            "wake_up": [7, 8], "sleep": [23, 25],
            "work_start": [9, 10], "work_end": [18, 19], "lunch": [11, 13],
        },
        "interests": [],
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = LifeSimulator(persona, llm, bus)
    return sim


def test_weekday_hints_include_work_start():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "上班时间约9-10" in hints  # work_start [9,10] from config


def test_weekday_hints_include_work_end():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "下班约18-19" in hints  # work_end [18,19] from config


def test_weekday_hints_include_lunch():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "午饭约11-13" in hints  # lunch [11,13] from config


def test_weekend_hints_do_not_include_schedule_annotation():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(14.0, is_weekday=False)
    # Weekend should not have schedule parenthetical (this test passes before
    # implementation too — it is a guard that the weekend path stays clean)
    assert "上班时间" not in hints
    assert "下班" not in hints


def test_schedule_annotation_uses_config_not_hardcoded():
    """Custom schedule values appear in hints, not hardcoded defaults."""
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    from unittest.mock import patch
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {
            "wake_up": [6, 7], "sleep": [22, 23],
            "work_start": [8, 9], "work_end": [17, 18], "lunch": [12, 13],
        },
        "interests": [],
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = LifeSimulator(persona, llm, bus)
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "上班时间约8-9" in hints   # custom work_start
    assert "下班约17-18" in hints     # custom work_end
    assert "午饭约12-13" in hints     # custom lunch
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_life_simulator_enriched.py -k "schedule" -v
```

Expected: 4 tests FAIL (the schedule-text tests). `test_weekend_hints_do_not_include_schedule_annotation` already passes before implementation — that is expected and correct (it guards the weekend path).

- [ ] **Step 3: Implement — refactor `_get_activity_hints()` and append schedule**

Replace the current `_get_activity_hints()` method in `life_simulator.py` (lines ~344-350) with:

```python
def _get_activity_hints(self, hour: float, is_weekday: bool) -> str:
    pool = self._ACTIVITY_POOL_WEEKDAY if is_weekday else self._ACTIVITY_POOL_WEEKEND
    h = int(hour) % 24
    base_hint = "发呆、刷手机、失眠"
    for start, end, hints in pool:
        if start <= h < end:
            base_hint = hints
            break

    if not is_weekday:
        return base_hint

    # Append schedule context from config (loop refactored from early-return to
    # break+variable to enable this appending — logic is equivalent)
    dp = self.persona.get("daily_patterns", {})
    ws = dp.get("work_start", [9, 10])
    we = dp.get("work_end", [18, 19])
    ln = dp.get("lunch", [11, 13])
    schedule = (
        f"（上班时间约{ws[0]}-{ws[1]}点，"
        f"午饭约{ln[0]}-{ln[1]}点，"
        f"下班约{we[0]}-{we[1]}点）"
    )
    return base_hint + "，" + schedule
```

- [ ] **Step 4: Run new schedule tests**

```bash
python -m pytest tests/test_life_simulator_enriched.py -k "schedule" -v
```

Expected: 5 tests PASS (all schedule tests, including the already-passing weekend guard)

- [ ] **Step 5: Run all life_simulator tests to confirm no regressions**

```bash
python -m pytest tests/test_life_simulator_enriched.py tests/test_life_woke_up.py -v
```

Expected: all PASS (the existing `test_activity_hints_weekday_*` tests check for keywords that are still present in `base_hint` before appending)

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass (count will be higher than 78 due to new tests)

- [ ] **Step 7: Commit**

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py
git commit -m "feat: add dynamic schedule hint to life_simulator activity hints"
```
