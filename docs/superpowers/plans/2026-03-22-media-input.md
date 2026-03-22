# Media Input Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add vision support so the bot can see and respond to user-sent photos and static stickers via base64, while politely declining videos/animations without involving the LLM.

**Architecture:** Thread a `media: list | None` parameter down the call chain `bot → orchestrator → expression → llm`. Add `LLMClient.call_vision()` that builds OpenAI multimodal content blocks; `expression.py` calls it unconditionally (it falls back to plain text when `images=[]`). Videos/animations are declined with a hardcoded reply in `bot.py` before reaching the orchestrator.

**Tech Stack:** python-telegram-bot 20.x, openai AsyncOpenAI client, base64 stdlib, pytest + unittest.mock

---

## File Map

| File | Change |
|---|---|
| `llm.py` | Add `call_vision()` method |
| `orchestrator.py` | Add `media: list = None` to `handle_message()`, annotate stored text |
| `expression.py` | Add `media` to `compose_reply()` + `_compose()`, replace `llm.call()` with `call_vision()` |
| `bot.py` | Add `_DECLINE_MEDIA`, `_download_and_encode()`, `_handle_media_message()`, `_handle_decline_media_message()`; register new handlers |
| `tests/test_llm_vision.py` | New: unit tests for `call_vision` |
| `tests/test_bot_media.py` | New: unit tests for bot media handlers |
| `tests/test_orchestrator_media.py` | New: unit tests for orchestrator media propagation |
| `tests/test_expression_media.py` | New: unit tests for expression media annotation + call_vision dispatch |

---

## Task 1: Add `call_vision` to `LLMClient`

**Files:**
- Modify: `llm.py`
- Create: `tests/test_llm_vision.py`

### Step 1.1 — Write the failing tests

- [ ] Create `tests/test_llm_vision.py`:

```python
# tests/test_llm_vision.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Minimal 1×1 red pixel JPEG in base64
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
    "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
    "/9oADAMBAAIRAxEAPwCwABmX/9k="
)

_TINY_WEBP_B64 = "UklGRlYAAABXRUJQVlA4IEoAAADQAQCdASoBAAEAAkA4JZACdAEO/gHOAAA="


def _make_llm():
    from llm import LLMClient
    cfg = {
        "expression": {"api_key": "test", "base_url": "https://api.test", "model": "test-model"},
        "utility": {"api_key": "test", "base_url": "https://api.test", "model": "cheap-model"},
    }
    return LLMClient(cfg)


def test_call_vision_builds_multimodal_content():
    """call_vision sends a content list with text + image_url nodes."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "看起来是红色"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    images = [{"mime_type": "image/jpeg", "base64": _TINY_JPEG_B64, "label": "photo"}]
    result = asyncio.run(llm.call_vision("这是什么颜色？", images=images, tier="expression"))

    assert result == "看起来是红色"
    msg = captured["messages"][-1]
    assert isinstance(msg["content"], list)
    text_nodes = [c for c in msg["content"] if c["type"] == "text"]
    image_nodes = [c for c in msg["content"] if c["type"] == "image_url"]
    assert len(text_nodes) == 1
    assert text_nodes[0]["text"] == "这是什么颜色？"
    assert len(image_nodes) == 1
    assert image_nodes[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_call_vision_label_field_ignored():
    """label field in image dict is not included in the API call."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    images = [{"mime_type": "image/webp", "base64": _TINY_WEBP_B64, "label": "sticker"}]
    asyncio.run(llm.call_vision("看一下", images=images, tier="expression"))

    image_node = [c for c in captured["messages"][-1]["content"] if c["type"] == "image_url"][0]
    assert "label" not in image_node
    assert "label" not in image_node["image_url"]


def test_call_vision_empty_images_falls_back_to_plain_call():
    """With no images, call_vision delegates to plain call() with same params."""
    llm = _make_llm()

    called_with = {}
    async def fake_create(**kwargs):
        called_with.update(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    asyncio.run(llm.call_vision("hello", images=[], tier="expression",
                                system_prompt="sys", temperature=0.5, max_tokens=100))

    msg = called_with["messages"][-1]
    # plain text fallback: content is a string, not a list
    assert isinstance(msg["content"], str)
    assert msg["content"] == "hello"
    assert called_with["temperature"] == 0.5
    assert called_with["max_tokens"] == 100


def test_call_vision_none_images_falls_back():
    """call_vision(images=None) also falls back to plain text."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    asyncio.run(llm.call_vision("hello", images=None, tier="expression"))

    msg = captured["messages"][-1]
    assert isinstance(msg["content"], str)


def test_call_vision_system_prompt_forwarded():
    """system_prompt appears in messages when provided."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    images = [{"mime_type": "image/jpeg", "base64": _TINY_JPEG_B64, "label": "photo"}]
    asyncio.run(llm.call_vision("q", images=images, tier="expression", system_prompt="你是林小晴"))

    system_msgs = [m for m in captured["messages"] if m["role"] == "system"]
    assert system_msgs and system_msgs[0]["content"] == "你是林小晴"
```

- [ ] Run tests to confirm they fail:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_llm_vision.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError: 'LLMClient' object has no attribute 'call_vision'`

### Step 1.2 — Implement `call_vision` in `llm.py`

- [ ] Open `llm.py` and add the method after `call_json` (around line 88):

```python
    async def call_vision(
        self,
        prompt: str,
        images: list = None,
        tier: str = "expression",
        system_prompt: str = "",
        temperature: float = 0.85,
        max_tokens: int = 2000,
    ) -> str:
        """Send a multimodal request (text + base64 images). Falls back to plain
        text call when images is empty or None."""
        if not images:
            return await self.call(
                prompt, tier=tier, system_prompt=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )

        client = self.expression_client if tier == "expression" else self.utility_client
        model = self.expression_model if tier == "expression" else self.utility_model

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"},
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        logger.debug(
            "[LLM-IN] tier=%s model=%s vision=True images=%d",
            tier, model, len(images),
        )

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                )
                output = resp.choices[0].message.content.strip()
                logger.debug("[LLM-OUT] tier=%s\n%s", tier, output)
                return output
            except Exception as e:
                logger.warning(f"LLM vision call failed (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return ""
```

- [ ] Run tests to confirm they pass:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_llm_vision.py -v
```

Expected: 5 tests PASSED

### Step 1.3 — Commit

- [ ] Commit:

```bash
cd /Users/admin/Work/virtual-persona
git add llm.py tests/test_llm_vision.py
git commit -m "feat: add call_vision multimodal method to LLMClient"
```

---

## Task 2: Thread `media` through `orchestrator.py`

**Files:**
- Modify: `orchestrator.py` (lines 86–163, specifically `handle_message` signature and body)
- Create: `tests/test_orchestrator_media.py`

### Step 2.1 — Write the failing tests

- [ ] Create `tests/test_orchestrator_media.py`:

```python
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
    assert call_kwargs.kwargs.get("media") == _TINY_JPEG or \
           (call_kwargs.args and _TINY_JPEG in call_kwargs.args)


def test_photo_label_appended_to_memory():
    """Photo media label '[图片]' is appended to stored message text."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="给你看", user_id=1, user_name="小明",
        chat_id=1, chat_type="private", media=_TINY_JPEG,
    ))

    stored_texts = [
        call.args[2] if len(call.args) > 2 else call.kwargs.get("content", "")
        for call in orch.memory.add_message.call_args_list
    ]
    # The user message stored in memory should contain [图片]
    user_stored = [t for t in stored_texts if "给你看" in t or "[图片]" in t]
    assert any("[图片]" in t for t in user_stored), f"No [图片] in stored texts: {stored_texts}"


def test_sticker_label_appended():
    """Static sticker label '[表情包]' is appended to stored message text."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="", user_id=1, user_name="小明",
        chat_id=1, chat_type="private", media=_TINY_STICKER,
    ))

    stored_texts = []
    for call in orch.memory.add_message.call_args_list:
        # memory.add_message(chat_id, role, text, ...)
        if len(call.args) >= 3:
            stored_texts.append(call.args[2])

    assert any("[表情包]" in t for t in stored_texts), f"No [表情包] in: {stored_texts}"


def test_no_media_unchanged():
    """handle_message with no media still works and passes media=None."""
    orch = _make_orchestrator()

    asyncio.run(orch.handle_message(
        text="你好", user_id=1, user_name="小明",
        chat_id=1, chat_type="private",
    ))

    call_kwargs = orch.expression.compose_reply.call_args
    passed_media = call_kwargs.kwargs.get("media")
    assert passed_media is None
```

- [ ] Run tests to confirm they fail:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_orchestrator_media.py -v 2>&1 | head -30
```

Expected: `FAILED` — `compose_reply` is not called with a `media` kwarg yet.

### Step 2.2 — Update `orchestrator.py`

- [ ] In `orchestrator.py`, update `handle_message` signature (line 87):

Find:
```python
    async def handle_message(
        self,
        text: str,
        user_id: int,
        user_name: str,
        chat_id: int,
        chat_type: str = "private",
        group_title: str = "",
        mentioned_me: bool = False,
    ) -> Optional[list]:
```

Replace with:
```python
    async def handle_message(
        self,
        text: str,
        user_id: int,
        user_name: str,
        chat_id: int,
        chat_type: str = "private",
        group_title: str = "",
        mentioned_me: bool = False,
        media: list = None,
    ) -> Optional[list]:
```

- [ ] In `orchestrator.py`, update the two `add_message` calls for the user message (steps 2 and 3 of `handle_message`, around lines 104–115). Find the block:

```python
        self.chat_ctx.add_message(
            chat_id, user_id, user_name, text,
            is_me=False, mentioned_me=mentioned_me,
        )

        # 3. 记入记忆缓冲
        self.memory.add_message(
            chat_id, "user", text,
            user_id=user_id, user_name=user_name,
            context_type="group" if chat_type in ("group", "supergroup") else "private",
        )
```

Replace with:
```python
        _media_label = ""
        if media:
            for m in media:
                if m.get("label") == "photo":
                    _media_label += " [图片]"
                elif m.get("label") == "sticker":
                    _media_label += " [表情包]"
        _stored_text = text + _media_label

        self.chat_ctx.add_message(
            chat_id, user_id, user_name, _stored_text,
            is_me=False, mentioned_me=mentioned_me,
        )

        # 3. 记入记忆缓冲
        self.memory.add_message(
            chat_id, "user", _stored_text,
            user_id=user_id, user_name=user_name,
            context_type="group" if chat_type in ("group", "supergroup") else "private",
        )
```

- [ ] In `orchestrator.py`, update the `expression.compose_reply` call (step 6, around line 137):

Find:
```python
        messages = await self.expression.compose_reply(
            user_message=text,
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            reply_mode=reply_mode,
        )
```

Replace with:
```python
        messages = await self.expression.compose_reply(
            user_message=text,
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            reply_mode=reply_mode,
            media=media,
        )
```

- [ ] Run the tests:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_orchestrator_media.py -v
```

Expected: 4 tests PASSED

- [ ] Also run existing orchestrator tests to confirm no regression:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_orchestrator_follow_up.py -v
```

Expected: PASSED

### Step 2.3 — Commit

- [ ] Commit:

```bash
cd /Users/admin/Work/virtual-persona
git add orchestrator.py tests/test_orchestrator_media.py
git commit -m "feat: thread media through orchestrator handle_message"
```

---

## Task 3: Update `expression.py` to use `call_vision`

**Files:**
- Modify: `expression.py` (lines 39–53 `compose_reply`, lines 64–72 `_compose`, lines 91–238 `_compose` body)
- Create: `tests/test_expression_media.py`

### Step 3.1 — Write the failing tests

- [ ] Create `tests/test_expression_media.py`:

```python
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


def test_call_plain_when_no_media():
    """compose_reply calls llm.call_vision with empty images when no media (fallback to plain)."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="你好", user_id=1, chat_id=1, chat_type="private",
    ))

    # call_vision is always used; with empty images it falls back to plain text
    expr.llm.call_vision.assert_called_once()
    images_arg = expr.llm.call_vision.call_args.kwargs.get("images") or \
                 expr.llm.call_vision.call_args.args[1]
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

    prompt_arg = expr.llm.call_vision.call_args.args[0] if expr.llm.call_vision.call_args.args \
                 else expr.llm.call_vision.call_args.kwargs.get("prompt", "")
    assert "photo" in prompt_arg or "图片" in prompt_arg or "帮我看看" in prompt_arg


def test_sticker_label_in_prompt():
    """Sticker label 'sticker' appears in the prompt annotation."""
    expr = _make_expression()

    asyncio.run(expr.compose_reply(
        user_message="", user_id=1, chat_id=1, chat_type="private", media=_STICKER,
    ))

    prompt_arg = expr.llm.call_vision.call_args.args[0] if expr.llm.call_vision.call_args.args \
                 else expr.llm.call_vision.call_args.kwargs.get("prompt", "")
    assert "sticker" in prompt_arg or "表情包" in prompt_arg
```

- [ ] Run tests to confirm they fail:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_expression_media.py -v 2>&1 | head -30
```

Expected: `FAILED` — `compose_reply` doesn't accept `media` yet / `call_vision` not called.

### Step 3.2 — Update `expression.py`

- [ ] Update `compose_reply` signature (around line 39):

Find:
```python
    async def compose_reply(
        self,
        user_message: str,
        user_id: int,
        chat_id: int,
        chat_type: str = "private",
        reply_mode: str = "direct",
    ) -> List[dict]:
        return await self._compose(
            user_message=user_message,
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            reply_mode=reply_mode,
        )
```

Replace with:
```python
    async def compose_reply(
        self,
        user_message: str,
        user_id: int,
        chat_id: int,
        chat_type: str = "private",
        reply_mode: str = "direct",
        media: list = None,
    ) -> List[dict]:
        return await self._compose(
            user_message=user_message,
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            reply_mode=reply_mode,
            media=media,
        )
```

- [ ] Update `_compose` signature (around line 64):

Find:
```python
    async def _compose(
        self,
        user_message: str = None,
        proactive_trigger: dict = None,
        user_id: int = 0,
        chat_id: int = 0,
        chat_type: str = "private",
        reply_mode: str = "direct",
    ) -> List[dict]:
```

Replace with:
```python
    async def _compose(
        self,
        user_message: str = None,
        proactive_trigger: dict = None,
        user_id: int = 0,
        chat_id: int = 0,
        chat_type: str = "private",
        reply_mode: str = "direct",
        media: list = None,
    ) -> List[dict]:
```

- [ ] Replace the "对方发来" section and LLM call in `_compose` (around lines 237–259). Find:

```python
        prompt = f"""你是{self.persona['name']}，一个真实的人在用手机聊天。
```

Within that prompt string, find the section near the bottom:
```python
## {'对方发来' if user_message else '你想主动说'}
{'"' + user_message + '"' if user_message else proactive_context}
```

Replace with:
```python
## {'对方发来' if user_message else '你想主动说'}
{_build_user_content_line(user_message, proactive_context, media)}
```

Then add this helper at module level (before the `ExpressionSynthesizer` class, around line 24):

```python
def _build_user_content_line(user_message, proactive_context, media):
    if user_message is not None:
        if media:
            labels = "、".join(m["label"] for m in media)
            media_note = f"（附带了 {labels}）"
        else:
            media_note = ""
        return f'"{user_message}"{media_note}'
    return proactive_context
```

- [ ] Replace the `self.llm.call(...)` call (around line 255):

Find:
```python
        raw = await self.llm.call(
            prompt, tier="expression",
            system_prompt=system_prompt,
            temperature=0.85,
        )
```

Replace with:
```python
        raw = await self.llm.call_vision(
            prompt,
            images=media or [],
            tier="expression",
            system_prompt=system_prompt,
            temperature=0.85,
            max_tokens=2000,
        )
```

- [ ] Run the tests:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_expression_media.py -v
```

Expected: 5 tests PASSED

- [ ] Run existing expression tests to confirm no regression:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_expression_persona_traits.py tests/test_expression_monologue.py -v
```

Expected: all PASSED

### Step 3.3 — Commit

- [ ] Commit:

```bash
cd /Users/admin/Work/virtual-persona
git add expression.py tests/test_expression_media.py
git commit -m "feat: thread media through expression and dispatch call_vision"
```

---

## Task 4: Add media handlers to `bot.py`

**Files:**
- Modify: `bot.py`
- Create: `tests/test_bot_media.py`

### Step 4.1 — Write the failing tests

- [ ] Create `tests/test_bot_media.py`:

```python
# tests/test_bot_media.py
import asyncio
import base64
from unittest.mock import MagicMock, AsyncMock, patch


def _make_bot():
    from bot import VirtualPersonaBot
    orch = MagicMock()
    orch.handle_message = AsyncMock(return_value=[
        {"type": "text", "content": "好漂亮～", "delay": 0}
    ])
    orch.set_proactive_callback = MagicMock()
    orch.persona = {"name": "林小晴"}

    cfg = {
        "bot_token": "test:token",
        "admin_user_id": 999,
        "allowed_users": [],
        "allowed_groups": [],
    }
    bot = VirtualPersonaBot(orch, cfg)
    bot.bot_username = "testbot"
    bot.bot_id = 12345
    return bot


def _make_photo_update(user_id=1, chat_id=1, chat_type="private",
                       caption=None, is_bot=False, reply_to_bot=False):
    """Build a minimal mock photo Update."""
    update = MagicMock()
    update.message = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.is_bot = is_bot
    update.effective_user.first_name = "小明"
    update.effective_user.last_name = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.effective_chat.title = "测试群"

    # photo: list of PhotoSize, use largest (last)
    photo_size = MagicMock()
    photo_size.file_id = "photo_file_id_abc"
    update.message.photo = [photo_size]
    update.message.sticker = None
    update.message.caption = caption

    if reply_to_bot:
        reply_msg = MagicMock()
        reply_msg.from_user = MagicMock()
        reply_msg.from_user.id = 12345  # bot_id
        update.message.reply_to_message = reply_msg
    else:
        update.message.reply_to_message = None

    update.message.message_id = 42
    return update


def _make_video_update(user_id=1, chat_id=1, chat_type="private", is_bot=False):
    update = MagicMock()
    update.message = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.is_bot = is_bot
    update.effective_user.first_name = "小明"
    update.effective_user.last_name = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.caption = None
    update.message.reply_to_message = None
    update.message.message_id = 43
    return update


def test_photo_message_calls_orchestrator_with_media():
    """Photo messages download the file and pass media to orchestrator."""
    bot_obj = _make_bot()
    update = _make_photo_update()

    fake_bytes = b"\xff\xd8\xff\xe0test_jpeg_bytes"
    expected_b64 = base64.b64encode(fake_bytes).decode()

    async def fake_get_file(file_id):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(fake_bytes))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    bot_obj.orch.handle_message.assert_called_once()
    call_kwargs = bot_obj.orch.handle_message.call_args.kwargs
    media = call_kwargs.get("media")
    assert media is not None and len(media) == 1
    assert media[0]["mime_type"] == "image/jpeg"
    assert media[0]["base64"] == expected_b64
    assert media[0]["label"] == "photo"


def test_photo_with_caption_passes_text():
    """Caption text is passed as the `text` argument to orchestrator."""
    bot_obj = _make_bot()
    update = _make_photo_update(caption="这是我的猫")

    async def fake_get_file(file_id):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8\xff"))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    call_kwargs = bot_obj.orch.handle_message.call_args.kwargs
    assert call_kwargs.get("text") == "这是我的猫"


def test_photo_download_failure_proceeds_without_media():
    """Download failure logs and calls orchestrator with media=None."""
    bot_obj = _make_bot()
    update = _make_photo_update(caption="看")

    async def fake_get_file(file_id):
        raise Exception("network error")

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    call_kwargs = bot_obj.orch.handle_message.call_args.kwargs
    assert call_kwargs.get("media") is None


def test_bot_sender_silently_ignored():
    """Messages from bots are silently dropped."""
    bot_obj = _make_bot()
    update = _make_photo_update(is_bot=True)
    ctx = MagicMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))
    bot_obj.orch.handle_message.assert_not_called()


def test_video_decline_reply_sent():
    """Video messages get a hardcoded decline reply."""
    from bot import _DECLINE_MEDIA
    bot_obj = _make_bot()
    update = _make_video_update()

    sent_texts = []

    async def fake_send(chat_id, text, **kwargs):
        sent_texts.append(text)

    ctx = MagicMock()
    ctx.bot.send_message = fake_send

    asyncio.run(bot_obj._handle_decline_media_message(update, ctx))

    assert len(sent_texts) == 1
    assert sent_texts[0] in _DECLINE_MEDIA
    bot_obj.orch.handle_message.assert_not_called()


def test_video_decline_suppressed_in_group_when_not_mentioned():
    """Video decline is silent in group chats when bot not mentioned."""
    bot_obj = _make_bot()
    update = _make_video_update(chat_type="group")

    sent_texts = []

    async def fake_send(chat_id, text, **kwargs):
        sent_texts.append(text)

    ctx = MagicMock()
    ctx.bot.send_message = fake_send

    asyncio.run(bot_obj._handle_decline_media_message(update, ctx))
    assert len(sent_texts) == 0


def test_mentioned_me_detected_from_caption():
    """mentioned_me=True when caption contains @botusername."""
    bot_obj = _make_bot()
    update = _make_photo_update(caption="@testbot 看这个")

    async def fake_get_file(file_id):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8"))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    call_kwargs = bot_obj.orch.handle_message.call_args.kwargs
    assert call_kwargs.get("mentioned_me") is True
```

- [ ] Run tests to confirm they fail:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_bot_media.py -v 2>&1 | head -40
```

Expected: `ImportError: cannot import name '_DECLINE_MEDIA'` or `AttributeError: '_handle_media_message'`

### Step 4.2 — Implement media handlers in `bot.py`

- [ ] Add imports at the top of `bot.py` (after `import random`):

```python
import base64
```

- [ ] Add `_DECLINE_MEDIA` constant after the `logger` line (before the `_retry_on_timeout` decorator):

```python
_DECLINE_MEDIA = [
    "这个我看不了哎～",
    "视频/动图我这边显示不出来哈哈",
    "嗯这个我打开不了，发图片给我吧",
]
```

- [ ] Add these three methods to `VirtualPersonaBot`, **before** `_handle_message` (around line 57). Insert after the `_is_allowed` method:

```python
    async def _download_and_encode(self, bot, file_id: str, label: str) -> dict | None:
        """Download a Telegram file and return a base64 media dict. Returns None on error."""
        try:
            tg_file = await bot.get_file(file_id)
            data = await tg_file.download_as_bytearray()
            b64 = base64.b64encode(bytes(data)).decode()
            mime = "image/webp" if label == "sticker" else "image/jpeg"
            return {"mime_type": mime, "base64": b64, "label": label}
        except Exception as e:
            logger.warning(f"[Bot] 媒体下载失败 ({label}): {e}")
            return None

    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo and static sticker messages."""
        if not update.message or not update.effective_user:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type

        # Guard: _is_allowed first, then is_bot
        if not self._is_allowed(user_id, chat_id, chat_type):
            return
        if update.effective_user.is_bot:
            return

        user_name = (
            update.effective_user.first_name or ""
        ) + (
            " " + (update.effective_user.last_name or "") if update.effective_user.last_name else ""
        )
        user_name = user_name.strip() or f"User{user_id}"

        # Extract caption (may be None or empty)
        caption = update.message.caption or ""

        # Detect mentioned_me
        mentioned_me = False
        if self.bot_username and caption:
            if f"@{self.bot_username}" in caption:
                mentioned_me = True
                caption = caption.replace(f"@{self.bot_username}", "").strip()
        if update.message.reply_to_message and update.message.reply_to_message.from_user:
            if update.message.reply_to_message.from_user.id == self.bot_id:
                mentioned_me = True

        # Determine file_id and label
        file_id = None
        label = "photo"
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
            label = "photo"
        elif update.message.sticker and not update.message.sticker.is_animated \
                and not update.message.sticker.is_video:
            file_id = update.message.sticker.file_id
            label = "sticker"

        # Download and encode (None on error → text-only reply)
        media = None
        if file_id:
            enc = await self._download_and_encode(context.bot, file_id, label)
            if enc:
                media = [enc]

        group_title = ""
        if chat_type in ("group", "supergroup"):
            group_title = update.effective_chat.title or ""

        logger.info(
            f"[Bot] 媒体消息 | {user_name}({user_id}) | label={label} | "
            f"caption={caption[:30]!r} | mentioned={mentioned_me}"
        )

        self.orch.set_proactive_callback(
            chat_id,
            lambda msgs, _cid=chat_id: self._send_messages(context.bot, _cid, msgs),
        )

        messages = await self.orch.handle_message(
            text=caption,
            user_id=user_id,
            user_name=user_name,
            chat_id=chat_id,
            chat_type=chat_type,
            group_title=group_title,
            mentioned_me=mentioned_me,
            media=media,
        )

        if messages is None:
            return

        await self._send_messages(
            context.bot, chat_id, messages,
            reply_to=update.message.message_id if chat_type != "private" else None,
        )

    async def _handle_decline_media_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle animation, video, video_note, animated/video sticker — decline politely."""
        if not update.message or not update.effective_user:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type

        # Guard: _is_allowed first, then is_bot
        if not self._is_allowed(user_id, chat_id, chat_type):
            return
        if update.effective_user.is_bot:
            return

        # Group-chat suppression: only reply if mentioned/addressed
        if chat_type in ("group", "supergroup"):
            caption = update.message.caption or ""
            mentioned_me = False
            if self.bot_username and f"@{self.bot_username}" in caption:
                mentioned_me = True
            if update.message.reply_to_message and update.message.reply_to_message.from_user:
                if update.message.reply_to_message.from_user.id == self.bot_id:
                    mentioned_me = True
            if not mentioned_me:
                return

        reply_text = random.choice(_DECLINE_MEDIA)
        await context.bot.send_message(chat_id=chat_id, text=reply_text)
```

- [ ] Register the new handlers in `run()`. Find the existing handler registration block (around line 484):

```python
        # 所有文本消息
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))
```

Add after it:

```python
        # 图片 + 静态表情包 → 视觉理解
        app.add_handler(MessageHandler(
            (filters.PHOTO | filters.Sticker.STATIC) & ~filters.COMMAND,
            self._handle_media_message,
        ))
        # 视频 / 动图 / 动态表情包 → 礼貌拒绝
        app.add_handler(MessageHandler(
            (filters.VIDEO | filters.ANIMATION | filters.VIDEO_NOTE
             | filters.Sticker.ANIMATED | filters.Sticker.VIDEO) & ~filters.COMMAND,
            self._handle_decline_media_message,
        ))
```

- [ ] Run the tests:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_bot_media.py -v
```

Expected: 7 tests PASSED

- [ ] Run full test suite for final regression check:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/ -v
```

Expected: all PASSED (no regressions)

### Step 4.3 — Commit

- [ ] Commit:

```bash
cd /Users/admin/Work/virtual-persona
git add bot.py tests/test_bot_media.py
git commit -m "feat: add photo/sticker media handlers and video decline to bot"
```

---

## Task 5: Final smoke-test and cleanup

### Step 5.1 — Full test suite

- [ ] Run full test suite one final time:

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASSED. Review any failures before proceeding.

### Step 5.2 — Syntax check on all changed files

- [ ] Verify no syntax errors:

```bash
cd /Users/admin/Work/virtual-persona
python -c "import llm, orchestrator, expression, bot; print('OK')"
```

Expected: `OK`

### Step 5.3 — Final commit (if any cleanup needed)

- [ ] If any minor fixes were needed:

```bash
cd /Users/admin/Work/virtual-persona
git add -p
git commit -m "fix: cleanup after media input feature"
```
