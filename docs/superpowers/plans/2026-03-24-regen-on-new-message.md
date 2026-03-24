# Regen-on-New-Message Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before sending a reply, if new messages have arrived from the user, incorporate them and regenerate the reply once.

**Architecture:** Split `orchestrator.handle_message` into three separate methods (`ingest_message`, `_generate_reply`, `_record_reply`), then add a per-chat async queue + worker in `bot.py` that accumulates rapid messages and does a pre-send staleness check, regenerating at most once if new messages arrived during LLM generation.

**Tech Stack:** Python 3.11+, asyncio, python-telegram-bot, pytest + unittest.mock

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `orchestrator.py` | Modify | Extract `ingest_message()`, `_generate_reply()`, `_record_reply()`; refactor `handle_message()` as thin wrapper |
| `bot.py` | Modify | Add `_chat_queues`, `_chat_workers`, `self._bot`, `_enqueue_message()`, `_chat_worker()`; migrate `_handle_message` + `_handle_media_message` to enqueue |
| `tests/test_orchestrator_queue.py` | Create | Tests for `ingest_message`, `_generate_reply`, `_record_reply` |
| `tests/test_bot_worker.py` | Create | Tests for `_enqueue_message`, `_chat_worker` (batching, regen) |
| `tests/test_bot_media.py` | Modify | Update assertions from `orch.handle_message` to `_enqueue_message` |

---

## Task 1: Tests for new orchestrator methods

**Files:**
- Create: `tests/test_orchestrator_queue.py`

The existing `_make_orchestrator()` helper in `test_orchestrator_media.py` creates a fully-mocked orchestrator. Copy that pattern here.

- [ ] **Step 1.1: Write failing tests for `ingest_message`**

```python
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
```

- [ ] **Step 1.2: Run tests — expect ALL to fail**

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_orchestrator_queue.py -v 2>&1 | head -40
```

Expected: `AttributeError: 'Orchestrator' object has no attribute 'ingest_message'` (or similar) for each test.

---

## Task 2: Refactor orchestrator.py

**Files:**
- Modify: `orchestrator.py`

Look at `orchestrator.py` lines 86–174 (`handle_message`). The refactoring:
- Steps 1–5 (lines 101–144) → `ingest_message()`
- Step 6 (lines 146–154) → `_generate_reply()`
- Steps 7–8 (lines 156–172) → `_record_reply()`
- `handle_message()` becomes a 5-line wrapper

- [ ] **Step 2.1: Add `ingest_message()` method**

Add this method to `Orchestrator` class (before `handle_message`):

```python
async def ingest_message(
    self,
    text: str,
    user_id: int,
    user_name: str,
    chat_id: int,
    chat_type: str = "private",
    group_title: str = "",
    mentioned_me: bool = False,
    media: list = None,
) -> Optional[str]:
    """
    Update all state for an incoming message WITHOUT generating a reply.
    Returns reply_mode string, or None if a group chat decided not to reply.
    """
    # 1. Update relationship
    self.relationships.get_or_create(user_id, user_name)

    # 2. Update chat context
    self.chat_ctx.get_or_create(chat_id, chat_type, group_title)
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

    # 3. Add to memory buffer
    self.memory.add_message(
        chat_id, "user", _stored_text,
        user_id=user_id, user_name=user_name,
        context_type="group" if chat_type in ("group", "supergroup") else "private",
    )

    # 4. Notify emotion system
    await self.bus.emit("user.message", {"text": text, "user_id": user_id})
    self.proactive.update_last_message_time(datetime.now())

    # 5. Group behavior decision
    reply_mode = "direct"
    if chat_type in ("group", "supergroup"):
        window = self.chat_ctx.get_or_create(chat_id, chat_type, group_title)
        decision = await self.group_behavior.should_respond(
            chat_window=window,
            sender_id=user_id,
            message_text=text,
            mentioned_me=mentioned_me,
            emotion_state=self.emotion.get_status(),
        )
        if not decision["should_reply"]:
            logger.debug(f"[Orchestrator] 群聊中选择不回复 (prob={decision['probability']})")
            return None
        reply_mode = decision["reply_mode"]

    return reply_mode
```

- [ ] **Step 2.2: Add `_generate_reply()` method**

Add after `ingest_message`:

```python
async def _generate_reply(
    self,
    user_message: str,
    user_id: int,
    chat_id: int,
    chat_type: str,
    reply_mode: str,
    media: list = None,
) -> list:
    """Generate a reply via the expression layer. Does NOT record it."""
    return await self.expression.compose_reply(
        user_message=user_message,
        user_id=user_id,
        chat_id=chat_id,
        chat_type=chat_type,
        reply_mode=reply_mode,
        media=media,
    )
```

- [ ] **Step 2.3: Add `_record_reply()` method**

Add after `_generate_reply`:

```python
async def _record_reply(
    self,
    messages: list,
    user_id: int,
    chat_id: int,
    chat_type: str,
) -> None:
    """Record the bot's reply into memory/context and update relationship."""
    context_type = "group" if chat_type in ("group", "supergroup") else "private"
    recent_convo = self.chat_ctx.get_recent_context(chat_id, limit=20)

    for msg in messages:
        if msg["type"] == "text":
            self.memory.add_message(
                chat_id, "assistant", msg["content"],
                context_type=context_type,
            )
            self.chat_ctx.add_message(
                chat_id, 0, self.persona["name"],
                msg["content"], is_me=True,
            )

    sentiment = 0.1
    await self.relationships.update_after_conversation(user_id, recent_convo, sentiment)
```

- [ ] **Step 2.4: Refactor `handle_message()` to be a thin wrapper**

Replace the existing `handle_message` body with:

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
    """
    处理任意来源的消息。
    返回消息列表 或 None（群聊中决定不回复时）。
    """
    reply_mode = await self.ingest_message(
        text=text, user_id=user_id, user_name=user_name,
        chat_id=chat_id, chat_type=chat_type, group_title=group_title,
        mentioned_me=mentioned_me, media=media,
    )
    if reply_mode is None:
        return None

    messages = await self._generate_reply(
        user_message=text, user_id=user_id, chat_id=chat_id,
        chat_type=chat_type, reply_mode=reply_mode, media=media,
    )
    await self._record_reply(
        messages=messages, user_id=user_id,
        chat_id=chat_id, chat_type=chat_type,
    )
    return messages
```

- [ ] **Step 2.5: Run the new tests**

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_orchestrator_queue.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 2.6: Run existing orchestrator tests to confirm no regressions**

```bash
python -m pytest tests/test_orchestrator_media.py tests/test_orchestrator_follow_up.py -v
```

Expected: all PASS.

- [ ] **Step 2.7: Commit**

```bash
git add orchestrator.py tests/test_orchestrator_queue.py
git commit -m "refactor(orchestrator): extract ingest_message/_generate_reply/_record_reply

handle_message becomes a thin wrapper; new methods allow the per-chat
worker in bot.py to defer reply recording until after staleness check."
```

---

## Task 3: Add per-chat queue infrastructure to bot.py

**Files:**
- Create: `tests/test_bot_worker.py`
- Modify: `bot.py`

- [ ] **Step 3.1: Write failing tests for queue infrastructure**

```python
# tests/test_bot_worker.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


def _make_bot_with_mocked_orch():
    """Bot with all orchestrator methods mocked for worker testing."""
    from bot import VirtualPersonaBot

    orch = MagicMock()
    orch.config = {"system": {"message_accumulation_ms": 0}}  # skip wait
    orch.ingest_message = AsyncMock(return_value="direct")
    orch._generate_reply = AsyncMock(return_value=[
        {"type": "text", "content": "好的", "delay": 0}
    ])
    orch._record_reply = AsyncMock()
    orch.set_proactive_callback = MagicMock()
    orch.persona = {"name": "林小晴"}

    cfg = {
        "bot_token": "test:token",
        "admin_user_id": 999,
        "allowed_users": [],
        "allowed_groups": [],
    }
    bot_obj = VirtualPersonaBot(orch, cfg)
    bot_obj.bot_username = "testbot"
    bot_obj.bot_id = 12345
    bot_obj._bot = MagicMock()
    bot_obj._bot.send_message = AsyncMock()
    bot_obj._bot.send_chat_action = AsyncMock()
    bot_obj._bot.send_sticker = AsyncMock()
    return bot_obj


def _make_msg(text="你好", user_id=1, chat_id=100, chat_type="private"):
    return {
        "text": text,
        "user_id": user_id,
        "user_name": "小明",
        "chat_id": chat_id,
        "chat_type": chat_type,
        "group_title": "",
        "mentioned_me": False,
        "media": None,
        "reply_to": None,
    }


def test_enqueue_creates_queue_and_starts_worker():
    """_enqueue_message creates a queue and an asyncio.Task for the chat."""
    bot_obj = _make_bot_with_mocked_orch()

    # We call enqueue inside a running event loop
    async def run():
        bot_obj._enqueue_message(100, _make_msg())
        assert 100 in bot_obj._chat_queues
        assert 100 in bot_obj._chat_workers
        assert not bot_obj._chat_workers[100].done()
        # cancel to stop worker before it awaits sleep
        bot_obj._chat_workers[100].cancel()
        try:
            await bot_obj._chat_workers[100]
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def test_worker_processes_single_message():
    """Worker ingests, generates, records, and sends for a single message."""
    bot_obj = _make_bot_with_mocked_orch()

    async def run():
        bot_obj._enqueue_message(100, _make_msg())
        # Wait for worker to finish
        await bot_obj._chat_workers[100]

    asyncio.run(run())

    bot_obj.orch.ingest_message.assert_awaited_once()
    bot_obj.orch._generate_reply.assert_awaited_once()
    bot_obj.orch._record_reply.assert_awaited_once()
    bot_obj._bot.send_message.assert_awaited_once()


def test_worker_batches_rapid_messages():
    """Two messages enqueued before accumulation window ends are batched: ingest×2, generate×1."""
    bot_obj = _make_bot_with_mocked_orch()

    async def run():
        # Enqueue both before worker starts (accumulation_ms=0 but we need both in queue first)
        bot_obj._enqueue_message(100, _make_msg("你好"))
        bot_obj._enqueue_message(100, _make_msg("还在吗"))
        await bot_obj._chat_workers[100]

    asyncio.run(run())

    assert bot_obj.orch.ingest_message.await_count == 2
    assert bot_obj.orch._generate_reply.await_count == 1
    assert bot_obj.orch._record_reply.await_count == 1


def test_worker_regenerates_when_new_message_arrives_during_generation():
    """If a message arrives while _generate_reply is awaited, regen once, record once."""
    bot_obj = _make_bot_with_mocked_orch()
    queue = None

    call_count = [0]

    async def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Simulate new message arriving during first generation
            bot_obj._chat_queues[100].put_nowait(_make_msg("等等再说"))
        return [{"type": "text", "content": f"reply_{call_count[0]}", "delay": 0}]

    bot_obj.orch._generate_reply = AsyncMock(side_effect=generate_side_effect)

    async def run():
        bot_obj._enqueue_message(100, _make_msg("你好"))
        await bot_obj._chat_workers[100]

    asyncio.run(run())

    # generate called twice (initial + regen), record called once (final only)
    assert bot_obj.orch._generate_reply.await_count == 2
    assert bot_obj.orch._record_reply.await_count == 1

    # The recorded reply is the regen result (reply_2)
    recorded_msgs = bot_obj.orch._record_reply.call_args.kwargs.get("messages") or \
                    bot_obj.orch._record_reply.call_args.args[0]
    assert recorded_msgs[0]["content"] == "reply_2"


def test_worker_sends_original_reply_when_regen_batch_all_group_skips():
    """If regen-batch messages are all group-skips, original reply is sent."""
    bot_obj = _make_bot_with_mocked_orch()

    call_count = [0]
    original_reply = [{"type": "text", "content": "original", "delay": 0}]

    async def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            bot_obj._chat_queues[100].put_nowait(_make_msg("群消息"))
        return original_reply

    bot_obj.orch._generate_reply = AsyncMock(side_effect=generate_side_effect)

    # regen-batch messages all return None (group skip)
    ingest_count = [0]
    async def ingest_side_effect(**kwargs):
        ingest_count[0] += 1
        # First call (initial): return "direct"; second call (regen batch): return None
        return "direct" if ingest_count[0] == 1 else None

    bot_obj.orch.ingest_message = AsyncMock(side_effect=ingest_side_effect)

    async def run():
        bot_obj._enqueue_message(100, _make_msg("私聊消息"))
        await bot_obj._chat_workers[100]

    asyncio.run(run())

    # Generate only once (regen skipped), record + send the original
    assert bot_obj.orch._generate_reply.await_count == 1
    recorded_msgs = bot_obj.orch._record_reply.call_args.kwargs.get("messages") or \
                    bot_obj.orch._record_reply.call_args.args[0]
    assert recorded_msgs[0]["content"] == "original"


def test_worker_cleans_up_worker_entry_after_finish():
    """After processing all messages, worker removes itself from _chat_workers."""
    bot_obj = _make_bot_with_mocked_orch()

    async def run():
        bot_obj._enqueue_message(100, _make_msg())
        task = bot_obj._chat_workers[100]
        await task

    asyncio.run(run())

    assert 100 not in bot_obj._chat_workers


def test_enqueue_does_not_start_duplicate_worker():
    """A second enqueue while worker is running does not start a second worker."""
    bot_obj = _make_bot_with_mocked_orch()

    # Slow down generate so worker doesn't finish instantly
    async def slow_generate(**kwargs):
        await asyncio.sleep(0.05)
        return [{"type": "text", "content": "hi", "delay": 0}]

    bot_obj.orch._generate_reply = AsyncMock(side_effect=slow_generate)

    async def run():
        bot_obj._enqueue_message(100, _make_msg("M1"))
        first_task = bot_obj._chat_workers[100]
        bot_obj._enqueue_message(100, _make_msg("M2"))
        second_task = bot_obj._chat_workers[100]
        assert first_task is second_task  # same task, no duplicate
        await first_task

    asyncio.run(run())
```

- [ ] **Step 3.2: Run tests — expect all to fail**

```bash
python -m pytest tests/test_bot_worker.py -v 2>&1 | head -30
```

Expected: `AttributeError: 'VirtualPersonaBot' object has no attribute '_enqueue_message'`

- [ ] **Step 3.3: Add queue data structures to `VirtualPersonaBot.__init__`**

In `bot.py`, inside `VirtualPersonaBot.__init__` (after `self.app: Application = None`), add:

```python
        # Per-chat message queues and workers for serialized processing
        self._chat_queues: dict = {}
        self._chat_workers: dict = {}
        self._bot = None  # set in _post_init
```

- [ ] **Step 3.4: Store `self._bot` in `_post_init`**

In `_post_init`, after `self.app = app`, add:

```python
        self._bot = app.bot
```

- [ ] **Step 3.5: Add `_enqueue_message()` method**

Add this method to `VirtualPersonaBot` (after `_is_allowed`):

```python
    def _enqueue_message(self, chat_id: int, message_data: dict) -> None:
        """Enqueue a message for per-chat serial processing and start a worker if needed."""
        if chat_id not in self._chat_queues:
            self._chat_queues[chat_id] = asyncio.Queue()
        self._chat_queues[chat_id].put_nowait(message_data)

        # Check-and-create is synchronous — no await between check and create —
        # to prevent a race where two concurrent handlers both see a done task.
        existing = self._chat_workers.get(chat_id)
        if existing is None or existing.done():
            task = asyncio.create_task(self._chat_worker(chat_id))
            self._chat_workers[chat_id] = task
```

- [ ] **Step 3.6: Add `_chat_worker()` coroutine**

Add this method after `_enqueue_message`:

```python
    async def _chat_worker(self, chat_id: int) -> None:
        """
        Per-chat worker: drains the queue, generates a reply, checks for new
        messages before sending, regenerates once if needed, then sends.
        """
        queue = self._chat_queues[chat_id]
        try:
            while not queue.empty():
                # Step 1: dequeue first message
                first = queue.get_nowait()
                batch = [first]

                # Step 2: accumulation window — collect rapid follow-ups
                acc_ms = self.orch.config.get("system", {}).get("message_accumulation_ms", 500)
                await asyncio.sleep(acc_ms / 1000)

                # Step 3: drain any messages that arrived during the wait
                while not queue.empty():
                    batch.append(queue.get_nowait())

                # Step 4: ingest all messages (update state for each)
                last_reply_mode = None
                last_msg = None
                for msg in batch:
                    try:
                        reply_mode = await self.orch.ingest_message(
                            text=msg["text"],
                            user_id=msg["user_id"],
                            user_name=msg["user_name"],
                            chat_id=chat_id,
                            chat_type=msg["chat_type"],
                            group_title=msg["group_title"],
                            mentioned_me=msg["mentioned_me"],
                            media=msg.get("media"),
                        )
                    except Exception as e:
                        logger.error(f"[Worker:{chat_id}] ingest error: {e}", exc_info=True)
                        continue
                    if reply_mode is not None:
                        last_reply_mode = reply_mode
                        last_msg = msg

                # If all messages were group-skips, nothing to generate this cycle
                if last_reply_mode is None or last_msg is None:
                    continue

                # Step 5: generate reply (NOT recorded yet)
                try:
                    messages = await self.orch._generate_reply(
                        user_message=last_msg["text"],
                        user_id=last_msg["user_id"],
                        chat_id=chat_id,
                        chat_type=last_msg["chat_type"],
                        reply_mode=last_reply_mode,
                        media=last_msg.get("media"),
                    )
                except Exception as e:
                    logger.error(f"[Worker:{chat_id}] generate error: {e}", exc_info=True)
                    continue

                # Step 6: pre-send staleness check — regenerate at most once
                if not queue.empty():
                    regen_batch = []
                    while not queue.empty():
                        regen_batch.append(queue.get_nowait())

                    regen_last_mode = None
                    regen_last_msg = None
                    for msg in regen_batch:
                        try:
                            reply_mode = await self.orch.ingest_message(
                                text=msg["text"],
                                user_id=msg["user_id"],
                                user_name=msg["user_name"],
                                chat_id=chat_id,
                                chat_type=msg["chat_type"],
                                group_title=msg["group_title"],
                                mentioned_me=msg["mentioned_me"],
                                media=msg.get("media"),
                            )
                        except Exception as e:
                            logger.error(f"[Worker:{chat_id}] regen ingest error: {e}", exc_info=True)
                            continue
                        if reply_mode is not None:
                            regen_last_mode = reply_mode
                            regen_last_msg = msg

                    # If all regen-batch messages were group-skips, fall through
                    # and send the original reply unchanged.
                    if regen_last_mode is not None and regen_last_msg is not None:
                        try:
                            messages = await self.orch._generate_reply(
                                user_message=regen_last_msg["text"],
                                user_id=regen_last_msg["user_id"],
                                chat_id=chat_id,
                                chat_type=regen_last_msg["chat_type"],
                                reply_mode=regen_last_mode,
                                media=regen_last_msg.get("media"),
                            )
                            last_msg = regen_last_msg  # for reply_to
                        except Exception as e:
                            logger.error(f"[Worker:{chat_id}] regen generate error: {e}", exc_info=True)
                            # fall through with original messages

                # Step 7: record reply (once, after regen decision is final)
                try:
                    await self.orch._record_reply(
                        messages=messages,
                        user_id=last_msg["user_id"],
                        chat_id=chat_id,
                        chat_type=last_msg["chat_type"],
                    )
                except Exception as e:
                    logger.error(f"[Worker:{chat_id}] record error: {e}", exc_info=True)

                # Step 8: send
                await self._send_messages(
                    self._bot, chat_id, messages,
                    reply_to=last_msg.get("reply_to"),
                )
        except Exception as e:
            logger.error(f"[Worker:{chat_id}] unhandled error: {e}", exc_info=True)
        finally:
            # Remove self from workers dict so next enqueue can start a fresh worker
            self._chat_workers.pop(chat_id, None)
```

- [ ] **Step 3.7: Run new worker tests**

```bash
python -m pytest tests/test_bot_worker.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 3.8: Commit (infrastructure only, handlers not yet migrated)**

```bash
git add bot.py tests/test_bot_worker.py
git commit -m "feat(bot): add per-chat queue infrastructure

_enqueue_message, _chat_worker, _chat_queues/_chat_workers dicts.
Worker accumulates rapid messages, generates reply, checks staleness
before send, regenerates once if new messages arrived during LLM call."
```

---

## Task 4: Migrate handlers and update handler tests

**Files:**
- Modify: `bot.py` — `_handle_message`, `_handle_media_message`
- Modify: `tests/test_bot_media.py` — update assertions from `handle_message` to `_enqueue_message`

- [ ] **Step 4.1: Update `test_bot_media.py` — replace `handle_message` assertions**

The existing bot factory in `test_bot_media.py` mocks `orch.handle_message`. After migration, the handlers call `_enqueue_message` instead. Update `_make_bot()` and all test assertions.

Replace the `_make_bot()` function:

```python
def _make_bot():
    from bot import VirtualPersonaBot
    orch = MagicMock()
    # handle_message is still on orch for backward compat tests, but handlers
    # no longer call it directly — they call _enqueue_message.
    orch.handle_message = AsyncMock(return_value=[
        {"type": "text", "content": "好漂亮～", "delay": 0}
    ])
    orch.ingest_message = AsyncMock(return_value="direct")
    orch._generate_reply = AsyncMock(return_value=[
        {"type": "text", "content": "好漂亮～", "delay": 0}
    ])
    orch._record_reply = AsyncMock()
    orch.set_proactive_callback = MagicMock()
    orch.config = {"system": {}}
    orch.persona = {"name": "林小晴"}

    cfg = {
        "bot_token": "test:token",
        "admin_user_id": 999,
        "allowed_users": [],
        "allowed_groups": [],
    }
    bot_obj = VirtualPersonaBot(orch, cfg)
    bot_obj.bot_username = "testbot"
    bot_obj.bot_id = 12345
    bot_obj._bot = MagicMock()
    bot_obj._bot.send_message = AsyncMock()
    bot_obj._bot.send_chat_action = AsyncMock()
    bot_obj._bot.send_sticker = AsyncMock()
    return bot_obj
```

Then update each test that previously asserted on `orch.handle_message` to instead capture what was enqueued. Add this helper at the top of each test (or as a shared fixture):

```python
def _capture_enqueue(bot_obj):
    """Replace _enqueue_message with a spy that captures enqueued messages."""
    enqueued = []
    bot_obj._enqueue_message = lambda chat_id, msg: enqueued.append((chat_id, msg))
    return enqueued
```

Update tests:

```python
def test_photo_message_calls_orchestrator_with_media():
    """Photo messages download the file and enqueue message with correct media."""
    bot_obj = _make_bot()
    update = _make_photo_update()
    enqueued = _capture_enqueue(bot_obj)

    fake_bytes = b"\xff\xd8\xff\xe0test_jpeg_bytes"
    expected_b64 = base64.b64encode(fake_bytes).decode()

    async def fake_get_file(file_id):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(fake_bytes))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    msg = enqueued[0][1]
    media = msg.get("media")
    assert media is not None and len(media) == 1
    assert media[0]["mime_type"] == "image/jpeg"
    assert media[0]["base64"] == expected_b64
    assert media[0]["label"] == "photo"


def test_photo_with_caption_passes_text():
    """Caption text is placed in the message_data text field."""
    bot_obj = _make_bot()
    update = _make_photo_update(caption="这是我的猫")
    enqueued = _capture_enqueue(bot_obj)

    async def fake_get_file(file_id):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8\xff"))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert enqueued[0][1]["text"] == "这是我的猫"


def test_photo_download_failure_proceeds_without_media():
    """Download failure enqueues message with media=None (caption present)."""
    bot_obj = _make_bot()
    update = _make_photo_update(caption="看")
    enqueued = _capture_enqueue(bot_obj)

    async def fake_get_file(file_id):
        raise Exception("network error")

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    assert enqueued[0][1].get("media") is None


def test_bot_sender_silently_ignored():
    """Messages from bots are silently dropped (nothing enqueued)."""
    bot_obj = _make_bot()
    update = _make_photo_update(is_bot=True)
    enqueued = _capture_enqueue(bot_obj)
    ctx = MagicMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))
    assert len(enqueued) == 0


def test_mentioned_me_detected_from_caption():
    """mentioned_me=True when caption contains @botusername."""
    bot_obj = _make_bot()
    update = _make_photo_update(caption="@testbot 看这个")
    enqueued = _capture_enqueue(bot_obj)

    async def fake_get_file(file_id):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8"))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert enqueued[0][1]["mentioned_me"] is True
```

For the frame extraction tests (`test_animation_calls_extract_frame`, `test_animation_frame_extraction_failure_proceeds_without_media`, `test_animated_sticker_calls_extract_tgs_frame`, `test_animated_sticker_tgs_failure_proceeds_without_media`): replace the final `bot_obj.orch.handle_message.assert_called_once()` and `call_args.kwargs` pattern with the same `enqueued` capture pattern. E.g.:

```python
def test_animation_calls_extract_frame():
    bot_obj = _make_bot()
    update = _make_animation_update()
    enqueued = _capture_enqueue(bot_obj)
    # ... setup fake_get_file and fake_extract as before ...

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    media = enqueued[0][1].get("media")
    assert media is not None and len(media) == 1
    assert media[0]["mime_type"] == "image/jpeg"
    assert media[0]["base64"] == expected_b64
```

The `test_video_decline_reply_sent` and `test_video_decline_suppressed_in_group_when_not_mentioned` tests do NOT need changes — `_handle_decline_media_message` still sends directly.

The ffmpeg/TGS extraction unit tests (`test_extract_first_frame_returns_none_when_ffmpeg_missing`, `test_extract_tgs_frame_returns_none_when_lottie_missing`) also do NOT need changes.

- [ ] **Step 4.2: Run updated tests — expect all to fail** (handlers still call old path)

```bash
python -m pytest tests/test_bot_media.py -v 2>&1 | head -30
```

Expected: most tests fail because `_enqueue_message` doesn't exist on bot or handlers still call `handle_message`.

- [ ] **Step 4.3: Migrate `_handle_message()` in bot.py**

Replace the tail of `_handle_message` (from `# 注册主动消息回调` to the end of the method) with:

```python
        # Register proactive callback (synchronous, before enqueue)
        async def send_proactive(messages):
            await self._send_messages(self._bot, chat_id, messages)
        self.orch.set_proactive_callback(chat_id, send_proactive)

        # Enqueue for per-chat serial processing
        self._enqueue_message(chat_id, {
            "text": text,
            "user_id": user_id,
            "user_name": user_name,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "group_title": group_title,
            "mentioned_me": mentioned_me,
            "media": None,
            "reply_to": update.message.message_id if chat_type != "private" else None,
        })
```

Remove the old `messages = await self.orch.handle_message(...)` and `await self._send_messages(...)` lines.

- [ ] **Step 4.4: Migrate `_handle_media_message()` in bot.py**

Replace the tail of `_handle_media_message` (from `self.orch.set_proactive_callback(...)` to the end of the method) with:

```python
        # Register proactive callback (synchronous, before enqueue)
        self.orch.set_proactive_callback(
            chat_id,
            lambda msgs, _cid=chat_id: self._send_messages(self._bot, _cid, msgs),
        )

        # Enqueue for per-chat serial processing
        self._enqueue_message(chat_id, {
            "text": caption,
            "user_id": user_id,
            "user_name": user_name,
            "chat_id": chat_id,
            "chat_type": chat_type,
            "group_title": group_title,
            "mentioned_me": mentioned_me,
            "media": media,
            "reply_to": update.message.message_id if chat_type != "private" else None,
        })
```

Remove the old `messages = await self.orch.handle_message(...)` and `await self._send_messages(...)` lines.

- [ ] **Step 4.5: Run updated bot media tests**

```bash
python -m pytest tests/test_bot_media.py -v
```

Expected: all PASS.

- [ ] **Step 4.6: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests PASS. Fix any failures before proceeding.

- [ ] **Step 4.7: Commit**

```bash
git add bot.py tests/test_bot_media.py
git commit -m "feat(bot): migrate message handlers to per-chat queue

_handle_message and _handle_media_message now enqueue messages instead
of directly calling orchestrator. The _chat_worker handles ingestion,
generation, staleness check, regen, and delivery."
```

---

## Task 5: Add `message_accumulation_ms` to config

**Files:**
- Modify: `config.yaml`

- [ ] **Step 5.1: Add config key to config.yaml**

In `config.yaml`, under the `system:` section, add:

```yaml
  message_accumulation_ms: 500  # ms to wait for rapid follow-up messages before generating reply
```

- [ ] **Step 5.2: Commit**

```bash
git add config.yaml
git commit -m "config: add message_accumulation_ms setting (default 500ms)"
```

---

## Verification

- [ ] **Run full test suite one final time**

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/ -v --tb=short
```

Expected: all tests green, no failures.
