# Design: Regenerate Reply When New Message Arrives Before Send

**Date:** 2026-03-24
**Status:** Approved (v3 — post spec-review round 2)

## Problem

Currently the bot processes each message independently and concurrently. If a user sends a second message while the bot is generating a reply to the first, the bot may:
1. Send a reply that ignores the new message entirely
2. Generate two separate replies that may be incoherent

## Goal

Before sending a reply, if new messages have arrived from the user since generation started, incorporate those messages and regenerate the reply once, then send.

## Design

### Scope of Changes

Two files: `orchestrator.py` and `bot.py`.

---

### orchestrator.py

#### New method: `ingest_message()`

Extracts the state-update portion of `handle_message` (steps 1–5) into a standalone method that can be called without triggering LLM generation.

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
```

**Returns:** `reply_mode` string (`"direct"`, `"piggyback"`, etc.) — or `None` if this is a group chat and the group behavior engine decided not to reply.

**Does:**
1. `relationships.get_or_create(user_id, user_name)`
2. `chat_ctx.get_or_create(chat_id, chat_type, group_title)` + `add_message(...)`
3. `memory.add_message(...)`
4. `bus.emit("user.message", ...)` + `proactive.update_last_message_time(...)`
5. Group behavior decision (if applicable)

**Does NOT do:** LLM generation, reply recording, relationship update post-conversation.

#### New private method: `_generate_reply()`

Generates the reply via the expression layer but does **not** record it into context or memory.

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
```

**Does:** calls `expression.compose_reply(...)` only.
**Returns:** list of message dicts.

#### New private method: `_record_reply()`

Records the bot's reply into memory and chat context, and updates the relationship.

```python
async def _record_reply(
    self,
    messages: list,
    user_id: int,
    chat_id: int,
    chat_type: str,
) -> None:
```

**Does:**
- For each `text` message: `memory.add_message(...)` + `chat_ctx.add_message(...)`
- `relationships.update_after_conversation(...)`

**Separation rationale:** By splitting generation from recording, the worker can defer recording until it has confirmed no regen is needed. This prevents phantom bot replies (a generated-but-not-sent reply) from polluting memory and chat context.

#### Refactored `handle_message()`

Becomes a thin wrapper that preserves full backward compatibility:

```python
async def handle_message(...) -> Optional[list]:
    reply_mode = await self.ingest_message(...)
    if reply_mode is None:
        return None
    messages = await self._generate_reply(...)
    await self._record_reply(messages, ...)
    return messages
```

---

### bot.py

#### Bot instance

During `_post_init`, store the bot instance for use by workers:
```python
self._bot = app.bot  # stored once, stable for the lifetime of the Application
```

This avoids threading the `Bot` object through every queued message dict.

#### New data structures on `VirtualPersonaBot`

Initialize in `__init__` (not `_post_init`) so they exist before any handler could theoretically fire:

```python
self._chat_queues: Dict[int, asyncio.Queue] = {}   # unbounded — intentional
self._chat_workers: Dict[int, asyncio.Task] = {}
```

The queue is unbounded (`asyncio.Queue()` with no maxsize). In flood scenarios, older messages simply queue up and are processed in the next delivery cycle after the one-regen limit is exhausted; this is acceptable.

#### Message data dict format

```python
{
    "text": str,
    "user_id": int,
    "user_name": str,
    "chat_id": int,
    "chat_type": str,
    "group_title": str,
    "mentioned_me": bool,
    "media": list | None,
    "reply_to": int | None,  # group chats only: original message_id to quote-reply
}
```

#### New method: `_enqueue_message(chat_id, message_data)`

```python
def _enqueue_message(self, chat_id: int, message_data: dict) -> None:
    if chat_id not in self._chat_queues:
        self._chat_queues[chat_id] = asyncio.Queue()
    self._chat_queues[chat_id].put_nowait(message_data)

    # Check-and-create must be synchronous (no await between check and create)
    # to avoid a race where two handlers both see a done task and spawn two workers.
    existing = self._chat_workers.get(chat_id)
    if existing is None or existing.done():
        task = asyncio.create_task(self._chat_worker(chat_id))
        self._chat_workers[chat_id] = task
```

#### New coroutine: `_chat_worker(chat_id)`

```python
async def _chat_worker(self, chat_id: int) -> None:
    queue = self._chat_queues[chat_id]
    try:
        while not queue.empty():
            # --- Step 1: dequeue first message ---
            first = queue.get_nowait()
            batch = [first]

            # --- Step 2: accumulation window ---
            # Collects rapid follow-up messages sent within the window.
            # Value can be tuned via config.yaml system.message_accumulation_ms (default 500).
            acc_ms = self.orch.config.get("system", {}).get("message_accumulation_ms", 500)
            await asyncio.sleep(acc_ms / 1000)

            # --- Step 3: drain remaining items from queue ---
            while not queue.empty():
                batch.append(queue.get_nowait())

            # --- Step 4: ingest all messages in batch ---
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
                    logger.error(f"[Worker:{chat_id}] ingest_message error: {e}", exc_info=True)
                    continue
                if reply_mode is not None:
                    last_reply_mode = reply_mode
                    last_msg = msg

            # If all messages were group-skips, nothing to generate; loop
            if last_reply_mode is None or last_msg is None:
                continue

            # --- Step 5: generate reply (NOT recorded yet) ---
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
                logger.error(f"[Worker:{chat_id}] _generate_reply error: {e}", exc_info=True)
                continue

            # --- Step 6: pre-send staleness check (one regen only) ---
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

                # If all regen-batch messages were group-skips (all returned None),
                # fall through and send the original reply unchanged.
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
                        last_msg = regen_last_msg  # update for reply_to selection
                    except Exception as e:
                        logger.error(f"[Worker:{chat_id}] regen generate error: {e}", exc_info=True)
                        # fall through with original messages

            # --- Step 7: record reply (only once, after regen decision is final) ---
            try:
                await self.orch._record_reply(
                    messages=messages,
                    user_id=last_msg["user_id"],
                    chat_id=chat_id,
                    chat_type=last_msg["chat_type"],
                )
            except Exception as e:
                logger.error(f"[Worker:{chat_id}] _record_reply error: {e}", exc_info=True)

            # --- Step 8: send ---
            # reply_to: use the last message's reply_to (most recent context for group quote)
            await self._send_messages(
                self._bot, chat_id, messages,
                reply_to=last_msg.get("reply_to"),
            )
    except Exception as e:
        logger.error(f"[Worker:{chat_id}] unhandled error: {e}", exc_info=True)
    finally:
        # Clean up worker entry so the next enqueue can start a fresh worker
        self._chat_workers.pop(chat_id, None)
```

**Outer loop note:** After Step 8 (send), the `while not queue.empty()` loop continues. If messages arrived during `_send_messages` (which includes typing delays), they are processed in the next iteration naturally — no special handling needed.

**`reply_to` policy:** The `reply_to` quote-reply message ID is taken from `last_msg` — the last message that produced a non-None `reply_mode`. After regen, `last_msg` is updated to the regen batch's last message, so the quote-reply points to the most recent message that triggered the reply. This is the most coherent behavior for group chats.

#### Modified `_handle_message()` and `_handle_media_message()`

Replace the direct `await orch.handle_message(...) + await _send_messages(...)` pattern:

```python
# Register proactive callback BEFORE enqueue (must remain synchronous through enqueue)
self.orch.set_proactive_callback(chat_id, ...)

# Build and enqueue
message_data = {
    "text": text,
    "user_id": user_id,
    "user_name": user_name,
    "chat_id": chat_id,
    "chat_type": chat_type,
    "group_title": group_title,
    "mentioned_me": mentioned_me,
    "media": media,
    "reply_to": update.message.message_id if chat_type != "private" else None,
}
self._enqueue_message(chat_id, message_data)
```

`_handle_decline_media_message` remains a direct send — it bypasses the orchestrator entirely and sends a hardcoded text. Because it does not go through `ingest_message` it cannot conflict with worker state. Interleaving with an active worker is acceptable since it is a standalone refuse-message.

---

## Data Flow: Normal and Regen Cases

```
Case 1: two rapid messages (batched in accumulation window)

  User sends M1 → enqueue(M1) → worker starts
  User sends M2 (within 500ms) → enqueue(M2)

  Worker:
    dequeue M1
    wait 500ms → drain → batch=[M1, M2]
    ingest(M1)   → reply_mode="direct", last_msg=M1
    ingest(M2)   → reply_mode="direct", last_msg=M2
    _generate_reply(user_message=M2.text) → reply_R  [uses context with M1+M2]
    queue empty → skip regen
    _record_reply(reply_R)
    send reply_R


Case 2: new message arrives during LLM generation (regen triggered)

  User sends M1 → enqueue(M1) → worker starts
  Worker: dequeue M1, wait 500ms, ingest(M1), starts _generate_reply [takes 2s]
  User sends M2 (during LLM call) → enqueue(M2)
  Worker: _generate_reply returns reply_R
    check queue → M2 found!
    ingest(M2)   → reply_mode="direct", regen_last_msg=M2
    _generate_reply(user_message=M2.text) → reply_R2  [context now has M1+M2]
    last_msg = M2
    _record_reply(reply_R2)   ← reply_R is never recorded
    send reply_R2
```

---

## Configuration

Add to `config.yaml` under `system`:
```yaml
system:
  message_accumulation_ms: 500  # wait window for batching rapid messages
```

---

## What Does NOT Change

- `expression.py`, `memory_hub.py`, `emotion_engine.py`, `relationship.py` — no changes
- `orchestrator.handle_message` public signature — unchanged, all existing callers work
- Proactive message handling — unchanged
- Admin commands — unchanged
- Group chat logic — still respected per message in `ingest_message`
- `_handle_decline_media_message` — remains a direct send

---

## Constraints

- Max one regeneration per delivery cycle
- 500ms accumulation window (configurable)
- Per-chat workers are independent; cross-chat state is unaffected
- Memory/emotion/relationship receive ALL messages regardless of regeneration
- Reply is recorded exactly once — always the final generated version
