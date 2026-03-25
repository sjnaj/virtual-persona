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

    async def run():
        bot_obj._enqueue_message(100, _make_msg())
        assert 100 in bot_obj._chat_queues
        assert 100 in bot_obj._chat_workers
        assert not bot_obj._chat_workers[100].done()
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

    call_count = [0]

    async def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            bot_obj._chat_queues[100].put_nowait(_make_msg("等等再说"))
        return [{"type": "text", "content": f"reply_{call_count[0]}", "delay": 0}]

    bot_obj.orch._generate_reply = AsyncMock(side_effect=generate_side_effect)

    async def run():
        bot_obj._enqueue_message(100, _make_msg("你好"))
        await bot_obj._chat_workers[100]

    asyncio.run(run())

    assert bot_obj.orch._generate_reply.await_count == 2
    assert bot_obj.orch._record_reply.await_count == 1

    recorded_msgs = bot_obj.orch._record_reply.call_args.kwargs["messages"]
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

    ingest_count = [0]
    async def ingest_side_effect(**kwargs):
        ingest_count[0] += 1
        return "direct" if ingest_count[0] == 1 else None

    bot_obj.orch.ingest_message = AsyncMock(side_effect=ingest_side_effect)

    async def run():
        bot_obj._enqueue_message(100, _make_msg("私聊消息"))
        await bot_obj._chat_workers[100]

    asyncio.run(run())

    assert bot_obj.orch._generate_reply.await_count == 1
    recorded_msgs = bot_obj.orch._record_reply.call_args.kwargs["messages"]
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

    async def slow_generate(**kwargs):
        await asyncio.sleep(0.05)
        return [{"type": "text", "content": "hi", "delay": 0}]

    bot_obj.orch._generate_reply = AsyncMock(side_effect=slow_generate)

    async def run():
        bot_obj._enqueue_message(100, _make_msg("M1"))
        first_task = bot_obj._chat_workers[100]
        bot_obj._enqueue_message(100, _make_msg("M2"))
        second_task = bot_obj._chat_workers[100]
        assert first_task is second_task
        await first_task

    asyncio.run(run())
