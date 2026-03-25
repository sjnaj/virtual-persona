# tests/test_bot_media.py
import asyncio
import base64
from unittest.mock import MagicMock, AsyncMock


def _make_bot():
    from bot import VirtualPersonaBot
    orch = MagicMock()
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


def _capture_enqueue(bot_obj):
    """Replace _enqueue_message with a spy that captures enqueued messages."""
    enqueued = []
    bot_obj._enqueue_message = lambda chat_id, msg: enqueued.append((chat_id, msg))
    return enqueued


def _make_photo_update(user_id=1, chat_id=1, chat_type="private",
                       caption=None, is_bot=False, reply_to_bot=False):
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
    enqueued = _capture_enqueue(bot_obj)
    update = _make_photo_update()

    fake_bytes = b"\xff\xd8\xff\xe0test_jpeg_bytes"
    expected_b64 = base64.b64encode(fake_bytes).decode()

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(fake_bytes))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    media = enqueued[0][1]["media"]
    assert media is not None and len(media) == 1
    assert media[0]["mime_type"] == "image/jpeg"
    assert media[0]["base64"] == expected_b64
    assert media[0]["label"] == "photo"


def test_photo_with_caption_passes_text():
    """Caption text is passed as the `text` argument to orchestrator."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_photo_update(caption="这是我的猫")

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8\xff"))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    assert enqueued[0][1]["text"] == "这是我的猫"


def test_photo_download_failure_proceeds_without_media():
    """Download failure calls orchestrator with media=None."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_photo_update(caption="看")

    async def fake_get_file(file_id, **kwargs):
        raise Exception("network error")

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    assert enqueued[0][1]["media"] is None


def test_bot_sender_silently_ignored():
    """Messages from bots are silently dropped."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_photo_update(is_bot=True)
    ctx = MagicMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))
    assert len(enqueued) == 0


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
    enqueued = _capture_enqueue(bot_obj)
    update = _make_photo_update(caption="@testbot 看这个")

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8"))
        return f

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    asyncio.run(bot_obj._handle_media_message(update, ctx))

    assert len(enqueued) == 1
    assert enqueued[0][1]["mentioned_me"] is True


# ===== Frame extraction tests =====

def _make_animation_update(caption=None, chat_type="private"):
    update = MagicMock()
    update.message = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 1
    update.effective_user.is_bot = False
    update.effective_user.first_name = "小明"
    update.effective_user.last_name = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 1
    update.effective_chat.type = chat_type
    update.effective_chat.title = ""
    update.message.photo = None
    update.message.sticker = None
    update.message.animation = MagicMock()
    update.message.animation.file_id = "anim_file_id"
    update.message.caption = caption
    update.message.reply_to_message = None
    update.message.message_id = 44
    return update


def test_animation_calls_extract_frame():
    """Animation messages call _extract_first_frame and send result as jpeg."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_animation_update()

    fake_mp4 = b"\x00\x00\x00\x18ftyp"  # fake MP4 bytes
    fake_jpeg = b"\xff\xd8\xff\xe0extracted_frame"
    expected_b64 = base64.b64encode(fake_jpeg).decode()

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(fake_mp4))
        return f

    async def fake_extract(self_inner, video_bytes):
        return fake_jpeg

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    import bot as bot_module
    original = bot_module.VirtualPersonaBot._extract_first_frame
    bot_module.VirtualPersonaBot._extract_first_frame = fake_extract

    try:
        asyncio.run(bot_obj._handle_media_message(update, ctx))
    finally:
        bot_module.VirtualPersonaBot._extract_first_frame = original

    assert len(enqueued) == 1
    media = enqueued[0][1]["media"]
    assert media is not None and len(media) == 1
    assert media[0]["mime_type"] == "image/jpeg"
    assert media[0]["base64"] == expected_b64


def test_animation_frame_extraction_failure_proceeds_without_media():
    """If frame extraction fails, orchestrator is called with media=None."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_animation_update(caption="看")

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\x00"))
        return f

    async def fake_extract(self_inner, video_bytes):
        return None  # extraction failed

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    import bot as bot_module
    original = bot_module.VirtualPersonaBot._extract_first_frame
    bot_module.VirtualPersonaBot._extract_first_frame = fake_extract

    try:
        asyncio.run(bot_obj._handle_media_message(update, ctx))
    finally:
        bot_module.VirtualPersonaBot._extract_first_frame = original

    assert len(enqueued) == 1
    assert enqueued[0][1]["media"] is None


def test_extract_first_frame_returns_none_when_ffmpeg_missing():
    """_extract_first_frame returns None gracefully when ffmpeg is not found."""
    bot_obj = _make_bot()

    async def run_test():
        import unittest.mock as um
        with um.patch("asyncio.create_subprocess_exec",
                      side_effect=FileNotFoundError("ffmpeg not found")):
            return await bot_obj._extract_first_frame(b"\x00\x01\x02")

    result = asyncio.run(run_test())
    assert result is None


# ===== TGS animated sticker tests =====

def _make_animated_sticker_update(caption=None, chat_type="private"):
    update = MagicMock()
    update.message = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 1
    update.effective_user.is_bot = False
    update.effective_user.first_name = "小明"
    update.effective_user.last_name = None
    update.effective_chat = MagicMock()
    update.effective_chat.id = 1
    update.effective_chat.type = chat_type
    update.effective_chat.title = ""
    update.message.photo = None
    sticker = MagicMock()
    sticker.file_id = "tgs_file_id"
    sticker.is_animated = True
    sticker.is_video = False
    update.message.sticker = sticker
    update.message.animation = None
    update.message.caption = caption
    update.message.reply_to_message = None
    update.message.message_id = 45
    return update


def test_animated_sticker_calls_extract_tgs_frame():
    """Animated sticker (.tgs) messages call _extract_tgs_frame and return png."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_animated_sticker_update()

    fake_tgs = b"\x1f\x8b\x08fake_tgs_data"
    fake_png = b"\x89PNG\r\nfirst_frame"
    expected_b64 = base64.b64encode(fake_png).decode()

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(fake_tgs))
        return f

    async def fake_extract_tgs(self_inner, tgs_bytes):
        return fake_png

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    import bot as bot_module
    original = bot_module.VirtualPersonaBot._extract_tgs_frame
    bot_module.VirtualPersonaBot._extract_tgs_frame = fake_extract_tgs

    try:
        asyncio.run(bot_obj._handle_media_message(update, ctx))
    finally:
        bot_module.VirtualPersonaBot._extract_tgs_frame = original

    assert len(enqueued) == 1
    media = enqueued[0][1]["media"]
    assert media is not None and len(media) == 1
    assert media[0]["mime_type"] == "image/png"
    assert media[0]["base64"] == expected_b64
    assert media[0]["label"] == "sticker"


def test_animated_sticker_tgs_failure_proceeds_without_media():
    """If TGS extraction fails, orchestrator is called with media=None."""
    bot_obj = _make_bot()
    enqueued = _capture_enqueue(bot_obj)
    update = _make_animated_sticker_update(caption="看")

    async def fake_get_file(file_id, **kwargs):
        f = MagicMock()
        f.download_as_bytearray = AsyncMock(return_value=bytearray(b"\x00"))
        return f

    async def fake_extract_tgs(self_inner, tgs_bytes):
        return None

    ctx = MagicMock()
    ctx.bot.get_file = fake_get_file
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()

    import bot as bot_module
    original = bot_module.VirtualPersonaBot._extract_tgs_frame
    bot_module.VirtualPersonaBot._extract_tgs_frame = fake_extract_tgs

    try:
        asyncio.run(bot_obj._handle_media_message(update, ctx))
    finally:
        bot_module.VirtualPersonaBot._extract_tgs_frame = original

    assert len(enqueued) == 1
    assert enqueued[0][1]["media"] is None


def test_extract_tgs_frame_returns_none_when_lottie_missing():
    """_extract_tgs_frame returns None gracefully when python-lottie is not installed."""
    bot_obj = _make_bot()

    async def run_test():
        import unittest.mock as um
        import sys
        # Simulate lottie not installed by patching the import
        with um.patch.dict(sys.modules, {"lottie": None,
                                          "lottie.parsers": None,
                                          "lottie.exporters": None}):
            return await bot_obj._extract_tgs_frame(b"\x1f\x8b\x08fake")

    result = asyncio.run(run_test())
    assert result is None
