"""
增强版 Telegram Bot
支持：多用户私聊 + 群聊 + 管理员命令
"""
import asyncio
import base64
import random
import logging
from datetime import datetime

from telegram import Update, BotCommand, BotCommandScopeDefault, BotCommandScopeChat
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters,
)
from telegram.constants import ChatAction
from telegram.error import TimedOut, NetworkError

logger = logging.getLogger(__name__)

_DECLINE_MEDIA = [
    "这个我看不了哎～",
    "视频/动图我这边显示不出来哈哈",
    "嗯这个我打开不了，发图片给我吧",
]


def _retry_on_timeout(func):
    """命令处理器超时时等 2 秒后重试一次"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (TimedOut, NetworkError):
            logger.warning(f"[Bot] {func.__name__} 超时，2 秒后重试...")
            await asyncio.sleep(2)
            return await func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


class VirtualPersonaBot:
    def __init__(self, orchestrator, tg_config: dict):
        self.orch = orchestrator
        self.admin_user_id = tg_config["admin_user_id"]
        self.bot_token = tg_config["bot_token"]
        self.bot_username: str = ""  # 启动后填入
        self.bot_id: int = 0        # 启动后填入，避免每次消息都请求 API

        # 白名单：允许互动的用户/群（留空=允许所有）
        self.allowed_users: set = set(tg_config.get("allowed_users", []))
        self.allowed_groups: set = set(tg_config.get("allowed_groups", []))

        self.app: Application = None

    def _is_allowed(self, user_id: int, chat_id: int, chat_type: str) -> bool:
        """权限检查"""
        if user_id == self.admin_user_id:
            return True
        if chat_type == "private":
            return not self.allowed_users or user_id in self.allowed_users
        else:
            return not self.allowed_groups or chat_id in self.allowed_groups

    async def _extract_first_frame(self, video_bytes: bytes) -> bytes | None:
        """Extract the first frame from an MP4/WebM file as JPEG bytes using ffmpeg.
        Returns None if ffmpeg is unavailable or extraction fails."""
        import os, tempfile
        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                input_path = f.name
            output_path = input_path + "_frame.jpg"
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", input_path,
                "-vframes", "1", "-q:v", "2",
                output_path, "-y",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            if proc.returncode == 0 and os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    return f.read()
            logger.warning(f"[Bot] ffmpeg 返回错误码 {proc.returncode}")
        except FileNotFoundError:
            logger.warning("[Bot] ffmpeg 未安装，无法提取视频帧")
        except Exception as e:
            logger.warning(f"[Bot] 帧提取失败: {e}")
        finally:
            for p in (input_path, output_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        return None

    async def _extract_tgs_frame(self, tgs_bytes: bytes) -> bytes | None:
        """Extract the first frame from a .tgs (gzip Lottie JSON) sticker as PNG bytes.
        Requires python-lottie and pycairo. Returns None if unavailable or on error."""
        import os, tempfile
        try:
            from lottie import parsers, exporters
        except ImportError:
            logger.warning("[Bot] python-lottie 未安装，无法处理 .tgs 贴纸")
            return None

        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tgs", delete=False) as f:
                f.write(tgs_bytes)
                input_path = f.name
            output_path = input_path + "_frame.png"

            def _render():
                animation = parsers.tgs.parse_tgs(input_path)
                exporters.png.export_png(animation, output_path, frame=0)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _render)

            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    return f.read()
            logger.warning("[Bot] TGS 渲染完成但输出文件不存在")
        except Exception as e:
            logger.warning(f"[Bot] TGS 帧提取失败: {e}")
        finally:
            for p in (input_path, output_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        return None

    async def _download_and_encode(self, bot, file_id: str, label: str,
                                    extract_frame: bool = False,
                                    extract_tgs: bool = False) -> dict | None:
        """Download a Telegram file and return a base64 media dict.
        If extract_frame=True, run ffmpeg to get the first video frame as JPEG.
        Returns None on error."""
        try:
            tg_file = await bot.get_file(file_id)
            data = bytes(await tg_file.download_as_bytearray())
            if extract_tgs:
                frame = await self._extract_tgs_frame(data)
                if frame is None:
                    return None
                return {"mime_type": "image/png", "base64": base64.b64encode(frame).decode(), "label": label}
            if extract_frame:
                frame = await self._extract_first_frame(data)
                if frame is None:
                    return None
                return {"mime_type": "image/jpeg", "base64": base64.b64encode(frame).decode(), "label": label}
            mime = "image/webp" if label == "sticker" else "image/jpeg"
            return {"mime_type": mime, "base64": base64.b64encode(data).decode(), "label": label}
        except Exception as e:
            logger.warning(f"[Bot] 媒体下载失败 ({label}): {e}")
            return None

    async def _handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photo, static sticker, animation (first frame), and video sticker (first frame)."""
        if not update.message or not update.effective_user:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type

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

        caption = update.message.caption or ""

        mentioned_me = False
        if self.bot_username and caption:
            if f"@{self.bot_username}" in caption:
                mentioned_me = True
                caption = caption.replace(f"@{self.bot_username}", "").strip()
        if update.message.reply_to_message and update.message.reply_to_message.from_user:
            if update.message.reply_to_message.from_user.id == self.bot_id:
                mentioned_me = True

        file_id = None
        label = "photo"
        extract_frame = False
        extract_tgs = False
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
            label = "photo"
        elif update.message.sticker:
            sticker = update.message.sticker
            if sticker.is_video:
                file_id = sticker.file_id
                label = "sticker"
                extract_frame = True
            elif sticker.is_animated:
                file_id = sticker.file_id
                label = "sticker"
                extract_tgs = True
            else:
                file_id = sticker.file_id
                label = "sticker"
        elif update.message.animation:
            file_id = update.message.animation.file_id
            label = "photo"
            extract_frame = True

        media = None
        if file_id:
            enc = await self._download_and_encode(context.bot, file_id, label,
                                                   extract_frame=extract_frame,
                                                   extract_tgs=extract_tgs)
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
        """Handle video and video_note — decline politely."""
        if not update.message or not update.effective_user:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type

        if not self._is_allowed(user_id, chat_id, chat_type):
            return
        if update.effective_user.is_bot:
            return

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

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """统一消息入口"""
        if not update.message or not update.effective_user:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        chat_type = update.effective_chat.type
        user_name = (
            update.effective_user.first_name or ""
        ) + (
            " " + (update.effective_user.last_name or "") if update.effective_user.last_name else ""
        )
        user_name = user_name.strip() or f"User{user_id}"

        # 权限检查
        if not self._is_allowed(user_id, chat_id, chat_type):
            return

        # 跳过自己的消息（在群里）
        if update.effective_user.is_bot:
            return

        # 获取消息文本
        text = update.message.text
        if not text:
            return

        # 检查是否被@
        mentioned_me = False
        if self.bot_username:
            mentioned_me = f"@{self.bot_username}" in text
            # 去掉@部分，保留内容
            text = text.replace(f"@{self.bot_username}", "").strip()

        # 检查是否被回复
        if update.message.reply_to_message:
            if update.message.reply_to_message.from_user:
                if update.message.reply_to_message.from_user.id == self.bot_id:
                    mentioned_me = True

        # 群聊信息
        group_title = ""
        if chat_type in ("group", "supergroup"):
            group_title = update.effective_chat.title or ""

        logger.info(
            f"[Bot] {'群聊' if chat_type != 'private' else '私聊'} "
            f"| {user_name}({user_id}) | {text[:40]}... "
            f"| mentioned={mentioned_me}"
        )

        # 注册主动消息回调
        async def send_proactive(messages):
            await self._send_messages(context.bot, chat_id, messages)
        self.orch.set_proactive_callback(chat_id, send_proactive)

        # 调用协调器
        messages = await self.orch.handle_message(
            text=text,
            user_id=user_id,
            user_name=user_name,
            chat_id=chat_id,
            chat_type=chat_type,
            group_title=group_title,
            mentioned_me=mentioned_me,
        )

        if messages is None:
            # 群聊中决定不回复
            return

        # 发送消息
        await self._send_messages(context.bot, chat_id, messages,
                                   reply_to=update.message.message_id if chat_type != "private" else None)

    async def _send_messages(self, bot, chat_id: int, messages: list,
                              reply_to: int = None):
        """按延迟逐条发送消息"""
        for i, msg in enumerate(messages):
            delay = msg.get("delay", 1)
            await asyncio.sleep(min(delay, 120))

            for attempt in range(2):
                try:
                    if msg["type"] == "text":
                        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                        typing_time = len(msg["content"]) * random.uniform(0.03, 0.06)
                        await asyncio.sleep(min(typing_time, 4))
                        await bot.send_message(
                            chat_id=chat_id,
                            text=msg["content"],
                            reply_to_message_id=reply_to if i == 0 else None,
                        )
                    elif msg["type"] == "sticker":
                        await bot.send_sticker(chat_id=chat_id, sticker=msg["file_id"])
                    break  # 成功，跳出重试
                except (TimedOut, NetworkError):
                    if attempt == 0:
                        logger.warning("[Bot] 发送超时，2 秒后重试...")
                        await asyncio.sleep(2)
                    else:
                        logger.error("[Bot] 发送超时，重试仍失败，跳过")
                except Exception as e:
                    logger.error(f"Send message error: {e}")
                    break

    # ========== 命令处理 ==========

    @_retry_on_timeout
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """开场白，对管理员额外显示命令列表"""
        user_id = update.effective_user.id
        persona_name = self.orch.persona.get("name", "我")
        persona = self.orch.persona

        welcome = (
            f"哦？你找我啊～ 我是{persona_name}，"
            f"{persona.get('occupation', '一个普通女生')}😊\n"
            f"养了只橘猫叫年糕，有什么想聊的随时找我~"
        )

        if user_id == self.admin_user_id:
            admin_help = (
                "\n\n🔧 管理员命令：\n"
                "/status - 系统状态\n"
                "/relationships - 关系列表\n"
                "/set_role <uid> <角色> <亲密度> - 设定关系\n"
                "/inject_memory <内容> - 注入记忆\n"
                "/allow_user <uid> - 允许用户互动\n"
                "/allow_group [gid] - 允许群互动\n"
                "/addsticker <mood> <tags> - 添加表情包\n"
                "/browse - 强制浏览\n"
                "/consolidate - 强制整合记忆\n"
                "/help - 再次查看此列表"
            )
            await update.message.reply_text(welcome + admin_help)
        else:
            await update.message.reply_text(welcome)

    # ========== 管理员命令 ==========

    @_retry_on_timeout
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return

        status = self.orch.get_full_status()
        life = status["life"]
        emo = status["emotion"]
        mem = status["memory"]

        rels_text = ""
        for r in status.get("relationships", [])[:8]:
            rels_text += f"  {r['name']}: {r['role']}(亲密{r['closeness']}) msgs={r['total_messages']}\n"

        text = (
            f"🧠 系统状态 ({datetime.now().strftime('%H:%M')})\n\n"
            f"📍 {life['current_action']} @ {life['location']}\n"
            f"   体力{life['physical']['energy']} 饥饿{life['physical']['hunger']}\n\n"
            f"💛 {emo['expression_style']['tone']}\n"
            f"   v={emo['valence']} a={emo['arousal']} 依恋={emo['attachment']}\n\n"
            f"💾 记忆: {mem['episodic_count']}情节 {mem['semantic_count']}事实\n"
            f"   缓冲{mem['buffer_size']} 活跃窗口{mem['active_chats']}\n\n"
            f"👥 关系({len(status.get('relationships', []))}):\n{rels_text}"
        )
        await update.message.reply_text(text)

    @_retry_on_timeout
    async def _cmd_relationships(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """查看所有关系详情"""
        if update.effective_user.id != self.admin_user_id:
            return
        rels = self.orch.relationships.list_all()
        if not rels:
            await update.message.reply_text("暂无关系记录")
            return
        lines = []
        for r in rels:
            prof = self.orch.relationships.get(r["user_id"])
            tags = "、".join(prof.personal_tags[:3]) if prof and prof.personal_tags else "-"
            lines.append(
                f"👤 {r['name']} (id:{r['user_id']})\n"
                f"   关系:{r['role']} 亲密:{r['closeness']} 信任:{r['trust']}\n"
                f"   消息:{r['total_messages']} 标签:{tags}"
            )
        await update.message.reply_text("\n\n".join(lines))

    @_retry_on_timeout
    async def _cmd_allow_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """添加允许互动的用户"""
        if update.effective_user.id != self.admin_user_id:
            return
        args = context.args
        if not args:
            await update.message.reply_text("用法: /allow_user <user_id>")
            return
        try:
            uid = int(args[0])
            self.allowed_users.add(uid)
            await update.message.reply_text(f"✅ 已允许用户 {uid}")
        except ValueError:
            await update.message.reply_text("请输入有效的user_id")

    @_retry_on_timeout
    async def _cmd_allow_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """添加允许互动的群"""
        if update.effective_user.id != self.admin_user_id:
            return
        args = context.args
        if not args:
            # 如果在群里发，直接添加当前群
            if update.effective_chat.type in ("group", "supergroup"):
                gid = update.effective_chat.id
                self.allowed_groups.add(gid)
                await update.message.reply_text(f"✅ 已允许本群 {gid}")
                return
            await update.message.reply_text("用法: /allow_group <group_id> 或在群里直接发")
            return
        try:
            gid = int(args[0])
            self.allowed_groups.add(gid)
            await update.message.reply_text(f"✅ 已允许群 {gid}")
        except ValueError:
            await update.message.reply_text("请输入有效的group_id")

    @_retry_on_timeout
    async def _cmd_set_role(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """手动设定关系初始值（引导角色认知）"""
        if update.effective_user.id != self.admin_user_id:
            return
        args = context.args
        if len(args) < 3:
            await update.message.reply_text(
                "用法: /set_role <user_id> <role> <closeness>\n"
                "例如: /set_role 123456 好朋友 0.7"
            )
            return
        try:
            uid = int(args[0])
            role = args[1]
            closeness = float(args[2])
        except (ValueError, IndexError):
            await update.message.reply_text("参数格式错误")
            return

        prof = self.orch.relationships.get_or_create(uid, f"用户{uid}")
        prof.role = role
        prof.closeness = closeness
        prof.trust = closeness * 0.8
        prof.comfort = closeness * 0.9
        self.orch.relationships._save()
        await update.message.reply_text(
            f"✅ 已设定 {prof.display_name} 的关系:\n"
            f"   角色={role} 亲密度={closeness}"
        )

    @_retry_on_timeout
    async def _cmd_inject_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """手动注入记忆（设定前置故事）"""
        if update.effective_user.id != self.admin_user_id:
            return
        if not context.args:
            await update.message.reply_text(
                "用法: /inject_memory <记忆内容>\n"
                "例如: /inject_memory 他上次说他最近在学吉他，还给我看了视频"
            )
            return
        memory_text = " ".join(context.args)
        now_str = datetime.now().isoformat()
        self.orch.memory.semantic.add(
            documents=[memory_text],
            metadatas=[{
                "confidence": 0.9,
                "source_privacy": "private",
                "created": now_str,
                "type": "fact",
                "injected": True,
            }],
            ids=[f"injected_{now_str}"],
        )
        await update.message.reply_text(f"✅ 已注入记忆：{memory_text[:50]}...")

    @_retry_on_timeout
    async def _cmd_addsticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return
        reply = update.message.reply_to_message
        if not reply:
            await update.message.reply_text(
                "回复一个表情包并发送 /addsticker <mood> <标签1> <标签2>..."
            )
            return
        file_id = None
        sticker_type = "sticker"
        description = ""
        if reply.sticker:
            file_id = reply.sticker.file_id
            sticker_type = "animated" if reply.sticker.is_animated else "sticker"
            description = reply.sticker.emoji or ""
        elif reply.animation:
            file_id = reply.animation.file_id
            sticker_type = "animation"
        elif reply.photo:
            file_id = reply.photo[-1].file_id
            sticker_type = "photo"
        if not file_id:
            await update.message.reply_text("请回复表情包/GIF/图片")
            return
        args = context.args or []
        mood = args[0] if args else "neutral"
        tags = args[1:] if len(args) > 1 else []
        sticker = self.orch.stickers.add_sticker(
            file_id=file_id, sticker_type=sticker_type,
            tags=tags, mood=mood, description=description,
        )
        await update.message.reply_text(f"✅ 表情包 {sticker['id']} mood={mood} tags={tags}")

    @_retry_on_timeout
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return
        await update.message.reply_text(
            "🔧 管理员命令：\n\n"
            "/status - 系统状态\n"
            "/relationships - 关系列表\n"
            "/set_role <uid> <角色> <亲密度> - 设定关系\n"
            "/inject_memory <内容> - 注入记忆\n"
            "/allow_user <uid> - 允许用户互动\n"
            "/allow_group [gid] - 允许群互动\n"
            "/addsticker <mood> <tags> - 添加表情包\n"
            "/browse - 强制浏览\n"
            "/consolidate - 强制整合记忆\n"
        )

    @_retry_on_timeout
    async def _cmd_browse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return
        await update.message.reply_text("🔄 浏览中...")
        await self.orch.browser.browse()
        await update.message.reply_text("✅ 完成")

    @_retry_on_timeout
    async def _cmd_consolidate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return
        await self.orch.memory.consolidate()
        await update.message.reply_text("✅ 记忆整合完成")

    async def _post_init(self, app: Application):
        self.app = app
        me = await app.bot.get_me()
        self.bot_username = me.username or ""
        self.bot_id = me.id
        logger.info(f"[Bot] 我是 @{self.bot_username} (id={self.bot_id})")

        # 注册对所有用户可见的命令
        try:
            await app.bot.set_my_commands(
                [BotCommand("start", "开始对话")],
                scope=BotCommandScopeDefault(),
            )
            # 管理员可见的完整命令列表
            admin_commands = [
                BotCommand("start", "开始对话"),
                BotCommand("status", "系统状态"),
                BotCommand("relationships", "关系列表"),
                BotCommand("set_role", "设定关系"),
                BotCommand("inject_memory", "注入记忆"),
                BotCommand("allow_user", "允许用户互动"),
                BotCommand("allow_group", "允许群互动"),
                BotCommand("addsticker", "添加表情包"),
                BotCommand("browse", "强制浏览"),
                BotCommand("consolidate", "整合记忆"),
                BotCommand("help", "管理帮助"),
            ]
            await app.bot.set_my_commands(
                admin_commands,
                scope=BotCommandScopeChat(chat_id=self.admin_user_id),
            )
        except Exception as e:
            logger.warning(f"[Bot] 设置命令列表失败: {e}")

        await self.orch.start_background_tasks()

    async def _error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """全局错误处理，避免超时等网络错误打印完整堆栈"""
        err = context.error
        if isinstance(err, (TimedOut, NetworkError)):
            logger.warning(f"[Bot] 网络错误（已忽略）: {err}")
        else:
            logger.error(f"[Bot] 未处理异常: {err}", exc_info=err)

    async def _post_shutdown(self, app: Application):
        logger.info("[Bot] 关闭中，整合最后的记忆...")
        await self.orch.stop()

    def run(self):
        app = (
            Application.builder()
            .token(self.bot_token)
            .get_updates_connection_pool_size(4)
            .get_updates_pool_timeout(10.0)
            .post_init(self._post_init)
            .post_shutdown(self._post_shutdown)
            .build()
        )

        # 命令处理
        app.add_handler(CommandHandler("start", self._cmd_start))

        # 管理命令
        for cmd, handler in [
            ("status", self._cmd_status),
            ("relationships", self._cmd_relationships),
            ("set_role", self._cmd_set_role),
            ("inject_memory", self._cmd_inject_memory),
            ("allow_user", self._cmd_allow_user),
            ("allow_group", self._cmd_allow_group),
            ("addsticker", self._cmd_addsticker),
            ("help", self._cmd_help),
            ("browse", self._cmd_browse),
            ("consolidate", self._cmd_consolidate),
        ]:
            app.add_handler(CommandHandler(cmd, handler))

        # 所有文本消息
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))

        # 图片 + 静态/video/animated 表情包 + 动图 → 视觉理解（动图/video sticker 截取第一帧，animated tgs 截取第一帧）
        app.add_handler(MessageHandler(
            (filters.PHOTO | filters.Sticker.STATIC | filters.Sticker.VIDEO
             | filters.Sticker.ANIMATED | filters.ANIMATION) & ~filters.COMMAND,
            self._handle_media_message,
        ))
        # 普通视频 / 圆形视频 → 礼貌拒绝
        app.add_handler(MessageHandler(
            (filters.VIDEO | filters.VIDEO_NOTE) & ~filters.COMMAND,
            self._handle_decline_media_message,
        ))

        app.add_error_handler(self._error_handler)

        logger.info("[Bot] 启动...")
        app.run_polling(drop_pending_updates=True)