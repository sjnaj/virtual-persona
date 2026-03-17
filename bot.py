"""
增强版 Telegram Bot
支持：多用户私聊 + 群聊 + 管理员命令
"""
import asyncio
import random
import logging
from datetime import datetime

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters,
)
from telegram.constants import ChatAction

logger = logging.getLogger(__name__)


class VirtualPersonaBot:
    def __init__(self, orchestrator, tg_config: dict):
        self.orch = orchestrator
        self.admin_user_id = tg_config["admin_user_id"]
        self.bot_token = tg_config["bot_token"]
        self.bot_username: str = ""  # 启动后填入

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
                if update.message.reply_to_message.from_user.id == (await context.bot.get_me()).id:
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

            try:
                if msg["type"] == "text":
                    await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                    typing_time = len(msg["content"]) * random.uniform(0.06, 0.1)
                    await asyncio.sleep(min(typing_time, 6))
                    await bot.send_message(
                        chat_id=chat_id,
                        text=msg["content"],
                        reply_to_message_id=reply_to if i == 0 else None,
                    )
                elif msg["type"] == "sticker":
                    await bot.send_sticker(chat_id=chat_id, sticker=msg["file_id"])
            except Exception as e:
                logger.error(f"Send message error: {e}")

    # ========== 管理员命令 ==========

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

    async def _cmd_browse(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return
        await update.message.reply_text("🔄 浏览中...")
        await self.orch.browser.browse()
        await update.message.reply_text("✅ 完成")

    async def _cmd_consolidate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user.id != self.admin_user_id:
            return
        await self.orch.memory.consolidate()
        await update.message.reply_text("✅ 记忆整合完成")

    async def _post_init(self, app: Application):
        self.app = app
        me = await app.bot.get_me()
        self.bot_username = me.username or ""
        logger.info(f"[Bot] 我是 @{self.bot_username}")
        await self.orch.start_background_tasks()

    def run(self):
        app = Application.builder().token(self.bot_token).post_init(self._post_init).build()

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

        logger.info("[Bot] 启动...")
        app.run_polling(drop_pending_updates=True)