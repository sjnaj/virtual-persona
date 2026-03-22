"""
增强版协调器 —— 管理多聊天窗口、统一调度
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Callable, Optional

from llm import LLMClient
from event_bus import EventBus
from life_simulator import LifeSimulator
from emotion_engine import EmotionEngine
from memory_hub import MemoryHub
from sticker_engine import StickerEngine
from browser_agent import BrowserAgent
from proactive_engine import ProactiveEngine
from relationship import RelationshipManager
from chat_context import ChatContextManager
from group_behavior import GroupBehaviorEngine
from expression import ExpressionSynthesizer
from inner_state import InnerStateManager

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.persona = config["persona"]
        self.running = False

        # 基础设施
        self.bus = EventBus()
        self.llm = LLMClient(config["llm"])

        # ---- 全局单例 Agent（她只有一个自我）----
        self.life_sim = LifeSimulator(self.persona, self.llm, self.bus)
        self.emotion = EmotionEngine(self.persona, self.bus)
        self.browser = BrowserAgent(
            self.persona, self.llm, self.bus,
            feeds_config=config.get("feeds", []),
        )
        self.stickers = StickerEngine(self.persona, self.llm)

        # ---- 关系层（每个人独立档案）----
        self.relationships = RelationshipManager(self.persona, self.llm, self.bus)

        # ---- 上下文层（每个聊天窗口独立）----
        self.chat_ctx = ChatContextManager(self.persona["name"])

        # ---- 记忆（全局共享但带隐私标记）----
        self.memory = MemoryHub(self.persona, self.llm, self.bus)

        # ---- 群聊行为 ----
        self.group_behavior = GroupBehaviorEngine(
            self.persona, self.llm, self.relationships
        )

        # ---- 主动引擎（需要知道目标用户列表）----
        self.proactive = ProactiveEngine(self.persona, self.bus)

        # ---- 内心独白层 ----
        inner_state_cfg = config.get("system", {})
        self.inner_state = InnerStateManager(
            self.persona, self.llm, self.emotion, self.life_sim, self.bus,
            ttl_hours=inner_state_cfg.get("pending_thought_ttl_hours", 48),
        )

        # ---- 表达层 ----
        self.expression = ExpressionSynthesizer(
            self.persona, self.llm, self.memory, self.emotion,
            self.life_sim, self.stickers, self.browser,
            self.relationships, self.chat_ctx,
            inner_state=self.inner_state,
        )

        # 主动消息回调: chat_id → callback
        self.proactive_callbacks: Dict[int, Callable] = {}

        # 后台任务句柄（用于 stop 时取消）
        self._tasks: list = []

    def set_proactive_callback(self, chat_id: int, callback: Callable):
        self.proactive_callbacks[chat_id] = callback

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
        # 1. 更新关系档案
        rel_prof = self.relationships.get_or_create(user_id, user_name)

        # 2. 更新聊天窗口
        window = self.chat_ctx.get_or_create(chat_id, chat_type, group_title)
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

        # 4. 通知情绪系统
        await self.bus.emit("user.message", {"text": text, "user_id": user_id})
        self.proactive.update_last_message_time(datetime.now())

        # 5. 群聊行为决策
        reply_mode = "direct"
        if chat_type in ("group", "supergroup"):
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

        # 6. 生成回复
        messages = await self.expression.compose_reply(
            user_message=text,
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            reply_mode=reply_mode,
            media=media,
        )

        # 7. 记录自己的回复
        for msg in messages:
            if msg["type"] == "text":
                self.memory.add_message(
                    chat_id, "assistant", msg["content"],
                    context_type="group" if chat_type in ("group", "supergroup") else "private",
                )
                self.chat_ctx.add_message(
                    chat_id, 0, self.persona["name"],
                    msg["content"], is_me=True,
                )

        # 8. 更新关系
        # 简单情感分析（后续可以用更精细的）
        sentiment = 0.1  # 默认微正
        recent_convo = self.chat_ctx.get_recent_context(chat_id, limit=20)
        await self.relationships.update_after_conversation(user_id, recent_convo, sentiment)

        return messages

    async def handle_proactive_trigger(self):
        """检查是否应该主动发消息"""
        trigger = self.proactive.evaluate(
            emotion_state=self.emotion.get_status(),
            life_status=self.life_sim.get_status(),
            memory_status=self.memory.get_status(),
        )
        if not trigger:
            return

        # 选择发给谁：follow_up 发给指定对象，其他触发器发给最亲近的人
        if trigger.get("type") == "follow_up":
            target_user_id = trigger.get("target_user_id", 0)
            target_chat_id = target_user_id   # 私聊 chat_id == user_id
            if not target_chat_id or target_chat_id not in self.proactive_callbacks:
                return  # 目标未注册，跳过
        else:
            all_rels = self.relationships.list_all()
            if not all_rels:
                return
            target = all_rels[0]
            target_chat_id = target["user_id"]

        callback = self.proactive_callbacks.get(target_chat_id)
        if not callback:
            return

        messages = await self.expression.compose_proactive(
            trigger, chat_id=target_chat_id, user_id=target_chat_id,
        )

        for msg in messages:
            if msg["type"] == "text":
                self.memory.add_message(
                    target_chat_id, "assistant", msg["content"],
                    context_type="private",
                )

        await callback(messages)

        if trigger.get("type") == "follow_up":
            await self.bus.emit("proactive.follow_up_fired", {
                "thought_content": trigger.get("content", ""),
            })

    # ========== 后台循环 ==========

    async def _life_tick_loop(self):
        interval = self.config.get("system", {}).get("life_tick_seconds", 900)
        while self.running:
            try:
                await self.life_sim.tick()
                self.emotion.passive_decay()
            except Exception as e:
                logger.error(f"Life tick error: {e}", exc_info=True)
            await asyncio.sleep(interval)

    async def _browse_loop(self):
        interval = self.config.get("system", {}).get("browse_interval_seconds", 3600)
        while self.running:
            try:
                life_status = self.life_sim.get_status()
                if await self.browser.should_browse(life_status):
                    await self.browser.browse()
            except Exception as e:
                logger.error(f"Browse error: {e}", exc_info=True)
            await asyncio.sleep(interval)

    async def _proactive_loop(self):
        interval = self.config.get("system", {}).get("proactive_check_seconds", 300)
        while self.running:
            try:
                await self.handle_proactive_trigger()
            except Exception as e:
                logger.error(f"Proactive error: {e}", exc_info=True)
            await asyncio.sleep(interval)

    async def _memory_loop(self):
        hours = self.config.get("system", {}).get("memory_consolidation_hours", 6)
        interval = hours * 3600
        while self.running:
            await asyncio.sleep(interval)
            try:
                await self.memory.consolidate()
                await self.memory.forget_and_distort()
            except Exception as e:
                logger.error(f"Memory error: {e}", exc_info=True)

    async def _inner_state_loop(self):
        import random
        cfg = self.config.get("system", {})
        interval_range = cfg.get("inner_state_interval_hours", [2, 3])
        while self.running:
            hours = random.uniform(interval_range[0], interval_range[1])
            await asyncio.sleep(hours * 3600)
            try:
                await self.inner_state.generate_monologue()
            except Exception as e:
                logger.error(f"Inner state error: {e}", exc_info=True)

    async def _conversation_end_check_loop(self):
        """定期检查是否有会话结束，触发记忆整合"""
        while self.running:
            for chat_id in list(self.chat_ctx.windows.keys()):
                if self.chat_ctx.check_conversation_ended(chat_id):
                    await self.memory.consolidate(chat_id)
            await asyncio.sleep(120)

    async def start_background_tasks(self):
        self.running = True
        self._tasks = [
            asyncio.create_task(self._life_tick_loop()),
            asyncio.create_task(self._browse_loop()),
            asyncio.create_task(self._proactive_loop()),
            asyncio.create_task(self._memory_loop()),
            asyncio.create_task(self._conversation_end_check_loop()),
            asyncio.create_task(self._inner_state_loop()),
        ]
        logger.info("[Orchestrator] 后台任务已启动")

    async def stop(self):
        self.running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.memory.consolidate()

    def get_full_status(self):
        return {
            "life": self.life_sim.get_status(),
            "emotion": self.emotion.get_status(),
            "memory": self.memory.get_status(),
            "browser": self.browser.get_status(),
            "relationships": self.relationships.list_all(),
            "active_chats": len(self.chat_ctx.windows),
        }