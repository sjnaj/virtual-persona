"""
主动发消息引擎 —— 模拟人内心产生"想找人聊"的冲动
"""
import random
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class ProactiveEngine:
    def __init__(self, persona: dict, event_bus):
        self.persona = persona
        self.bus = event_bus
        self.extraversion = persona["personality"]["extraversion"]

        # 来自各Agent的触发信号
        self.pending_triggers: list = []

        # 订阅事件
        self.bus.subscribe("life.notable_event", self._on_notable_event)
        self.bus.subscribe("browser.found_interesting", self._on_interesting_content)
        self.bus.subscribe("inner_state.updated", self._on_inner_state_updated)

        self.last_proactive_time: Optional[datetime] = None
        self.last_message_time: Optional[datetime] = None  # 来自用户的

    async def _on_notable_event(self, event):
        self.pending_triggers.append({
            "type": "life_event",
            "urgency": 0.45,
            "content": event.data.get("shareable", ""),
            "detail": event.data.get("detail", ""),
            "source": "life_simulator",
            "time": datetime.now(),
        })

    async def _on_interesting_content(self, event):
        self.pending_triggers.append({
            "type": "content_share",
            "urgency": 0.55,
            "content": event.data.get("share_text", ""),
            "seen_content": event.data.get("content", ""),
            "reaction": event.data.get("reaction", ""),
            "source": "browser",
            "time": datetime.now(),
        })

    async def _on_inner_state_updated(self, event):
        """将 pending_thoughts 中有明确对象的条目转为 follow_up 触发器（去重）"""
        existing_contents = {
            t["content"] for t in self.pending_triggers if t.get("type") == "follow_up"
        }
        for thought in event.data.get("pending_thoughts", []):
            if thought.get("target_user_id", 0) > 0:
                content = thought.get("content", "")
                if content and content not in existing_contents:
                    self.pending_triggers.append({
                        "type": "follow_up",
                        "urgency": thought.get("urgency", 0.5),
                        "content": content,
                        "target_user_id": thought["target_user_id"],
                        "source": "inner_state",
                        "time": datetime.now(),
                    })
                    existing_contents.add(content)

    def evaluate(
        self,
        emotion_state: dict,
        life_status: dict,
        memory_status: dict,
    ) -> Optional[dict]:
        """
        综合评估是否应该主动发消息。
        返回 trigger dict 或 None。
        """
        now = datetime.now()
        hour = now.hour

        # 深夜和清晨不主动发消息
        if hour < 7 or hour >= 23:
            return None

        # 如果在睡觉/忙，不发
        if life_status.get("is_sleeping"):
            return None

        # 冷却时间：至少间隔30分钟
        if self.last_proactive_time and (now - self.last_proactive_time).seconds < 1800:
            return None

        triggers = list(self.pending_triggers)

        # ===== 情绪驱动的触发 =====
        valence = emotion_state.get("valence", 0)
        attachment = emotion_state.get("attachment", 0.3)
        vulnerability = emotion_state.get("vulnerability", 0.3)

        # 很开心想分享
        if valence > 0.5:
            triggers.append({
                "type": "mood_share",
                "urgency": 0.3 + valence * 0.2,
                "content": "心情不错想找人聊天",
                "source": "emotion",
                "time": now,
            })

        # 难过想找人
        if valence < -0.3 and vulnerability > 0.4:
            triggers.append({
                "type": "seek_comfort",
                "urgency": 0.5 + abs(valence) * 0.3,
                "content": "心情不好想找人说说话",
                "source": "emotion",
                "time": now,
            })

        # ===== 关系驱动：想对方了 =====
        last_msg = memory_status.get("last_message_time")
        if last_msg:
            try:
                last_dt = datetime.fromisoformat(last_msg) if isinstance(last_msg, str) else last_msg
                hours_since = (now - last_dt).total_seconds() / 3600
            except:
                hours_since = 0
        else:
            hours_since = 24  # 从未聊过，视为很久

        if hours_since > 8 and attachment > 0.4:
            triggers.append({
                "type": "missing",
                "urgency": min(0.7, attachment * hours_since / 24),
                "content": "好久没聊了想搭个话",
                "source": "relationship",
                "time": now,
            })

        # ===== 筛选最佳触发 =====
        # 清理过期触发（超过2小时的不再有效）
        triggers = [
            t for t in triggers
            if (now - t["time"]).total_seconds() < 7200
        ]

        if not triggers:
            return None

        best = max(triggers, key=lambda t: t["urgency"])

        # 性格阈值：内向的人阈值更高
        threshold = 0.35 + (1 - self.extraversion) * 0.25

        if best["urgency"] > threshold:
            # 从pending中移除已使用的触发
            if best in self.pending_triggers:
                self.pending_triggers.remove(best)
            self.last_proactive_time = now
            logger.info(f"[Proactive] 触发! type={best['type']} urgency={best['urgency']:.2f}")
            return best

        return None

    def update_last_message_time(self, dt: datetime):
        self.last_message_time = dt