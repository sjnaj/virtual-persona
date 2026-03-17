"""
多维情绪系统 —— 带有惯性的情绪状态
"""
import math
import random
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    valence: float = 0.2      # -1(不开心) ~ +1(开心)
    arousal: float = 0.0      # -1(平静/低落) ~ +1(兴奋)
    irritability: float = 0.1 # 烦躁阈值
    vulnerability: float = 0.3 # 脆弱程度
    attachment: float = 0.3    # 对用户的依恋

    def clamp(self):
        self.valence = max(-1, min(1, self.valence))
        self.arousal = max(-1, min(1, self.arousal))
        self.irritability = max(0, min(1, self.irritability))
        self.vulnerability = max(0, min(1, self.vulnerability))
        self.attachment = max(0, min(1, self.attachment))


class EmotionEngine:
    INERTIA = 0.75  # 情绪惯性系数

    def __init__(self, persona: dict, event_bus):
        self.persona = persona
        self.bus = event_bus
        self.state = EmotionState()
        self.neuroticism = persona["personality"]["neuroticism"]

        # 订阅事件
        self.bus.subscribe("life.tick", self._on_life_tick)
        self.bus.subscribe("user.message", self._on_user_message)
        self.bus.subscribe("user.long_silence", self._on_user_silence)

    async def _on_life_tick(self, event):
        mood_impact = event.data.get("mood_impact", 0)
        if mood_impact != 0:
            impact_scaled = mood_impact / 10.0
            # 神经质越高，情绪波动越大
            impact_scaled *= (0.7 + self.neuroticism * 0.6)
            self._apply_impact(impact_scaled, impact_scaled * 0.5)

    async def _on_user_message(self, event):
        """收到用户消息时的情绪反应"""
        # 收到消息本身就有轻微正面影响
        self._apply_impact(0.05 + self.state.attachment * 0.1, 0.1)
        # 依恋感微增
        self.state.attachment = min(1.0, self.state.attachment + 0.005)
        self.state.clamp()

    async def _on_user_silence(self, event):
        hours = event.data.get("hours", 0)
        if hours > 12 and self.state.attachment > 0.5:
            self._apply_impact(-0.05, -0.05)

    def _apply_impact(self, valence_delta: float, arousal_delta: float):
        self.state.valence = (
            self.state.valence * self.INERTIA +
            valence_delta * (1 - self.INERTIA)
        ) + valence_delta * 0.3  # 直接冲击分量
        self.state.arousal = (
            self.state.arousal * self.INERTIA +
            arousal_delta * (1 - self.INERTIA)
        ) + arousal_delta * 0.2

        # 持续负面情绪累积烦躁
        if valence_delta < -0.1:
            self.state.irritability += 0.02
        elif valence_delta > 0.1:
            self.state.irritability = max(0, self.state.irritability - 0.01)

        self.state.clamp()

    def apply_conversation_sentiment(self, sentiment: float):
        """对话情感分析后的影响。sentiment: -1~+1"""
        self._apply_impact(sentiment * 0.3, abs(sentiment) * 0.2)

    def passive_decay(self):
        """自然回归中性（每tick调用）"""
        self.state.valence *= 0.98
        self.state.arousal *= 0.95
        self.state.irritability *= 0.97
        self.state.clamp()

    def get_expression_style(self) -> dict:
        v = self.state.valence
        a = self.state.arousal
        irr = self.state.irritability

        if v > 0.4 and a > 0.2:
            return {
                "tone": "开心活泼",
                "message_length": "偏长",
                "emoji_freq": "高",
                "typo_rate": 0.04,
                "response_speed": "fast",
                "sticker_mood": "happy",
            }
        elif v > 0.2:
            return {
                "tone": "正常友好",
                "message_length": "正常",
                "emoji_freq": "中",
                "typo_rate": 0.02,
                "response_speed": "normal",
                "sticker_mood": "neutral_positive",
            }
        elif v > -0.2:
            return {
                "tone": "平淡",
                "message_length": "偏短",
                "emoji_freq": "低",
                "typo_rate": 0.01,
                "response_speed": "normal",
                "sticker_mood": "neutral",
            }
        elif v > -0.5 and irr > 0.4:
            return {
                "tone": "有点烦躁",
                "message_length": "短",
                "emoji_freq": "很低",
                "typo_rate": 0.01,
                "response_speed": "slow",
                "sticker_mood": "annoyed",
            }
        else:
            return {
                "tone": "低落",
                "message_length": "很短",
                "emoji_freq": "几乎没有",
                "typo_rate": 0.0,
                "response_speed": "very_slow",
                "sticker_mood": "sad",
            }

    def get_status(self) -> dict:
        style = self.get_expression_style()
        return {
            "valence": round(self.state.valence, 3),
            "arousal": round(self.state.arousal, 3),
            "irritability": round(self.state.irritability, 3),
            "vulnerability": round(self.state.vulnerability, 3),
            "attachment": round(self.state.attachment, 3),
            "expression_style": style,
        }