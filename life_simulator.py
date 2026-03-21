"""
生活模拟器 —— 不是预设日程，而是状态驱动的决策
每个 tick 根据当前状态让 AI 决定"她在做什么"
"""
import math
import random
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PhysicalState:
    energy: float = 80.0
    hunger: float = 20.0
    comfort: float = 70.0
    sleep_debt: float = 0.0

    def clamp(self):
        self.energy = max(0, min(100, self.energy))
        self.hunger = max(0, min(100, self.hunger))
        self.comfort = max(0, min(100, self.comfort))
        self.sleep_debt = max(0, min(100, self.sleep_debt))


@dataclass
class LifeEvent:
    time: str
    action: str
    location: str
    detail: str
    mood_impact: float
    energy_change: float
    notable: bool
    shareable_thought: Optional[str]


class LifeSimulator:
    def __init__(self, persona: dict, llm, event_bus):
        self.persona = persona
        self.llm = llm
        self.bus = event_bus

        self.physical = PhysicalState()
        self.location: str = "家"
        self.current_action: str = "睡觉"
        self.current_detail: str = ""
        self.is_sleeping: bool = True
        self.woke_up_today: bool = False

        self.event_log: List[LifeEvent] = []
        self.today_summary: List[str] = []

    async def tick(self):
        now = datetime.now()
        hour = now.hour + now.minute / 60.0

        # 基础生理消耗
        self._passive_updates(hour)

        # 如果在睡觉，判断是否该醒
        if self.is_sleeping:
            wake_range = self.persona.get("daily_patterns", {}).get("wake_up", [7, 8])
            if hour >= wake_range[0] + random.uniform(0, wake_range[1] - wake_range[0]):
                if not self.woke_up_today:
                    self.is_sleeping = False
                    self.woke_up_today = True
                    self.physical.energy = 60 + random.uniform(0, 30) - self.physical.sleep_debt
                    self.physical.energy = max(30, self.physical.energy)
                    self.current_action = "刚醒，赖床中"
                    self.location = "家"
                    logger.info(f"[LifeSim] 起床了，体力={self.physical.energy:.0f}")
                    await self.bus.emit("life.woke_up", {})
            else:
                return  # 还在睡，什么都不做

        # 到了深夜，判断是否该睡
        sleep_range = self.persona.get("daily_patterns", {}).get("sleep", [23, 25])
        sleep_hour = sleep_range[0] + random.uniform(0, sleep_range[1] - sleep_range[0])
        # sleep_hour 可能超过 24（如 25 表示次日凌晨 1 点），需要折算
        if sleep_hour >= 24:
            should_sleep = hour >= 23 or hour < (sleep_hour - 24)
        else:
            should_sleep = hour >= sleep_hour or hour < 5
        if should_sleep and not self.is_sleeping:
            if self.physical.energy < 30 or should_sleep:
                self.is_sleeping = True
                self.woke_up_today = False
                self.current_action = "睡觉"
                self.location = "家"
                await self.bus.emit("life.sleeping", {})
                logger.info("[LifeSim] 睡了")
                return

        # 正常tick：让AI决定她在做什么
        recent_actions = [e.action for e in self.event_log[-6:]]

        prompt = f"""你是"{self.persona['name']}"的生活模拟器。现在是 {now.strftime("%Y-%m-%d %H:%M")} ({'工作日' if now.weekday() < 5 else '周末'})。

当前状态：
- 体力：{self.physical.energy:.0f}/100
- 饥饿感：{self.physical.hunger:.0f}/100（越高越饿）
- 位置：{self.location}
- 上一个动作：{self.current_action}
- 天气：晴 22°C（简化）

最近几个动作：{recent_actions}

她的职业：{self.persona.get('occupation', '')}
她的性格：外向{self.persona['personality']['extraversion']:.1f} 自律{self.persona['personality']['conscientiousness']:.1f}

基于以上信息，她接下来15分钟最可能做什么？
要求：
1. 大部分时候是平淡的日常，不要每次都有戏剧性事件
2. 工作日白天应该在上班
3. 考虑饥饿感（>60该吃饭了）、体力（<30会想休息）
4. 偶尔可以有小事件（同事八卦、外卖好吃、猫做了什么等）

必须输出JSON：
{{"action":"在做什么","location":"家/公司/通勤/外面","detail":"具体细节（一句话）","mood_impact":0,"energy_change":-2,"notable":false,"shareable_thought":null}}

notable=true表示这件事她可能想跟朋友说。shareable_thought是她想说的话（口语化，可以为null）。mood_impact范围-10到10。energy_change一般是负数（消耗体力），吃饭/休息可以是正数。"""

        result = await self.llm.call_json(prompt, tier="utility")
        if not result:
            return

        # 更新状态
        self.current_action = result.get("action", self.current_action)
        self.current_detail = result.get("detail", "")
        self.location = result.get("location", self.location)
        self.physical.energy += result.get("energy_change", -2)

        # 饥饿感：根据当前行为独立更新，与 notable 无关
        eating_keywords = ["吃", "外卖", "午饭", "晚饭", "早饭", "餐"]
        if any(kw in self.current_action for kw in eating_keywords):
            self.physical.hunger -= 20
        else:
            self.physical.hunger += 3  # 随时间变饿

        self.physical.clamp()

        event = LifeEvent(
            time=now.strftime("%H:%M"),
            action=self.current_action,
            location=self.location,
            detail=self.current_detail,
            mood_impact=result.get("mood_impact", 0),
            energy_change=result.get("energy_change", -2),
            notable=result.get("notable", False),
            shareable_thought=result.get("shareable_thought"),
        )
        self.event_log.append(event)
        if len(self.event_log) > 100:
            self.event_log = self.event_log[-100:]

        # 广播事件
        await self.bus.emit("life.tick", {
            "action": self.current_action,
            "location": self.location,
            "detail": self.current_detail,
            "mood_impact": result.get("mood_impact", 0),
        })

        if event.notable:
            await self.bus.emit("life.notable_event", {
                "action": self.current_action,
                "detail": self.current_detail,
                "shareable": event.shareable_thought,
            })
            logger.info(f"[LifeSim] 值得分享的事件: {event.shareable_thought}")

        logger.debug(f"[LifeSim] {now.strftime('%H:%M')} | {self.current_action} @ {self.location} | 体力={self.physical.energy:.0f} 饥饿={self.physical.hunger:.0f}")

    def _passive_updates(self, hour: float):
        """被动的生理变化"""
        if not self.is_sleeping:
            self.physical.energy -= random.uniform(0.5, 1.5)
            self.physical.hunger += random.uniform(1, 3)
        else:
            self.physical.energy += random.uniform(2, 5)
            self.physical.hunger += random.uniform(0.5, 1)
            self.physical.sleep_debt = max(0, self.physical.sleep_debt - 2)
        self.physical.clamp()

    def get_status(self) -> dict:
        return {
            "current_action": self.current_action,
            "current_detail": self.current_detail,
            "location": self.location,
            "is_sleeping": self.is_sleeping,
            "physical": {
                "energy": round(self.physical.energy),
                "hunger": round(self.physical.hunger),
            },
            "recent_events": [
                {"time": e.time, "action": e.action, "detail": e.detail}
                for e in self.event_log[-5:]
            ],
        }