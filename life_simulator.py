"""
生活模拟器 —— 不是预设日程，而是状态驱动的决策
每个 tick 根据当前状态让 AI 决定"她在做什么"
"""
import json
import math
import random
import logging
import dataclasses
import httpx
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
from collections import Counter

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
class YearGaoState:
    location: str = "沙发"   # 沙发/床上/窗台/猫窝/厨房/消失了
    mood: str = "发呆"        # 赖床/玩耍/讨食/发呆/黏人/暴走


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

        self.yearago: YearGaoState = YearGaoState()
        self._yearago_ticks_since_change: int = 0

        self.daily_weather: dict = {}
        self._state_path: Path = Path("data/life_state.json")
        self._load_state()

    async def tick(self):
        now = datetime.now()
        hour = now.hour + now.minute / 60.0

        # 更新年糕状态（每次 tick 都更新，包括睡眠中）
        self._update_yearago(hour)

        # 确保今日天气已获取
        today = now.date().isoformat()
        if self.daily_weather.get("date") != today:
            city = self.persona.get("city", "北京")
            self.daily_weather = await self._fetch_weather(city)

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

        # ── 构建行为多样性提示 ──
        is_weekday = now.weekday() < 5
        blacklist = self._get_activity_blacklist()
        hints = self._get_activity_hints(hour, is_weekday)

        blacklist_text = ""
        if blacklist:
            blacklist_text = f"\n最近这些行为已经出现太多次了，这轮请避免选择：{blacklist}"

        recent_actions = [e.action for e in self.event_log[-6:]]
        weather_str = (
            f"{self.daily_weather['condition']} {self.daily_weather['temp']}°C"
            if self.daily_weather else "未知"
        )
        yearago_str = f"你的猫年糕现在：在{self.yearago.location}{self.yearago.mood}"
        yearago_notable = ""
        if self.yearago.mood in ("讨食", "暴走"):
            yearago_notable = "\n（年糕现在比较闹腾，相关的事情比平时更值得分享）"

        interests_str = "、".join(self.persona.get("interests", []))

        prompt = f"""你是"{self.persona['name']}"的生活模拟器。现在是 {now.strftime("%Y-%m-%d %H:%M")} ({'工作日' if is_weekday else '周末'})。

当前状态：
- 体力：{self.physical.energy:.0f}/100
- 饥饿感：{self.physical.hunger:.0f}/100（越高越饿）
- 位置：{self.location}
- 上一个动作：{self.current_action}
- 今天天气：{weather_str}
- {yearago_str}{yearago_notable}

最近几个动作：{recent_actions}

她的职业：{self.persona.get('occupation', '')}
{"她的兴趣爱好：" + interests_str + chr(10) if interests_str else ""}她的性格：外向{self.persona['personality']['extraversion']:.1f} 自律{self.persona['personality']['conscientiousness']:.1f}

基于以上信息，她接下来15分钟最可能做什么？
要求：
1. 大部分时候是平淡的日常，不要每次都有戏剧性事件
2. 工作日白天应该在上班
3. 考虑饥饿感（>60该吃饭了）、体力（<30会想休息）
4. 偶尔可以有小事件（同事八卦、外卖好吃、猫做了什么等）{blacklist_text}
这个时间段她比较可能做的事情有：{hints}，也可以有其他合理安排。

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

        eating_keywords = ["吃", "外卖", "午饭", "晚饭", "早饭", "餐"]
        if any(kw in self.current_action for kw in eating_keywords):
            self.physical.hunger -= 20
        else:
            self.physical.hunger += 3

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

        logger.debug(
            f"[LifeSim] {now.strftime('%H:%M')} | {self.current_action} @ {self.location}"
            f" | 体力={self.physical.energy:.0f} 饥饿={self.physical.hunger:.0f}"
            f" | 年糕:{self.yearago.mood}"
        )

        # 有效 tick：持久化状态
        self._save_state()

    # ── 状态持久化 ────────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """同步读取持久化状态（__init__ 中调用）"""
        try:
            text = self._state_path.read_text(encoding="utf-8")
            state = json.loads(text)
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning(f"[LifeSim] 加载状态失败，使用默认值: {e}")
            return

        try:
            saved_at = datetime.fromisoformat(state["saved_at"])
            if (datetime.now() - saved_at).total_seconds() > 86400:
                logger.info("[LifeSim] 状态超过 24h，重置")
                return
        except Exception:
            return

        ph = state.get("physical", {})
        self.physical.energy     = float(ph.get("energy",     self.physical.energy))
        self.physical.hunger     = float(ph.get("hunger",     self.physical.hunger))
        self.physical.comfort    = float(ph.get("comfort",    self.physical.comfort))
        self.physical.sleep_debt = float(ph.get("sleep_debt", self.physical.sleep_debt))
        self.physical.clamp()

        self.location       = state.get("location",       self.location)
        self.current_action = state.get("current_action", self.current_action)
        self.current_detail = state.get("current_detail", self.current_detail)
        self.is_sleeping    = state.get("is_sleeping",    self.is_sleeping)
        self.woke_up_today  = state.get("woke_up_today",  self.woke_up_today)
        self.daily_weather  = state.get("daily_weather",  self.daily_weather)

        yg = state.get("yearago")
        if yg:
            try:
                self.yearago = YearGaoState(**yg)
            except Exception:
                pass
        self._yearago_ticks_since_change = state.get("yearago_ticks", 0)

        raw_log = state.get("event_log", [])
        reconstructed = []
        for d in raw_log:
            try:
                reconstructed.append(LifeEvent(**d))
            except Exception:
                pass
        self.event_log = reconstructed

        logger.info(f"[LifeSim] 状态已恢复: {self.current_action} @ {self.location}")

    def _save_state(self) -> None:
        """调度异步保存，不阻塞 tick"""
        import asyncio
        asyncio.create_task(self._do_save())

    async def _do_save(self, path: "Path | None" = None) -> None:
        target = path or self._state_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "physical":       dataclasses.asdict(self.physical),
            "location":       self.location,
            "current_action": self.current_action,
            "current_detail": self.current_detail,
            "is_sleeping":    self.is_sleeping,
            "woke_up_today":  self.woke_up_today,
            "daily_weather":  self.daily_weather,
            "yearago":        dataclasses.asdict(self.yearago),
            "yearago_ticks":  self._yearago_ticks_since_change,
            "event_log":      [dataclasses.asdict(e) for e in self.event_log[-25:]],
            "saved_at":       datetime.now().isoformat(),
        }
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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

    # ── 行为多样性 ────────────────────────────────────────────────────────────

    _ACTIVITY_POOL_WEEKDAY = [
        (7,  9,  "起床洗漱、吃早饭、通勤"),
        (9,  12, "开会、做设计、摸鱼、喝咖啡"),
        (12, 14, "吃午饭、午休、散步"),
        (14, 18, "做设计、改稿、摸鱼、下午茶"),
        (18, 20, "通勤回家、买晚饭、做饭"),
        (20, 24, "看剧、刷手机、练字、洗澡、陪年糕玩、发呆"),
    ]

    _ACTIVITY_POOL_WEEKEND = [
        (7,  10, "睡懒觉、赖床、慢慢起床"),
        (10, 12, "吃早午饭、逛超市、出门买奶茶"),
        (12, 18, "出门逛街、做美甲、看展、朋友聚餐、午睡"),
        (18, 24, "买晚饭、看综艺、刷视频、洗澡、和朋友聊天"),
    ]

    def _get_activity_blacklist(self) -> list:
        if not self.event_log:
            return []
        window = self.event_log[-min(8, len(self.event_log)):]
        counts = Counter(e.action for e in window)
        return [action for action, cnt in counts.items() if cnt >= 2]

    def _get_activity_hints(self, hour: float, is_weekday: bool) -> str:
        pool = self._ACTIVITY_POOL_WEEKDAY if is_weekday else self._ACTIVITY_POOL_WEEKEND
        h = int(hour) % 24
        for start, end, hints in pool:
            if start <= h < end:
                return hints
        return "发呆、刷手机、失眠"

    # ── 年糕状态 ──────────────────────────────────────────────────────────────

    _YEARAGO_MOODS_BY_HOUR = [
        # (start, end, [(mood, weight), ...])  start inclusive, end exclusive, 0–23
        (0,  6,  [("发呆", 5), ("消失了", 4), ("玩耍", 1)]),
        (6,  8,  [("讨食", 6), ("黏人", 3), ("发呆", 1)]),
        (8,  17, [("赖床", 5), ("发呆", 4), ("玩耍", 1)]),
        (17, 19, [("讨食", 6), ("玩耍", 3), ("黏人", 1)]),
        (19, 23, [("玩耍", 4), ("暴走", 3), ("黏人", 2), ("发呆", 1)]),
        (23, 24, [("发呆", 5), ("消失了", 4), ("玩耍", 1)]),
    ]

    _YEARAGO_LOCATION_BY_MOOD = {
        "讨食":   "厨房",
        "赖床":   None,
        "发呆":   None,
        "玩耍":   None,
        "黏人":   None,
        "暴走":   "消失了",
        "消失了": "消失了",
    }

    _YEARAGO_INDOOR_LOCATIONS = ["沙发", "床上", "窗台", "猫窝"]

    # ── 天气 ──────────────────────────────────────────────────────────────────

    _WEATHER_MAPPING = [
        (["sunny", "clear"],           "晴"),
        (["partly cloudy", "partly"],  "多云"),
        (["cloudy", "overcast"],       "阴"),
        (["drizzle", "light rain"],    "小雨"),
        (["heavy rain", "torrential"], "大雨"),
        (["rain", "shower"],           "雨"),
        (["thunderstorm", "thunder"],   "雷阵雨"),
        (["snow", "blizzard"],         "雪"),
        (["fog", "mist", "haze"],      "雾霾"),
    ]

    _SEASON_DEFAULTS = {
        "spring": (range(3, 6),  "多云", 18),
        "summer": (range(6, 9),  "晴热", 32),
        "autumn": (range(9, 12), "晴",   22),
        "winter": ([12, 1, 2],   "阴",   6),
    }

    def _map_weather_condition(self, raw: str) -> str:
        lower = raw.lower()
        for keywords, chinese in self._WEATHER_MAPPING:
            if any(kw in lower for kw in keywords):
                return chinese
        return raw

    def _season_default_weather(self) -> dict:
        month = datetime.now().month
        for _season, (months, condition, temp) in self._SEASON_DEFAULTS.items():
            if month in months:
                return {
                    "condition": condition,
                    "temp": temp,
                    "date": datetime.now().date().isoformat(),
                }
        return {"condition": "晴", "temp": 20, "date": datetime.now().date().isoformat()}

    async def _fetch_weather(self, city: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"https://wttr.in/{city}?format=j1")
                data = resp.json()
                cond_raw = data["current_condition"][0]["weatherDesc"][0]["value"]
                temp = int(data["current_condition"][0]["temp_C"])
                return {
                    "condition": self._map_weather_condition(cond_raw),
                    "temp": temp,
                    "date": datetime.now().date().isoformat(),
                }
        except Exception as e:
            logger.warning(f"[LifeSim] 天气获取失败 ({city}): {e}")
            return self._season_default_weather()

    def _update_yearago(self, hour: float) -> None:
        """每次 tick 更新猫咪状态（纯规则，无 LLM）"""
        self._yearago_ticks_since_change += 1
        if self._yearago_ticks_since_change < 3:
            return
        if random.random() >= 0.35:
            return

        moods, weights = [], []
        h = int(hour) % 24
        for start, end, pool in self._YEARAGO_MOODS_BY_HOUR:
            if start <= h < end:
                for mood, w in pool:
                    moods.append(mood)
                    weights.append(w)
                break
        if not moods:
            moods, weights = ["发呆"], [1]

        new_mood = random.choices(moods, weights=weights)[0]
        fixed_loc = self._YEARAGO_LOCATION_BY_MOOD.get(new_mood)
        new_loc = fixed_loc if fixed_loc else random.choice(self._YEARAGO_INDOOR_LOCATIONS)

        self.yearago = YearGaoState(location=new_loc, mood=new_mood)
        self._yearago_ticks_since_change = 0

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
            "weather": {
                "condition": self.daily_weather.get("condition", "未知"),
                "temp": self.daily_weather.get("temp", 0),
            },
            "yearago": {
                "location": self.yearago.location,
                "mood": self.yearago.mood,
            },
        }