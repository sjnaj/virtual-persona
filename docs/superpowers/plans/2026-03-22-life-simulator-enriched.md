# LifeSimulator 丰富化实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 LifeSimulator 重启丢状态和刷小红书死循环两个 Bug，并增强行为多样性、真实天气感知和猫咪年糕互动。

**Architecture:** 所有改动限定在 `life_simulator.py` 单文件内，新增 `YearGaoState` dataclass 和9个新方法，注入进已有的 `tick()` / `get_status()` 流程；另外在 `config.yaml` 追加 `persona.city` 字段。

**Tech Stack:** Python 3.11+, asyncio, dataclasses, pathlib, json, httpx（已安装）, pytest, unittest.mock

**Spec:** `docs/superpowers/specs/2026-03-22-life-simulator-design.md`

---

## 文件结构

| 操作 | 文件 | 改动内容 |
|------|------|---------|
| Modify | `life_simulator.py` | 全部新功能（唯一改动的源码文件） |
| Modify | `config.yaml` | 在 `persona:` 下追加 `city: "上海"` |
| Create | `tests/test_life_simulator_enriched.py` | 本次所有新测试 |

---

## Task 1: YearGaoState dataclass + _update_yearago

**Files:**
- Modify: `life_simulator.py`（在文件顶部 import 区追加所有新 import；在 `PhysicalState` 之后添加 `YearGaoState`；在 `LifeSimulator.__init__` 中初始化；新增 `_update_yearago` 方法）
- Create: `tests/test_life_simulator_enriched.py`

- [ ] **Step 0: 先把所有新 import 加到 `life_simulator.py` 顶部**（后续 Task 都依赖这些）

在 `life_simulator.py` 现有 import 区末尾（`logger = ...` 行之前）追加：

```python
import json
import dataclasses
import httpx
from pathlib import Path
from collections import Counter
```

- [ ] **Step 1: 创建测试文件，写 YearGaoState 和 _update_yearago 的失败测试**

```python
# tests/test_life_simulator_enriched.py
import asyncio
import json
import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


def make_sim():
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {"wake_up": [7, 8], "sleep": [23, 25]},
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    return LifeSimulator(persona, llm, bus)


# ── Task 1: YearGaoState + _update_yearago ──────────────────────────────────

def test_yearago_state_defaults():
    from life_simulator import YearGaoState
    state = YearGaoState()
    assert state.location == "沙发"
    assert state.mood == "发呆"


def test_sim_has_yearago_attribute():
    sim = make_sim()
    assert hasattr(sim, "yearago")
    assert hasattr(sim, "_yearago_ticks_since_change")


def test_update_yearago_increments_tick_counter():
    sim = make_sim()
    sim._yearago_ticks_since_change = 0
    # Does not change (cooldown < 2), but counter must increment
    sim._update_yearago(14.0)
    assert sim._yearago_ticks_since_change == 1


def test_update_yearago_no_change_when_cooldown_below_2():
    sim = make_sim()
    sim._yearago_ticks_since_change = 1  # < 2: must not switch
    original_mood = sim.yearago.mood
    original_loc = sim.yearago.location
    sim._update_yearago(7.0)
    assert sim.yearago.mood == original_mood
    assert sim.yearago.location == original_loc


def test_update_yearago_switches_when_cooldown_met_and_random_allows():
    sim = make_sim()
    sim._yearago_ticks_since_change = 2  # ≥ 2: allowed to switch
    sim.yearago.mood = "发呆"
    # Force random.random() < 0.35 and choices to return "讨食"
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["讨食"]):
        sim._update_yearago(7.0)
    assert sim.yearago.mood == "讨食"
    assert sim.yearago.location == "厨房"
    assert sim._yearago_ticks_since_change == 0  # reset after switch


def test_update_yearago_no_switch_when_random_above_threshold():
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    sim.yearago.mood = "发呆"
    # random.random() > 0.35: do not switch
    with patch("life_simulator.random.random", return_value=0.9):
        sim._update_yearago(7.0)
    assert sim.yearago.mood == "发呆"


def test_update_yearago_location_follows_mood_raoshaof():
    """赖床/发呆 → 室内软家具位置"""
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["赖床"]):
        sim._update_yearago(10.0)
    assert sim.yearago.location in ("沙发", "床上", "猫窝")


def test_update_yearago_location_chaos_on_baozou():
    """暴走 → 消失了"""
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["暴走"]):
        sim._update_yearago(21.0)
    assert sim.yearago.location == "消失了"


def test_update_yearago_night_range_covered():
    """深夜 0–5 时段应该有 mood 候选（不能落入兜底）"""
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["发呆"]):
        sim._update_yearago(2.0)  # 凌晨 2 点
    # 应该切换成功（不是原始默认值 "发呆" 也不是兜底），关键是不报错且 counter 重置
    assert sim._yearago_ticks_since_change == 0
```

- [ ] **Step 2: 运行测试，确认全部失败**

```bash
cd /Users/admin/Work/virtual-persona
python -m pytest tests/test_life_simulator_enriched.py -v 2>&1 | head -40
```

预期：ImportError 或 AttributeError（`YearGaoState` 不存在）

- [ ] **Step 3: 实现 YearGaoState dataclass 和 _update_yearago**

在 `life_simulator.py` 中，在 `LifeEvent` dataclass 之后插入：

```python
@dataclass
class YearGaoState:
    location: str = "沙发"   # 沙发/床上/窗台/猫窝/厨房/消失了
    mood: str = "发呆"        # 赖床/玩耍/讨食/发呆/黏人/暴走
```

在 `LifeSimulator.__init__` 中，`self.event_log` 初始化之后追加：

```python
        self.yearago: YearGaoState = YearGaoState()
        self._yearago_ticks_since_change: int = 0
```

在 `LifeSimulator` 类末尾（`get_status` 之前）添加：

```python
    # ── 年糕状态 ──────────────────────────────────────────────────────────────

    _YEARAGO_MOODS_BY_HOUR = [
        # (start, end, [(mood, weight), ...])  -- start inclusive, end exclusive, 0–23
        (0,  6,  [("发呆", 5), ("消失了", 4), ("玩耍", 1)]),
        (6,  8,  [("讨食", 6), ("黏人", 3), ("发呆", 1)]),
        (8,  17, [("赖床", 5), ("发呆", 4), ("玩耍", 1)]),
        (17, 19, [("讨食", 6), ("玩耍", 3), ("黏人", 1)]),
        (19, 23, [("玩耍", 4), ("暴走", 3), ("黏人", 2), ("发呆", 1)]),
        (23, 24, [("发呆", 5), ("消失了", 4), ("玩耍", 1)]),
    ]

    _YEARAGO_LOCATION_BY_MOOD = {
        "讨食":  "厨房",
        "赖床":  None,   # 随机选室内
        "发呆":  None,
        "玩耍":  None,
        "黏人":  None,
        "暴走":  "消失了",
        "消失了": "消失了",
    }

    _YEARAGO_INDOOR_LOCATIONS = ["沙发", "床上", "窗台", "猫窝"]

    def _update_yearago(self, hour: float) -> None:
        """每次 tick 更新猫咪状态（纯规则，无 LLM）"""
        self._yearago_ticks_since_change += 1
        if self._yearago_ticks_since_change < 2:
            return  # 冷却中，不切换
        if random.random() >= 0.35:
            return  # 概率没到，不切换

        # 选 mood（h 在 0–23 范围内）
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

        # 选 location
        fixed_loc = self._YEARAGO_LOCATION_BY_MOOD.get(new_mood)
        new_loc = fixed_loc if fixed_loc else random.choice(self._YEARAGO_INDOOR_LOCATIONS)

        self.yearago = YearGaoState(location=new_loc, mood=new_mood)
        self._yearago_ticks_since_change = 0
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "yearago"
```

预期：全部 PASS

- [ ] **Step 5: 确保现有测试不受影响**

```bash
python -m pytest tests/test_life_woke_up.py -v
```

预期：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py
git commit -m "feat: add YearGaoState dataclass and _update_yearago to LifeSimulator"
```

---

## Task 2: 天气辅助方法

**Files:**
- Modify: `life_simulator.py`（在文件顶部 import 区新增 `import httpx`；添加 `_season_default_weather`、`_map_weather_condition`、`_fetch_weather` 方法）
- Modify: `tests/test_life_simulator_enriched.py`（追加天气测试）

- [ ] **Step 1: 追加天气相关失败测试**

在 `tests/test_life_simulator_enriched.py` 末尾追加：

```python
# ── Task 2: 天气辅助方法 ─────────────────────────────────────────────────────

def test_season_default_weather_spring():
    sim = make_sim()
    with patch("life_simulator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 4, 15)
        w = sim._season_default_weather()
    assert w["condition"] == "多云"
    assert w["temp"] == 18
    assert "date" in w


def test_season_default_weather_summer():
    sim = make_sim()
    with patch("life_simulator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 7, 20)
        w = sim._season_default_weather()
    assert w["temp"] == 32


def test_season_default_weather_autumn():
    sim = make_sim()
    with patch("life_simulator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 10, 1)
        w = sim._season_default_weather()
    assert w["condition"] == "晴"
    assert w["temp"] == 22


def test_season_default_weather_winter():
    sim = make_sim()
    with patch("life_simulator.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 1, 10)
        w = sim._season_default_weather()
    assert w["condition"] == "阴"
    assert w["temp"] == 6


def test_map_weather_condition_sunny():
    sim = make_sim()
    assert sim._map_weather_condition("Sunny") == "晴"
    assert sim._map_weather_condition("Clear") == "晴"


def test_map_weather_condition_rain():
    sim = make_sim()
    assert sim._map_weather_condition("Light rain shower") == "小雨"
    assert sim._map_weather_condition("Heavy Rain") == "大雨"


def test_map_weather_condition_unknown_passthrough():
    sim = make_sim()
    assert sim._map_weather_condition("Sandstorm") == "Sandstorm"


def test_fetch_weather_success():
    sim = make_sim()

    async def run():
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "current_condition": [{
                "temp_C": "22",
                "weatherDesc": [{"value": "Sunny"}]
            }]
        }
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=mock_resp)))
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("life_simulator.httpx.AsyncClient", return_value=mock_cm):
            return await sim._fetch_weather("上海")

    result = asyncio.run(run())
    assert result["condition"] == "晴"
    assert result["temp"] == 22
    assert result["date"] == datetime.now().date().isoformat()


def test_fetch_weather_request_error_returns_default():
    import httpx as _httpx
    sim = make_sim()

    async def run():
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(
            side_effect=_httpx.RequestError("timeout")
        )
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("life_simulator.httpx.AsyncClient", return_value=mock_cm):
            return await sim._fetch_weather("上海")

    result = asyncio.run(run())
    assert "condition" in result
    assert "temp" in result


def test_fetch_weather_bad_json_returns_default():
    sim = make_sim()

    async def run():
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"unexpected": "format"}
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=MagicMock(get=AsyncMock(return_value=mock_resp)))
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("life_simulator.httpx.AsyncClient", return_value=mock_cm):
            return await sim._fetch_weather("上海")

    result = asyncio.run(run())
    assert "condition" in result
    assert "temp" in result
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "weather or season or map_weather or fetch_weather"
```

预期：AttributeError（方法不存在）

- [ ] **Step 3: 实现天气方法**

> `import httpx`、`from pathlib import Path`、`from collections import Counter` 已在 Task 1 Step 0 添加，无需重复。

在 `LifeSimulator` 类（`_update_yearago` 方法之后）追加：

```python
    # ── 天气 ──────────────────────────────────────────────────────────────────

    _WEATHER_MAPPING = [
        (["sunny", "clear"],             "晴"),
        (["partly cloudy", "partly"],    "多云"),
        (["cloudy", "overcast"],         "阴"),
        (["drizzle", "light rain"],      "小雨"),
        (["heavy rain", "torrential"],   "大雨"),
        (["rain", "shower"],             "雨"),
        (["thunder", "storm"],           "雷阵雨"),
        (["snow", "blizzard"],           "雪"),
        (["fog", "mist", "haze"],        "雾霾"),
    ]

    _SEASON_DEFAULTS = {
        # (months, condition, temp)
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
        return raw  # 未匹配则原文透传

    def _season_default_weather(self) -> dict:
        month = datetime.now().month
        for season, (months, condition, temp) in self._SEASON_DEFAULTS.items():
            if month in months:
                return {
                    "condition": condition,
                    "temp": temp,
                    "date": datetime.now().date().isoformat(),
                }
        # 兜底（不应到达）
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "weather or season or map_weather or fetch_weather"
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py
git commit -m "feat: add weather fetching helpers to LifeSimulator (wttr.in + seasonal fallback)"
```

---

## Task 3: 行为多样性辅助方法

**Files:**
- Modify: `life_simulator.py`（添加 `_get_activity_blacklist`、`_get_activity_hints` 方法）
- Modify: `tests/test_life_simulator_enriched.py`（追加多样性测试）

- [ ] **Step 1: 追加行为多样性失败测试**

在 `tests/test_life_simulator_enriched.py` 末尾追加：

```python
# ── Task 3: 行为多样性辅助方法 ───────────────────────────────────────────────

from life_simulator import LifeEvent


def _make_event(action: str) -> "LifeEvent":
    return LifeEvent(
        time="10:00", action=action, location="家", detail="",
        mood_impact=0, energy_change=-2, notable=False, shareable_thought=None,
    )


def test_activity_blacklist_empty_on_cold_start():
    sim = make_sim()
    sim.event_log = []
    assert sim._get_activity_blacklist() == []


def test_activity_blacklist_empty_when_no_repeats():
    sim = make_sim()
    sim.event_log = [_make_event(f"动作{i}") for i in range(8)]
    assert sim._get_activity_blacklist() == []


def test_activity_blacklist_catches_action_appearing_twice():
    sim = make_sim()
    sim.event_log = [_make_event("刷小红书")] * 2 + [_make_event("做设计")] * 6
    blacklist = sim._get_activity_blacklist()
    assert "刷小红书" in blacklist


def test_activity_blacklist_only_checks_last_8():
    sim = make_sim()
    # 10 events: first 2 are repeats, but they're outside the last 8 window
    sim.event_log = (
        [_make_event("刷小红书")] * 2 +
        [_make_event(f"动作{i}") for i in range(8)]
    )
    blacklist = sim._get_activity_blacklist()
    assert "刷小红书" not in blacklist


def test_activity_hints_weekday_morning():
    sim = make_sim()
    hints = sim._get_activity_hints(hour=8.0, is_weekday=True)
    assert any(kw in hints for kw in ["通勤", "早饭", "洗漱"])


def test_activity_hints_weekday_work_hours():
    sim = make_sim()
    hints = sim._get_activity_hints(hour=10.0, is_weekday=True)
    assert any(kw in hints for kw in ["设计", "开会", "摸鱼", "咖啡"])


def test_activity_hints_weekday_lunch():
    sim = make_sim()
    hints = sim._get_activity_hints(hour=12.5, is_weekday=True)
    assert any(kw in hints for kw in ["午饭", "午休", "散步"])


def test_activity_hints_weekday_evening():
    sim = make_sim()
    hints = sim._get_activity_hints(hour=21.0, is_weekday=True)
    assert any(kw in hints for kw in ["剧", "洗澡", "年糕", "发呆"])


def test_activity_hints_weekend_midday():
    sim = make_sim()
    hints = sim._get_activity_hints(hour=14.0, is_weekday=False)
    assert any(kw in hints for kw in ["逛街", "午睡", "看展", "聚餐", "美甲"])


def test_activity_hints_returns_nonempty_string():
    sim = make_sim()
    # 任意时段都应返回非空字符串
    for h in [0, 3, 7, 9, 12, 15, 19, 22]:
        assert sim._get_activity_hints(hour=float(h), is_weekday=True)
        assert sim._get_activity_hints(hour=float(h), is_weekday=False)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "blacklist or hints"
```

预期：AttributeError（方法不存在）

- [ ] **Step 3: 实现行为多样性方法**

在 `life_simulator.py` 的 `_fetch_weather` 方法之后追加：

```python
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

    def _get_activity_blacklist(self) -> list[str]:
        """返回近期出现 ≥2 次的 action 列表（供 prompt 禁止）"""
        if not self.event_log:
            return []
        window = self.event_log[-min(8, len(self.event_log)):]
        counts = Counter(e.action for e in window)
        return [action for action, cnt in counts.items() if cnt >= 2]

    def _get_activity_hints(self, hour: float, is_weekday: bool) -> str:
        """返回当前时段适合做的事情（逗号分隔字符串）"""
        pool = self._ACTIVITY_POOL_WEEKDAY if is_weekday else self._ACTIVITY_POOL_WEEKEND
        h = int(hour) % 24
        for start, end, hints in pool:
            if start <= h < end:
                return hints
        # 深夜兜底
        return "发呆、刷手机、失眠"
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "blacklist or hints"
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py
git commit -m "feat: add activity blacklist and time-slot hints helpers to LifeSimulator"
```

---

## Task 4: 状态持久化

**Files:**
- Modify: `life_simulator.py`（`__init__` 中初始化 `_state_path` 和 `daily_weather`；添加 `_load_state`、`_save_state`、`_do_save` 方法）
- Modify: `tests/test_life_simulator_enriched.py`（追加持久化测试）

- [ ] **Step 1: 追加持久化失败测试**

在 `tests/test_life_simulator_enriched.py` 末尾追加：

```python
# ── Task 4: 持久化 ───────────────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    from life_simulator import LifeSimulator, LifeEvent, YearGaoState
    from event_bus import EventBus

    sim = make_sim()
    sim.current_action = "看剧"
    sim.location = "卧室"
    sim.is_sleeping = False
    sim.woke_up_today = True
    sim.physical.energy = 65.0
    sim.physical.hunger = 42.0
    sim.yearago = YearGaoState(location="窗台", mood="发呆")
    sim._yearago_ticks_since_change = 3
    sim.daily_weather = {"condition": "晴", "temp": 25, "date": "2026-03-22"}
    sim.event_log = [
        LifeEvent("10:00", "做设计", "公司", "在改稿", 0, -2, False, None)
    ]

    state_file = tmp_path / "life_state.json"
    sim._state_path = state_file

    asyncio.run(sim._do_save())

    sim2 = make_sim()
    sim2._state_path = state_file
    sim2._load_state()

    assert sim2.current_action == "看剧"
    assert sim2.location == "卧室"
    assert sim2.is_sleeping is False
    assert sim2.woke_up_today is True
    assert abs(sim2.physical.energy - 65.0) < 0.1
    assert abs(sim2.physical.hunger - 42.0) < 0.1
    assert sim2.yearago.location == "窗台"
    assert sim2.yearago.mood == "发呆"
    assert sim2._yearago_ticks_since_change == 3
    assert sim2.daily_weather["condition"] == "晴"
    assert len(sim2.event_log) == 1
    assert sim2.event_log[0].action == "做设计"  # LifeEvent attribute, not dict key


def test_load_state_stale_file_uses_defaults(tmp_path):
    stale_time = (datetime.now() - timedelta(hours=25)).isoformat()
    state_file = tmp_path / "life_state.json"
    state_file.write_text(json.dumps({
        "physical": {"energy": 10, "hunger": 90, "comfort": 50, "sleep_debt": 20},
        "location": "外面",
        "current_action": "跑步",
        "current_detail": "",
        "is_sleeping": False,
        "woke_up_today": True,
        "daily_weather": {"condition": "晴", "temp": 20, "date": "2026-01-01"},
        "yearago": {"location": "沙发", "mood": "发呆"},
        "yearago_ticks": 0,
        "event_log": [],
        "saved_at": stale_time,
    }))

    sim = make_sim()
    sim._state_path = state_file
    sim._load_state()

    # Defaults should be used, not the stale values
    assert sim.physical.energy == 80.0
    assert sim.current_action == "睡觉"


def test_load_state_missing_file_uses_defaults(tmp_path):
    sim = make_sim()
    sim._state_path = tmp_path / "nonexistent.json"
    sim._load_state()  # should not raise
    assert sim.physical.energy == 80.0


def test_load_state_corrupt_json_uses_defaults(tmp_path):
    state_file = tmp_path / "life_state.json"
    state_file.write_text("not valid json {{{")

    sim = make_sim()
    sim._state_path = state_file
    sim._load_state()  # should not raise

    assert sim.physical.energy == 80.0


def test_load_state_event_log_deserialized_as_lifeevent(tmp_path):
    """event_log 必须反序列化为 LifeEvent 实例（属性访问，不是字典访问）"""
    from life_simulator import LifeEvent
    sim = make_sim()
    sim.event_log = [
        LifeEvent("10:00", "做设计", "公司", "改稿", 0, -2, False, None)
    ]
    state_file = tmp_path / "life_state.json"
    sim._state_path = state_file
    asyncio.run(sim._do_save())

    sim2 = make_sim()
    sim2._state_path = state_file
    sim2._load_state()

    assert hasattr(sim2.event_log[0], "action")   # LifeEvent instance
    assert sim2.event_log[0].action == "做设计"
    # Verify it's NOT a plain dict
    assert not isinstance(sim2.event_log[0], dict)


def test_do_save_limits_event_log_to_25(tmp_path):
    from life_simulator import LifeEvent
    sim = make_sim()
    sim.event_log = [
        LifeEvent(f"{i:02d}:00", f"动作{i}", "家", "", 0, -2, False, None)
        for i in range(40)
    ]
    state_file = tmp_path / "life_state.json"
    sim._state_path = state_file
    asyncio.run(sim._do_save())

    raw = json.loads(state_file.read_text())
    assert len(raw["event_log"]) == 25
    assert raw["event_log"][-1]["action"] == "动作39"  # most recent
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "save or load or round_trip or corrupt or stale or event_log_de"
```

预期：AttributeError（`_do_save` 等方法不存在）

- [ ] **Step 3: 实现持久化方法**

**3a.** 所有新 import 已在 Task 1 Step 0 添加，无需重复。

**3b.** 在 `LifeSimulator.__init__` 的 `self._yearago_ticks_since_change` 初始化之后追加：

```python
        self.daily_weather: dict = {}      # {"condition":…, "temp":…, "date":…}
        self._state_path: Path = Path("data/life_state.json")
        self._load_state()
```

**3c.** 在类末尾追加持久化方法：

```python
    # ── 状态持久化 ────────────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """同步读取持久化状态（__init__ 中调用，此时事件循环尚未繁忙）"""
        try:
            text = self._state_path.read_text(encoding="utf-8")
            state = json.loads(text)
        except FileNotFoundError:
            return  # 首次启动，使用默认值
        except Exception as e:
            logger.warning(f"[LifeSim] 加载状态失败，使用默认值: {e}")
            return

        # 检查是否过期（超过 24h）
        try:
            saved_at = datetime.fromisoformat(state["saved_at"])
            if (datetime.now() - saved_at).total_seconds() > 86400:
                logger.info("[LifeSim] 状态超过 24h，重置")
                return
        except Exception:
            return

        # 恢复物理状态
        ph = state.get("physical", {})
        self.physical.energy    = float(ph.get("energy",    self.physical.energy))
        self.physical.hunger    = float(ph.get("hunger",    self.physical.hunger))
        self.physical.comfort   = float(ph.get("comfort",   self.physical.comfort))
        self.physical.sleep_debt = float(ph.get("sleep_debt", self.physical.sleep_debt))
        self.physical.clamp()

        self.location        = state.get("location",        self.location)
        self.current_action  = state.get("current_action",  self.current_action)
        self.current_detail  = state.get("current_detail",  self.current_detail)
        self.is_sleeping     = state.get("is_sleeping",     self.is_sleeping)
        self.woke_up_today   = state.get("woke_up_today",   self.woke_up_today)
        self.daily_weather   = state.get("daily_weather",   self.daily_weather)

        # 恢复年糕状态
        yg = state.get("yearago")
        if yg:
            try:
                self.yearago = YearGaoState(**yg)
            except Exception:
                pass
        self._yearago_ticks_since_change = state.get("yearago_ticks", 0)

        # 恢复 event_log（重建为 LifeEvent 实例）
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
        """调度异步保存（tick 末尾调用，不阻塞）"""
        asyncio.create_task(self._do_save())

    async def _do_save(self, path: "Path | None" = None) -> None:
        """实际写文件"""
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "save or load or round_trip or corrupt or stale or event_log_de"
```

预期：全部 PASS

- [ ] **Step 5: 确保所有已有测试通过**

```bash
python -m pytest tests/ -v
```

预期：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py
git commit -m "feat: add state persistence to LifeSimulator (save/load life_state.json)"
```

---

## Task 5: 将所有新功能接入 tick() 和 get_status()

**Files:**
- Modify: `life_simulator.py`（修改 `tick()` 和 `get_status()`）
- Modify: `tests/test_life_simulator_enriched.py`（追加集成测试）

- [ ] **Step 1: 追加集成测试**

在 `tests/test_life_simulator_enriched.py` 末尾追加：

```python
# ── Task 5: tick() 和 get_status() 集成 ─────────────────────────────────────

def _make_tick_sim(action_result: dict = None):
    """构造一个处于醒着状态、LLM 返回给定结果的 sim"""
    sim = make_sim()
    sim.is_sleeping = False
    sim.woke_up_today = True
    sim.physical.energy = 60.0
    sim.daily_weather = {
        "condition": "晴", "temp": 25,
        "date": datetime.now().date().isoformat(),
    }
    default_result = {
        "action": "做设计", "location": "公司", "detail": "在改稿",
        "mood_impact": 1, "energy_change": -2, "notable": False,
        "shareable_thought": None,
    }
    sim.llm.call_json = AsyncMock(return_value=action_result or default_result)
    return sim


def test_tick_calls_update_yearago():
    sim = _make_tick_sim()
    called = []
    original = sim._update_yearago
    def track(h): called.append(h); return original(h)
    sim._update_yearago = track

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 15, 0)
            await sim.tick()
    asyncio.run(run())
    assert len(called) >= 1


def test_tick_prompt_contains_blacklisted_action():
    """连续相同 action 后，下一 tick 的 prompt 应包含黑名单条目"""
    sim = _make_tick_sim()
    sim.event_log = [_make_event("刷小红书")] * 3

    captured_prompts = []
    original_call_json = sim.llm.call_json

    async def capture(prompt, **kwargs):
        captured_prompts.append(prompt)
        return await original_call_json(prompt, **kwargs)

    sim.llm.call_json = capture

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 15, 0)
            await sim.tick()
    asyncio.run(run())

    assert captured_prompts, "call_json never called"
    assert "刷小红书" in captured_prompts[0]


def test_tick_prompt_contains_activity_hints():
    """tick 的 prompt 应包含时段活动提示"""
    sim = _make_tick_sim()
    captured_prompts = []
    original = sim.llm.call_json
    async def capture(prompt, **kwargs):
        captured_prompts.append(prompt)
        return await original(prompt, **kwargs)
    sim.llm.call_json = capture

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 15, 0)
            await sim.tick()
    asyncio.run(run())
    assert any("时间段" in p or "比较可能" in p for p in captured_prompts)


def test_tick_prompt_contains_real_weather():
    sim = _make_tick_sim()
    sim.daily_weather = {"condition": "小雨", "temp": 16, "date": datetime.now().date().isoformat()}
    captured_prompts = []
    original = sim.llm.call_json
    async def capture(prompt, **kwargs):
        captured_prompts.append(prompt)
        return await original(prompt, **kwargs)
    sim.llm.call_json = capture

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 15, 0)
            await sim.tick()
    asyncio.run(run())
    assert any("小雨" in p for p in captured_prompts)


def test_tick_prompt_contains_yearago():
    sim = _make_tick_sim()
    from life_simulator import YearGaoState
    sim.yearago = YearGaoState(location="厨房", mood="讨食")
    captured_prompts = []
    original = sim.llm.call_json
    async def capture(prompt, **kwargs):
        captured_prompts.append(prompt)
        return await original(prompt, **kwargs)
    sim.llm.call_json = capture

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 18, 0)
            await sim.tick()
    asyncio.run(run())
    assert any("年糕" in p for p in captured_prompts)
    assert any("讨食" in p or "闹腾" in p for p in captured_prompts)


def test_get_status_includes_weather_and_yearago():
    from life_simulator import YearGaoState
    sim = make_sim()
    sim.daily_weather = {"condition": "晴", "temp": 25, "date": "2026-03-22"}
    sim.yearago = YearGaoState(location="窗台", mood="发呆")

    status = sim.get_status()

    assert "weather" in status
    assert status["weather"]["condition"] == "晴"
    assert status["weather"]["temp"] == 25
    assert "yearago" in status
    assert status["yearago"]["location"] == "窗台"
    assert status["yearago"]["mood"] == "发呆"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v -k "tick_ or get_status"
```

预期：AssertionError（prompt 还没注入）

- [ ] **Step 3: 修改 tick()**

用以下完整的 `tick()` 方法替换 `life_simulator.py` 中现有的 `tick()` 方法：

```python
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

        # ── 构建 prompt ──
        recent_actions = [e.action for e in self.event_log[-6:]]
        weather_str = (
            f"{self.daily_weather['condition']} {self.daily_weather['temp']}°C"
            if self.daily_weather
            else "未知"
        )
        yearago_str = f"你的猫年糕现在：在{self.yearago.location}{self.yearago.mood}"
        yearago_notable = ""
        if self.yearago.mood in ("讨食", "暴走"):
            yearago_notable = "\n（年糕现在比较闹腾，相关的事情比平时更值得分享）"

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
她的性格：外向{self.persona['personality']['extraversion']:.1f} 自律{self.persona['personality']['conscientiousness']:.1f}

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

        # 有效 tick：保存状态
        self._save_state()
```

- [ ] **Step 4: 修改 get_status()**

用以下方法替换现有的 `get_status()`：

```python
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
```

- [ ] **Step 5: 运行全部测试**

```bash
python -m pytest tests/test_life_simulator_enriched.py -v
```

预期：全部 PASS

- [ ] **Step 6: 修复 test_life_woke_up.py 的天气兼容性**

新 `tick()` 在 `daily_weather["date"]` 不等于今天时会发起真实 HTTP 请求。`test_life_woke_up.py` 中的 sim 没有预设 `daily_weather`，会导致测试发起网络请求（不稳定）。

在 `tests/test_life_woke_up.py` 的 `make_sim()` 函数中，`sim = LifeSimulator(...)` 这行**之后**追加：

```python
    from datetime import date
    sim.daily_weather = {
        "condition": "晴", "temp": 22,
        "date": date.today().isoformat(),
    }
```

- [ ] **Step 7: 运行所有测试，确认全部通过**

```bash
python -m pytest tests/ -v
```

预期：全部 PASS。`test_life_woke_up.py` 中 `random.uniform` 的 side_effect 顺序不变（`_update_yearago` 只调用 `random.random` / `random.choices`，不调用 `random.uniform`）。

- [ ] **Step 8: Commit**

```bash
git add life_simulator.py tests/test_life_simulator_enriched.py tests/test_life_woke_up.py
git commit -m "feat: wire yearago/weather/diversity/persistence into tick() and get_status()"
```

---

## Task 6: 更新 config.yaml

**Files:**
- Modify: `config.yaml`（在 `persona:` 区块末尾追加 `city: "上海"`）

- [ ] **Step 1: 在 config.yaml 中追加 city 字段**

在 `config.yaml` 的 `persona:` 区块内、`daily_patterns:` 之前（任意位置均可）追加一行：

```yaml
  city: "上海"          # 用于获取真实天气，不填则默认北京
```

- [ ] **Step 2: 运行所有测试，最终确认**

```bash
python -m pytest tests/ -v
```

预期：全部 PASS

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "config: add persona.city field for real weather fetching"
```

---

## 验收清单

完成所有 Task 后，请手动验证以下行为：

- [ ] 启动 bot，记录当前状态，重启后状态与重启前一致（`/status` 对比）
- [ ] 运行 bot 超过 3 个 tick，观察 log 中 action 不再连续重复相同条目
- [ ] log 中 prompt 包含今日真实天气（条件：网络可达）；或在断网时退化为季节默认值
- [ ] log 中可见年糕状态变化（`年糕:讨食` 等）
- [ ] 超过 24h 停机后重启，状态重置为默认值
