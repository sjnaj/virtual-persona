# tests/test_life_simulator_enriched.py
import asyncio
import json
import dataclasses
from datetime import datetime, timedelta, date
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
    sim._yearago_ticks_since_change = 2
    sim.yearago.mood = "发呆"
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["讨食"]):
        sim._update_yearago(7.0)
    assert sim.yearago.mood == "讨食"
    assert sim.yearago.location == "厨房"
    assert sim._yearago_ticks_since_change == 0


def test_update_yearago_no_switch_when_random_above_threshold():
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    sim.yearago.mood = "发呆"
    with patch("life_simulator.random.random", return_value=0.9):
        sim._update_yearago(7.0)
    assert sim.yearago.mood == "发呆"


def test_update_yearago_location_follows_mood_indoor():
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["赖床"]):
        sim._update_yearago(10.0)
    assert sim.yearago.location in ("沙发", "床上", "窗台", "猫窝")


def test_update_yearago_location_chaos_on_baozou():
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["暴走"]):
        sim._update_yearago(21.0)
    assert sim.yearago.location == "消失了"


def test_update_yearago_night_range_covered():
    sim = make_sim()
    sim._yearago_ticks_since_change = 5
    with patch("life_simulator.random.random", return_value=0.1), \
         patch("life_simulator.random.choices", return_value=["发呆"]):
        sim._update_yearago(2.0)
    assert sim._yearago_ticks_since_change == 0


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
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("life_simulator.httpx.AsyncClient", return_value=mock_cm):
            return await sim._fetch_weather("上海")

    result = asyncio.run(run())
    assert result["condition"] == "晴"
    assert result["temp"] == 22
    assert "date" in result


def test_fetch_weather_request_error_returns_default():
    import httpx as _httpx
    sim = make_sim()

    async def run():
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=_httpx.RequestError("timeout"))
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
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        with patch("life_simulator.httpx.AsyncClient", return_value=mock_cm):
            return await sim._fetch_weather("上海")

    result = asyncio.run(run())
    assert "condition" in result
    assert "temp" in result


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
    for h in [0, 3, 7, 9, 12, 15, 19, 22]:
        assert sim._get_activity_hints(hour=float(h), is_weekday=True)
        assert sim._get_activity_hints(hour=float(h), is_weekday=False)


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
    assert sim2.event_log[0].action == "做设计"


def _make_sim_no_load():
    """make_sim but suppress the _load_state() call inside __init__."""
    from life_simulator import LifeSimulator
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = make_sim()
    return sim


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

    sim = _make_sim_no_load()
    sim._state_path = state_file
    sim._load_state()

    assert sim.physical.energy == 80.0
    assert sim.current_action == "睡觉"


def test_load_state_missing_file_uses_defaults(tmp_path):
    sim = _make_sim_no_load()
    sim._state_path = tmp_path / "nonexistent.json"
    sim._load_state()
    assert sim.physical.energy == 80.0


def test_load_state_corrupt_json_uses_defaults(tmp_path):
    state_file = tmp_path / "life_state.json"
    state_file.write_text("not valid json {{{")

    sim = _make_sim_no_load()
    sim._state_path = state_file
    sim._load_state()

    assert sim.physical.energy == 80.0


def test_load_state_event_log_deserialized_as_lifeevent(tmp_path):
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

    assert hasattr(sim2.event_log[0], "action")
    assert sim2.event_log[0].action == "做设计"
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
    assert raw["event_log"][-1]["action"] == "动作39"


# ── Task 5: tick() 和 get_status() 集成 ─────────────────────────────────────

def _make_tick_sim(action_result: dict = None):
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
    from life_simulator import YearGaoState
    sim = _make_tick_sim()
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


# ── Task: interests in tick prompt ──────────────────────────────────────────

def _make_tick_sim_with_interests():
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {
            "wake_up": [7, 8], "sleep": [23, 25],
            "work_start": [9, 10], "work_end": [18, 19], "lunch": [11, 13],
        },
        "interests": ["猫", "穿搭", "甜品"],
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = LifeSimulator(persona, llm, bus)
    sim.is_sleeping = False
    sim.woke_up_today = True
    sim.physical.energy = 60.0
    sim.daily_weather = {
        "condition": "晴", "temp": 25,
        "date": datetime.now().date().isoformat(),
    }
    default_result = {
        "action": "做设计", "location": "公司", "detail": "改稿",
        "mood_impact": 0, "energy_change": -2, "notable": False,
        "shareable_thought": None,
    }
    sim.llm.call_json = AsyncMock(return_value=default_result)
    return sim


def test_tick_prompt_contains_interests():
    sim = _make_tick_sim_with_interests()
    captured = []
    original = sim.llm.call_json
    async def capture(prompt, **kwargs):
        captured.append(prompt)
        return await original(prompt, **kwargs)
    sim.llm.call_json = capture

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 15, 0)
            await sim.tick()
    asyncio.run(run())

    assert captured, "call_json never called"
    prompt = captured[0]
    assert "她的兴趣爱好" in prompt
    # Use "穿搭" (not "猫" — "猫" appears in yearago_str regardless of this change)
    assert "穿搭" in prompt
    assert "甜品" in prompt


# ── Task: dynamic schedule hint ─────────────────────────────────────────────

def _make_sim_with_schedule():
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    from unittest.mock import patch
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {
            "wake_up": [7, 8], "sleep": [23, 25],
            "work_start": [9, 10], "work_end": [18, 19], "lunch": [11, 13],
        },
        "interests": [],
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = LifeSimulator(persona, llm, bus)
    return sim


def test_weekday_hints_include_work_start():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "上班时间约9-10" in hints  # work_start [9,10] from config


def test_weekday_hints_include_work_end():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "下班约18-19" in hints  # work_end [18,19] from config


def test_weekday_hints_include_lunch():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "午饭约11-13" in hints  # lunch [11,13] from config


def test_weekend_hints_do_not_include_schedule_annotation():
    sim = _make_sim_with_schedule()
    hints = sim._get_activity_hints(14.0, is_weekday=False)
    # Weekend should not have schedule parenthetical (this test passes before
    # implementation too — it is a guard that the weekend path stays clean)
    assert "上班时间" not in hints
    assert "下班" not in hints


def test_schedule_annotation_uses_config_not_hardcoded():
    """Custom schedule values appear in hints, not hardcoded defaults."""
    from life_simulator import LifeSimulator
    from event_bus import EventBus
    from unittest.mock import patch
    bus = EventBus()
    persona = {
        "name": "测试",
        "occupation": "设计师",
        "personality": {"extraversion": 0.7, "conscientiousness": 0.5},
        "daily_patterns": {
            "wake_up": [6, 7], "sleep": [22, 23],
            "work_start": [8, 9], "work_end": [17, 18], "lunch": [12, 13],
        },
        "interests": [],
    }
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    with patch.object(LifeSimulator, '_load_state', lambda self: None):
        sim = LifeSimulator(persona, llm, bus)
    hints = sim._get_activity_hints(10.0, is_weekday=True)
    assert "上班时间约8-9" in hints   # custom work_start
    assert "下班约17-18" in hints     # custom work_end
    assert "午饭约12-13" in hints     # custom lunch
