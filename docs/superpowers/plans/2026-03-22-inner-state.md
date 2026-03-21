# Inner State Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `InnerStateManager` that gives the persona cross-conversation continuity and unpredictable emotion carry-over via a persisted inner monologue.

**Architecture:** A new `inner_state.py` module owns two dataclasses (`InnerMonologue`, `PendingThought`) and an `InnerStateManager` that subscribes to bus events, generates periodic LLM monologues, persists state to disk, and wires into the emotion engine, proactive engine, and expression layer.

**Tech Stack:** Python 3.10+, asyncio, dataclasses, json, unittest.mock (tests), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `inner_state.py` | `InnerMonologue`, `PendingThought` dataclasses + `InnerStateManager` |
| Create | `tests/test_inner_state.py` | Unit tests for inner_state.py |
| Create | `tests/test_emotion_baseline.py` | Unit tests for EmotionEngine daily_baseline |
| Create | `tests/test_life_woke_up.py` | Unit test for life.woke_up emit |
| Create | `tests/test_memory_consolidated_event.py` | Unit test for memory.consolidated emit |
| Modify | `emotion_engine.py:31-90` | Add `daily_baseline` field + `set_daily_baseline()` + fix `passive_decay()` |
| Modify | `life_simulator.py:65-76` | Emit `life.woke_up` after wake transition |
| Modify | `memory_hub.py:58-155` | Emit `memory.consolidated` per chat after consolidate |
| Modify | `proactive_engine.py:12-144` | Subscribe to `inner_state.updated` |
| Modify | `expression.py:18-215` | Accept `inner_state` kwarg; inject monologue into prompt |
| Modify | `orchestrator.py:25-259` | Wire `InnerStateManager`; add `_inner_state_loop`; emit `proactive.follow_up_fired` |
| Modify | `config.yaml:56-61` | Add `inner_state_interval_hours` + `pending_thought_ttl_hours` |

---

## Task 1: InnerMonologue + PendingThought dataclasses with persistence

**Files:**
- Create: `inner_state.py`
- Create: `tests/test_inner_state.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_inner_state.py
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path


def test_inner_monologue_round_trips():
    from inner_state import InnerMonologue
    m = InnerMonologue(text="心里有点空", mood_tint=0.1, mood_reason="今天累了",
                       generated_at="2026-03-22T10:00:00")
    assert InnerMonologue.from_dict(m.to_dict()) == m


def test_pending_thought_round_trips():
    from inner_state import PendingThought
    t = PendingThought(content="想问小明", target_user_id=123, urgency=0.6,
                       source="conversation",
                       created_at="2026-03-22T10:00:00",
                       expires_at="2026-03-24T10:00:00")
    assert PendingThought.from_dict(t.to_dict()) == t


def test_pending_thought_is_expired():
    from inner_state import PendingThought
    past = (datetime.now() - timedelta(hours=1)).isoformat()
    t = PendingThought(content="x", target_user_id=0, urgency=0.5, source="life_event",
                       created_at=past, expires_at=past)
    assert t.is_expired() is True


def test_pending_thought_not_expired():
    from inner_state import PendingThought
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    t = PendingThought(content="x", target_user_id=0, urgency=0.5, source="life_event",
                       created_at=datetime.now().isoformat(), expires_at=future)
    assert t.is_expired() is False


def test_inner_state_manager_save_load(tmp_path):
    from inner_state import InnerStateManager, InnerMonologue, PendingThought
    from unittest.mock import MagicMock
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "inner_state.json"),
    )
    mgr.yesterday_final_valence = 0.3
    mgr.current_monologue = InnerMonologue(
        text="测试独白", mood_tint=0.2, mood_reason="测试原因",
        generated_at="2026-03-22T10:00:00"
    )
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    mgr.pending_thoughts = [
        PendingThought(content="待办", target_user_id=1, urgency=0.5,
                       source="conversation",
                       created_at="2026-03-22T10:00:00",
                       expires_at=future)
    ]
    mgr._save()

    mgr2 = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "inner_state.json"),
    )
    assert mgr2.yesterday_final_valence == 0.3
    assert mgr2.current_monologue.text == "测试独白"
    assert len(mgr2.pending_thoughts) == 1
    assert mgr2.pending_thoughts[0].content == "待办"


def test_inner_state_manager_load_missing_file(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "nonexistent.json"),
    )
    assert mgr.yesterday_final_valence == 0.0
    assert mgr.current_monologue is None
    assert mgr.pending_thoughts == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_inner_state.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'inner_state'`

- [ ] **Step 3: Implement dataclasses and InnerStateManager skeleton**

```python
# inner_state.py
"""
内心独白层 —— 跨对话连续性和情绪跨日延续
"""
import json
import random
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SAVE_PATH = "./data/inner_state.json"


@dataclass
class InnerMonologue:
    text: str
    mood_tint: float   # -1~1，仅供 prompt 注入，不作为衰减地板
    mood_reason: str
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InnerMonologue":
        return cls(**{k: v for k, v in d.items()
                     if k in cls.__dataclass_fields__})


@dataclass
class PendingThought:
    content: str
    target_user_id: int   # 0=不针对某人
    urgency: float        # 0~1
    source: str           # "conversation" / "life_event" / "browsing"
    created_at: str
    expires_at: str

    def is_expired(self) -> bool:
        return datetime.fromisoformat(self.expires_at) < datetime.now()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingThought":
        return cls(**{k: v for k, v in d.items()
                     if k in cls.__dataclass_fields__})


class InnerStateManager:
    def __init__(self, persona: dict, llm, emotion, life_sim, event_bus,
                 save_path: str = DEFAULT_SAVE_PATH,
                 ttl_hours: int = 48):
        self.persona = persona
        self.llm = llm
        self.emotion = emotion
        self.life_sim = life_sim
        self.bus = event_bus
        self.save_path = save_path
        self.ttl_hours = ttl_hours

        self.current_monologue: Optional[InnerMonologue] = None
        self.yesterday_final_valence: float = 0.0
        self.pending_thoughts: List[PendingThought] = []
        self._recent_convos: List[str] = []   # max 3, from memory.consolidated

        self.bus.subscribe("life.sleeping", self._on_sleep)
        self.bus.subscribe("life.woke_up", self._on_woke_up)
        self.bus.subscribe("memory.consolidated", self._on_memory_consolidated)
        self.bus.subscribe("proactive.follow_up_fired", self._on_follow_up_fired)

        self._load()

    def _load(self):
        path = Path(self.save_path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("current_monologue"):
                self.current_monologue = InnerMonologue.from_dict(
                    data["current_monologue"]
                )
            self.yesterday_final_valence = data.get("yesterday_final_valence", 0.0)
            self.pending_thoughts = [
                PendingThought.from_dict(t)
                for t in data.get("pending_thoughts", [])
            ]
            logger.info(f"[InnerState] 加载: {len(self.pending_thoughts)} pending thoughts")
        except Exception as e:
            logger.error(f"[InnerState] 加载失败: {e}")

    def _save(self):
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_monologue": (self.current_monologue.to_dict()
                                  if self.current_monologue else None),
            "yesterday_final_valence": self.yesterday_final_valence,
            "pending_thoughts": [t.to_dict() for t in self.pending_thoughts],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def _on_sleep(self, event):
        self.yesterday_final_valence = self.emotion.state.valence
        self._save()
        logger.info(f"[InnerState] 入睡快照 valence={self.yesterday_final_valence:.3f}")

    async def _on_woke_up(self, event):
        await self.on_wake_up()

    async def on_wake_up(self):
        now = datetime.now()
        weekday_adj = {0: -0.05, 4: 0.05, 5: 0.03, 6: 0.03}.get(now.weekday(), 0.0)
        sleep_debt_adj = -0.1 if self.life_sim.physical.sleep_debt > 30 else 0.0
        baseline = (
            self.yesterday_final_valence * 0.4
            + random.uniform(-0.15, 0.15)
            + weekday_adj
            + sleep_debt_adj
        )
        baseline = max(-1.0, min(1.0, baseline))
        self.emotion.set_daily_baseline(baseline)
        logger.info(f"[InnerState] 起床基线 baseline={baseline:.3f}")

    async def _on_memory_consolidated(self, event):
        convo_text = event.data.get("convo_text", "")
        if convo_text:
            self._recent_convos.append(convo_text)
            if len(self._recent_convos) > 3:
                self._recent_convos = self._recent_convos[-3:]

    async def _on_follow_up_fired(self, event):
        thought_content = event.data.get("thought_content", "")
        self.pending_thoughts = [
            t for t in self.pending_thoughts if t.content != thought_content
        ]
        self._save()
        logger.info(f"[InnerState] follow_up 已发送，移除: {thought_content[:30]}")

    async def generate_monologue(self):
        if self.life_sim.is_sleeping:
            return

        now = datetime.now()
        self.pending_thoughts = [t for t in self.pending_thoughts if not t.is_expired()]

        emotion_status = self.emotion.get_status()
        life_status = self.life_sim.get_status()

        recent_notable = [e for e in self.life_sim.event_log[-12:] if e.notable]
        notable_text = "\n".join(
            f"- {e.time} {e.detail}" for e in recent_notable
        ) or "没有特别的事"
        convos_text = "\n---\n".join(self._recent_convos) if self._recent_convos else "没有最近的对话"
        pending_text = "\n".join(
            f"- {t.content}（urgency={t.urgency:.1f}）" for t in self.pending_thoughts
        ) or "没有"

        prompt = f"""你是{self.persona['name']}，现在 {now.strftime('%H:%M')}，请以她的第一人称写一段内心独白。

当前状态：
- 心情：valence={emotion_status['valence']:.2f}，arousal={emotion_status['arousal']:.2f}
- 正在做：{life_status['current_action']} @ {life_status['location']}
- 体力：{life_status['physical']['energy']}/100

最近发生的事：
{notable_text}

最近的对话（参考，不要复述）：
{convos_text[:800]}

脑子里还挂着的事：
{pending_text}

请输出JSON：
{{
  "monologue": "她此刻的内心独白（一句话，口语化）",
  "mood_tint": 0.0,
  "mood_reason": "情绪色调的原因（简短）",
  "new_pending_thoughts": [
    {{"content": "想做/想说/想问的事", "target_user_id": 0, "urgency": 0.5, "source": "conversation"}}
  ],
  "resolved_thoughts": ["已经处理掉的事（原文）"]
}}

mood_tint 范围 -1~1。"""

        result = await self.llm.call_json(prompt, tier="utility")
        if not result:
            logger.warning("[InnerState] 独白生成失败，保留上次")
            return

        ttl = timedelta(hours=self.ttl_hours)
        now_str = now.isoformat()

        self.current_monologue = InnerMonologue(
            text=result.get("monologue", ""),
            mood_tint=float(result.get("mood_tint", 0.0)),
            mood_reason=result.get("mood_reason", ""),
            generated_at=now_str,
        )

        resolved = set(result.get("resolved_thoughts", []))
        self.pending_thoughts = [
            t for t in self.pending_thoughts if t.content not in resolved
        ]

        for raw in result.get("new_pending_thoughts", []):
            content = raw.get("content", "").strip()
            if content:
                self.pending_thoughts.append(PendingThought(
                    content=content,
                    target_user_id=int(raw.get("target_user_id", 0)),
                    urgency=float(raw.get("urgency", 0.5)),
                    source=raw.get("source", "conversation"),
                    created_at=now_str,
                    expires_at=(now + ttl).isoformat(),
                ))

        self._save()
        await self.bus.emit("inner_state.updated", {
            "pending_thoughts": [
                t.to_dict() for t in self.pending_thoughts
                if t.target_user_id > 0
            ],
        })
        logger.info(f"[InnerState] 独白生成完成，pending={len(self.pending_thoughts)}")

    def get_monologue_text(self) -> str:
        return self.current_monologue.text if self.current_monologue else ""
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_inner_state.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add inner_state.py tests/test_inner_state.py
git commit -m "feat: add InnerMonologue/PendingThought dataclasses and InnerStateManager skeleton"
```

---

## Task 2: InnerStateManager — on_sleep, on_wake_up, _on_memory_consolidated, _on_follow_up_fired

**Files:**
- Modify: `tests/test_inner_state.py` (append tests)

The implementation is already in `inner_state.py` from Task 1. This task adds behaviour tests.

- [ ] **Step 1: Append tests to tests/test_inner_state.py**

```python
# append to tests/test_inner_state.py

def test_on_sleep_snapshots_valence(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock
    from event_bus import Event
    bus = MagicMock()
    bus.subscribe = MagicMock()
    emotion = MagicMock()
    emotion.state.valence = 0.42
    life_sim = MagicMock()
    life_sim.physical.sleep_debt = 0
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    asyncio.run(mgr._on_sleep(Event("life.sleeping", {})))
    assert mgr.yesterday_final_valence == pytest.approx(0.42)
    # verify persisted by loading with a fresh instance
    bus2 = MagicMock()
    bus2.subscribe = MagicMock()
    mgr2 = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus2,
        save_path=str(tmp_path / "s.json"),
    )
    assert mgr2.yesterday_final_valence == pytest.approx(0.42)


def test_on_wake_up_calls_set_daily_baseline(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock, call
    bus = MagicMock()
    bus.subscribe = MagicMock()
    emotion = MagicMock()
    emotion.state.valence = 0.3
    life_sim = MagicMock()
    life_sim.physical.sleep_debt = 0
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    mgr.yesterday_final_valence = 0.5
    asyncio.run(mgr.on_wake_up())
    assert emotion.set_daily_baseline.called
    called_val = emotion.set_daily_baseline.call_args[0][0]
    assert -1.0 <= called_val <= 1.0


def test_on_memory_consolidated_buffers_convos(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock
    from event_bus import Event
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    for i in range(4):
        asyncio.run(mgr._on_memory_consolidated(
            Event("memory.consolidated", {"chat_id": i, "convo_text": f"对话{i}"})
        ))
    assert len(mgr._recent_convos) == 3
    assert mgr._recent_convos[0] == "对话1"   # oldest dropped


def test_on_follow_up_fired_removes_thought(tmp_path):
    from inner_state import InnerStateManager, PendingThought
    from unittest.mock import MagicMock
    from event_bus import Event
    from datetime import timedelta
    bus = MagicMock()
    bus.subscribe = MagicMock()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=MagicMock(), emotion=MagicMock(),
        life_sim=MagicMock(), event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    mgr.pending_thoughts = [
        PendingThought(content="想问小明", target_user_id=1, urgency=0.5,
                       source="conversation",
                       created_at=datetime.now().isoformat(),
                       expires_at=future),
        PendingThought(content="其他事情", target_user_id=2, urgency=0.3,
                       source="conversation",
                       created_at=datetime.now().isoformat(),
                       expires_at=future),
    ]
    asyncio.run(mgr._on_follow_up_fired(
        Event("proactive.follow_up_fired", {"thought_content": "想问小明"})
    ))
    assert len(mgr.pending_thoughts) == 1
    assert mgr.pending_thoughts[0].content == "其他事情"
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_inner_state.py -v
```
Expected: all tests PASS (including the 4 new ones)

- [ ] **Step 3: Commit**

```bash
git add tests/test_inner_state.py
git commit -m "test: add InnerStateManager behaviour tests (on_sleep, on_wake_up, memory, follow_up)"
```

---

## Task 3: InnerStateManager — generate_monologue()

**Files:**
- Modify: `tests/test_inner_state.py` (append tests)

- [ ] **Step 1: Append tests**

```python
# append to tests/test_inner_state.py

def test_generate_monologue_skips_when_sleeping(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    life_sim = MagicMock()
    life_sim.is_sleeping = True
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=MagicMock(),
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    asyncio.run(mgr.generate_monologue())
    llm.call_json.assert_not_called()


def test_generate_monologue_on_llm_failure_preserves_previous(tmp_path):
    from inner_state import InnerStateManager, InnerMonologue
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value=None)
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    prev = InnerMonologue(text="上次的独白", mood_tint=0.1,
                          mood_reason="测试", generated_at="2026-01-01T00:00:00")
    mgr.current_monologue = prev
    asyncio.run(mgr.generate_monologue())
    assert mgr.current_monologue.text == "上次的独白"


def test_generate_monologue_updates_state(tmp_path):
    from inner_state import InnerStateManager
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "monologue": "今天有点累，但还好",
        "mood_tint": -0.1,
        "mood_reason": "没睡好",
        "new_pending_thoughts": [
            {"content": "想问张三项目进度", "target_user_id": 999, "urgency": 0.7, "source": "conversation"}
        ],
        "resolved_thoughts": [],
    })
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 60}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": -0.1, "arousal": 0.0}
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    emitted_events = []
    async def capture(event):
        emitted_events.append(event)
    bus.subscribe("inner_state.updated", capture)
    asyncio.run(mgr.generate_monologue())
    assert mgr.current_monologue.text == "今天有点累，但还好"
    assert len(mgr.pending_thoughts) == 1
    assert mgr.pending_thoughts[0].content == "想问张三项目进度"
    assert mgr.pending_thoughts[0].target_user_id == 999
    # verify inner_state.updated was emitted with the thought in payload
    assert len(emitted_events) == 1
    payload_thoughts = emitted_events[0].data["pending_thoughts"]
    assert any(t["content"] == "想问张三项目进度" for t in payload_thoughts)


def test_generate_monologue_removes_resolved_thoughts(tmp_path):
    from inner_state import InnerStateManager, PendingThought
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    from datetime import timedelta
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "monologue": "清爽",
        "mood_tint": 0.2,
        "mood_reason": "处理完了",
        "new_pending_thoughts": [],
        "resolved_thoughts": ["想问张三项目进度"],
    })
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": 0.2, "arousal": 0.1}
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    mgr.pending_thoughts = [
        PendingThought(content="想问张三项目进度", target_user_id=999, urgency=0.7,
                       source="conversation",
                       created_at=datetime.now().isoformat(), expires_at=future)
    ]
    asyncio.run(mgr.generate_monologue())
    assert len(mgr.pending_thoughts) == 0


def test_generate_monologue_cleans_expired_thoughts(tmp_path):
    from inner_state import InnerStateManager, PendingThought
    from unittest.mock import MagicMock, AsyncMock
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "monologue": "ok", "mood_tint": 0.0, "mood_reason": "",
        "new_pending_thoughts": [], "resolved_thoughts": [],
    })
    life_sim = MagicMock()
    life_sim.is_sleeping = False
    life_sim.event_log = []
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    emotion = MagicMock()
    emotion.get_status.return_value = {"valence": 0.0, "arousal": 0.0}
    past = (datetime.now() - timedelta(hours=1)).isoformat()
    mgr = InnerStateManager(
        persona={"name": "测试"}, llm=llm, emotion=emotion,
        life_sim=life_sim, event_bus=bus,
        save_path=str(tmp_path / "s.json"),
    )
    mgr.pending_thoughts = [
        PendingThought(content="已过期", target_user_id=1, urgency=0.5,
                       source="conversation", created_at=past, expires_at=past)
    ]
    asyncio.run(mgr.generate_monologue())
    assert len(mgr.pending_thoughts) == 0
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_inner_state.py -v
```
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_inner_state.py
git commit -m "test: add generate_monologue tests"
```

---

## Task 4: EmotionEngine — daily_baseline + set_daily_baseline() + passive_decay fix

**Files:**
- Modify: `emotion_engine.py`
- Create: `tests/test_emotion_baseline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_emotion_baseline.py
from unittest.mock import MagicMock


def make_engine():
    from emotion_engine import EmotionEngine
    bus = MagicMock()
    bus.subscribe = MagicMock()
    persona = {"personality": {"neuroticism": 0.5}}
    return EmotionEngine(persona, bus)


def test_daily_baseline_defaults_to_zero():
    engine = make_engine()
    assert engine.daily_baseline == 0.0


def test_set_daily_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    assert engine.daily_baseline == 0.3


def test_passive_decay_valence_converges_toward_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.valence = 0.9
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.valence - 0.3) < 0.05


def test_passive_decay_valence_rises_toward_baseline_when_below():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.valence = -0.5
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.valence - 0.3) < 0.05


def test_passive_decay_arousal_decays_to_zero_regardless_of_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.arousal = 0.8
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.arousal) < 0.05


def test_passive_decay_baseline_zero_preserves_original_behaviour():
    """With baseline=0 (default), valence still decays toward zero."""
    engine = make_engine()
    engine.state.valence = 0.8
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.valence) < 0.05


def test_passive_decay_irritability_decays_to_zero_regardless_of_baseline():
    engine = make_engine()
    engine.set_daily_baseline(0.3)
    engine.state.irritability = 0.9
    for _ in range(200):
        engine.passive_decay()
    assert abs(engine.state.irritability) < 0.05
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_emotion_baseline.py -v 2>&1 | head -20
```
Expected: `AttributeError: 'EmotionEngine' object has no attribute 'daily_baseline'`

- [ ] **Step 3: Modify emotion_engine.py**

In `EmotionEngine.__init__` add after `self.neuroticism = ...`:
```python
self.daily_baseline: float = 0.0
```

After `passive_decay()` add new method:
```python
def set_daily_baseline(self, value: float):
    """由 InnerStateManager 在每天起床时调用一次，设定当日情绪地板。"""
    self.daily_baseline = max(-1.0, min(1.0, value))
```

In `passive_decay()` replace `self.state.valence *= 0.98` with:
```python
self.state.valence = self.daily_baseline + (self.state.valence - self.daily_baseline) * 0.98
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_emotion_baseline.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add emotion_engine.py tests/test_emotion_baseline.py
git commit -m "feat: add EmotionEngine daily_baseline and fix passive_decay to converge toward baseline"
```

---

## Task 5: life_simulator.py — emit life.woke_up

**Files:**
- Modify: `life_simulator.py`
- Create: `tests/test_life_woke_up.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_life_woke_up.py
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime


def make_sim(hour: float):
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
    llm.call_json = AsyncMock(return_value={
        "action": "刷手机", "location": "家", "detail": "看微博",
        "mood_impact": 0, "energy_change": -2, "notable": False, "shareable_thought": None
    })
    sim = LifeSimulator(persona, llm, bus)
    sim.is_sleeping = True
    sim.woke_up_today = False
    sim.physical.energy = 50
    return sim, bus


def test_life_woke_up_event_emitted():
    sim, bus = make_sim(7.5)
    emitted_events = []

    async def capture(event):
        emitted_events.append(event.name)

    bus.subscribe("life.woke_up", capture)

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 7, 30)
            # Force wake up: use side_effect so only the wake-check call returns 0.0
            # while the energy-calculation call returns its real value.
            # Call order in tick(): _passive_updates (2 calls), wake check (1 call).
            with patch("life_simulator.random.uniform",
                       side_effect=[0.5, 0.5, 0.0, 15.0]):
                # side_effect list: passive energy, passive hunger, wake threshold, energy gain
                await sim.tick()

    asyncio.run(run())
    assert "life.woke_up" in emitted_events


def test_life_woke_up_not_emitted_while_sleeping():
    sim, bus = make_sim(6.0)
    emitted_events = []

    async def capture(event):
        emitted_events.append(event.name)

    bus.subscribe("life.woke_up", capture)

    async def run():
        with patch("life_simulator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 22, 6, 0)
            await sim.tick()

    asyncio.run(run())
    assert "life.woke_up" not in emitted_events
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_life_woke_up.py::test_life_woke_up_event_emitted -v 2>&1 | head -20
```
Expected: FAIL — `life.woke_up` not in emitted_events

- [ ] **Step 3: Modify life_simulator.py**

Find the wake-up block (lines 68-75). Place the emit **after all state mutations** so subscribers observe a fully-consistent state:

```python
if not self.woke_up_today:
    self.is_sleeping = False
    self.woke_up_today = True
    self.physical.energy = 60 + random.uniform(0, 30) - self.physical.sleep_debt
    self.physical.energy = max(30, self.physical.energy)
    self.current_action = "刚醒，赖床中"
    self.location = "家"
    logger.info(f"[LifeSim] 起床了，体力={self.physical.energy:.0f}")
    await self.bus.emit("life.woke_up", {})
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_life_woke_up.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add life_simulator.py tests/test_life_woke_up.py
git commit -m "feat: emit life.woke_up event on wake transition in LifeSimulator"
```

---

## Task 6: memory_hub.py — emit memory.consolidated

**Files:**
- Modify: `memory_hub.py`
- Create: `tests/test_memory_consolidated_event.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_memory_consolidated_event.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


def test_consolidate_emits_memory_consolidated(tmp_path):
    from memory_hub import MemoryHub
    from event_bus import EventBus
    bus = EventBus()
    llm = MagicMock()
    llm.call_json = AsyncMock(return_value={
        "facts": [{"fact": "他喜欢猫", "about_person": "用户", "confidence": 0.9, "source_privacy": "private"}],
        "episodes": [],
        "emotional_impressions": [],
    })
    persona = {"name": "测试"}
    hub = MemoryHub(persona, llm, bus, persist_dir=str(tmp_path / "chroma"))

    # Add 5 messages to chat 1 (minimum 4 needed to consolidate)
    for i in range(5):
        hub.add_message(1, "user", f"消息{i}", user_id=100, user_name="用户", context_type="private")

    emitted = []
    async def capture(event):
        emitted.append(event)
    bus.subscribe("memory.consolidated", capture)

    asyncio.run(hub.consolidate(chat_id=1))

    assert len(emitted) == 1
    assert emitted[0].data["chat_id"] == 1
    assert "消息" in emitted[0].data["convo_text"]
    assert len(emitted[0].data["convo_text"]) <= 1500
```

- [ ] **Step 3: Modify memory_hub.py**

In `consolidate()`, after `self.buffers[cid].clear()` and the `logger.info(...)` line, add:

```python
await self.bus.emit("memory.consolidated", {
    "chat_id": cid,
    "convo_text": convo_text[:1500],
})
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_memory_consolidated_event.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add memory_hub.py tests/test_memory_consolidated_event.py
git commit -m "feat: emit memory.consolidated event after each chat consolidation"
```

---

## Task 7: proactive_engine.py — subscribe to inner_state.updated

**Files:**
- Modify: `proactive_engine.py`
- Create: `tests/test_proactive_follow_up.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_proactive_follow_up.py
import asyncio
from unittest.mock import MagicMock
from event_bus import EventBus, Event
from datetime import datetime, timedelta


def test_inner_state_updated_adds_follow_up_triggers():
    from proactive_engine import ProactiveEngine
    bus = EventBus()
    persona = {"name": "测试", "personality": {"extraversion": 0.7}}
    engine = ProactiveEngine(persona, bus)

    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    event = Event("inner_state.updated", {
        "pending_thoughts": [
            {"content": "想问小明项目进度", "target_user_id": 999,
             "urgency": 0.7, "source": "conversation",
             "created_at": now_str, "expires_at": future},
        ]
    })
    asyncio.run(engine._on_inner_state_updated(event))

    assert any(
        t.get("type") == "follow_up" and t.get("content") == "想问小明项目进度"
        for t in engine.pending_triggers
    )


def test_inner_state_updated_ignores_zero_target():
    from proactive_engine import ProactiveEngine
    bus = EventBus()
    persona = {"name": "测试", "personality": {"extraversion": 0.7}}
    engine = ProactiveEngine(persona, bus)

    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    event = Event("inner_state.updated", {
        "pending_thoughts": [
            {"content": "随便想想", "target_user_id": 0,
             "urgency": 0.5, "source": "life_event",
             "created_at": now_str, "expires_at": future},
        ]
    })
    asyncio.run(engine._on_inner_state_updated(event))
    assert not any(t.get("type") == "follow_up" for t in engine.pending_triggers)


def test_inner_state_updated_deduplicates_triggers():
    from proactive_engine import ProactiveEngine
    bus = EventBus()
    persona = {"name": "测试", "personality": {"extraversion": 0.7}}
    engine = ProactiveEngine(persona, bus)

    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    thought = {"content": "想问小明", "target_user_id": 1,
               "urgency": 0.6, "source": "conversation",
               "created_at": now_str, "expires_at": future}
    event = Event("inner_state.updated", {"pending_thoughts": [thought]})
    asyncio.run(engine._on_inner_state_updated(event))
    asyncio.run(engine._on_inner_state_updated(event))  # same event twice
    follow_ups = [t for t in engine.pending_triggers if t.get("type") == "follow_up"]
    assert len(follow_ups) == 1   # not duplicated
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_proactive_follow_up.py -v 2>&1 | head -20
```
Expected: `AttributeError: 'ProactiveEngine' object has no attribute '_on_inner_state_updated'`

- [ ] **Step 3: Modify proactive_engine.py**

In `__init__`, after the existing `self.bus.subscribe(...)` calls, add:
```python
self.bus.subscribe("inner_state.updated", self._on_inner_state_updated)
```

Add the new handler method (after `_on_interesting_content`):
```python
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
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_proactive_follow_up.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add proactive_engine.py tests/test_proactive_follow_up.py
git commit -m "feat: ProactiveEngine subscribes to inner_state.updated for follow_up triggers"
```

---

## Task 8: expression.py — inject inner_state monologue into prompt

**Files:**
- Modify: `expression.py`
- Create: `tests/test_expression_monologue.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_expression_monologue.py
import asyncio
from unittest.mock import MagicMock, AsyncMock


def make_expression(monologue_text: str = ""):
    from expression import ExpressionSynthesizer
    persona = {
        "name": "测试", "background": "测试背景",
        "speaking_style": "随意",
    }
    llm = MagicMock()
    llm.call = AsyncMock(return_value="测试回复")
    memory = MagicMock()
    memory.recall = AsyncMock(return_value={
        "certain": [], "vague": [], "feelings": [], "private_hints": []
    })
    emotion = MagicMock()
    emotion.get_expression_style.return_value = {
        "tone": "正常友好", "message_length": "正常", "emoji_freq": "中",
        "typo_rate": 0.0, "response_speed": "normal", "sticker_mood": "neutral"
    }
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    life_sim = MagicMock()
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司",
        "physical": {"energy": 80}
    }
    stickers = MagicMock()
    stickers.should_use_sticker = AsyncMock(return_value=False)
    browser = MagicMock()
    browser.get_recent_interesting.return_value = []
    rel = MagicMock()
    rel.get_social_context_for_prompt.return_value = "陌生人"
    rel.get.return_value = None
    chat_ctx = MagicMock()
    chat_ctx.get_recent_context.return_value = ""
    inner_state = MagicMock()
    inner_state.get_monologue_text.return_value = monologue_text

    synth = ExpressionSynthesizer(
        persona, llm, memory, emotion, life_sim,
        stickers, browser, rel, chat_ctx,
        inner_state=inner_state,
    )
    return synth, llm


def test_monologue_injected_into_prompt():
    synth, llm = make_expression(monologue_text="有点想回家，工作太烦了")
    asyncio.run(synth.compose_reply(
        user_message="你好", user_id=1, chat_id=1
    ))
    prompt_arg = llm.call.call_args[0][0]
    assert "你心里在转的" in prompt_arg
    assert "有点想回家，工作太烦了" in prompt_arg


def test_no_monologue_section_when_empty():
    synth, llm = make_expression(monologue_text="")
    asyncio.run(synth.compose_reply(
        user_message="你好", user_id=1, chat_id=1
    ))
    prompt_arg = llm.call.call_args[0][0]
    assert "你心里在转的" not in prompt_arg


def test_no_monologue_section_when_inner_state_none():
    from expression import ExpressionSynthesizer
    persona = {
        "name": "测试", "background": "测试背景", "speaking_style": "随意",
    }
    llm = MagicMock()
    llm.call = AsyncMock(return_value="回复")
    memory = MagicMock()
    memory.recall = AsyncMock(return_value={
        "certain": [], "vague": [], "feelings": [], "private_hints": []
    })
    emotion = MagicMock()
    emotion.get_expression_style.return_value = {
        "tone": "正常友好", "message_length": "正常", "emoji_freq": "中",
        "typo_rate": 0.0, "response_speed": "normal", "sticker_mood": "neutral"
    }
    emotion.get_status.return_value = {"valence": 0.1, "arousal": 0.0}
    life_sim = MagicMock()
    life_sim.get_status.return_value = {
        "current_action": "上班", "location": "公司", "physical": {"energy": 80}
    }
    stickers = MagicMock()
    stickers.should_use_sticker = AsyncMock(return_value=False)
    browser = MagicMock()
    browser.get_recent_interesting.return_value = []
    rel = MagicMock()
    rel.get_social_context_for_prompt.return_value = "陌生人"
    rel.get.return_value = None
    chat_ctx = MagicMock()
    chat_ctx.get_recent_context.return_value = ""
    synth = ExpressionSynthesizer(
        persona, llm, memory, emotion, life_sim,
        stickers, browser, rel, chat_ctx,
        inner_state=None,
    )
    asyncio.run(synth.compose_reply(user_message="你好", user_id=1, chat_id=1))
    prompt_arg = llm.call.call_args[0][0]
    assert "你心里在转的" not in prompt_arg
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_expression_monologue.py -v 2>&1 | head -20
```
Expected: `TypeError: __init__() got an unexpected keyword argument 'inner_state'`

- [ ] **Step 3: Modify expression.py**

In `ExpressionSynthesizer.__init__`, add `inner_state=None` as the last parameter:
```python
def __init__(self, persona, llm, memory, emotion, life_sim,
             sticker_engine, browser, relationship_mgr, chat_ctx_mgr,
             inner_state=None):
    ...
    self.inner_state = inner_state
```

In `_compose()`, after the browsing_text block and before the prompt construction, add:
```python
# ---- 内心独白 ----
monologue_text = self.inner_state.get_monologue_text() if self.inner_state else ""
```

In the prompt string `f"""你是{self.persona['name']}...`, after the `## 此刻状态` block insert:
```python
## 你心里在转的
{monologue_text}

```
But only when `monologue_text` is non-empty. Use conditional insertion:
```python
monologue_section = f"\n## 你心里在转的\n{monologue_text}\n" if monologue_text else ""
```
Then reference `{monologue_section}` in the f-string right after the status block.

- [ ] **Step 4: Run tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_expression_monologue.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add expression.py tests/test_expression_monologue.py
git commit -m "feat: inject inner_state monologue into expression prompt"
```

---

## Task 9: orchestrator.py — wire InnerStateManager + _inner_state_loop + proactive.follow_up_fired

**Files:**
- Modify: `orchestrator.py`
- Modify: `config.yaml`
- Create: `tests/test_orchestrator_follow_up.py`

- [ ] **Step 1: Modify orchestrator.py**

**(a) Import:**
```python
from inner_state import InnerStateManager
```

**(b) In `__init__`, after `self.proactive = ProactiveEngine(...)`, add:**
```python
# ---- 内心独白层 ----
inner_state_cfg = config.get("system", {})
self.inner_state = InnerStateManager(
    self.persona, self.llm, self.emotion, self.life_sim, self.bus,
    ttl_hours=inner_state_cfg.get("pending_thought_ttl_hours", 48),
)
```

**(c) Pass `inner_state` to ExpressionSynthesizer:**
```python
self.expression = ExpressionSynthesizer(
    self.persona, self.llm, self.memory, self.emotion,
    self.life_sim, self.stickers, self.browser,
    self.relationships, self.chat_ctx,
    inner_state=self.inner_state,
)
```

**(d) Add `_inner_state_loop()` method:**
```python
async def _inner_state_loop(self):
    cfg = self.config.get("system", {})
    interval_range = cfg.get("inner_state_interval_hours", [2, 3])
    while self.running:
        import random
        hours = random.uniform(interval_range[0], interval_range[1])
        await asyncio.sleep(hours * 3600)
        try:
            await self.inner_state.generate_monologue()
        except Exception as e:
            logger.error(f"Inner state error: {e}", exc_info=True)
```

**(e) In `start_background_tasks()`, add:**
```python
asyncio.create_task(self._inner_state_loop())
```

**(f) In `handle_proactive_trigger()`, fix routing for `follow_up` triggers and emit fired event:**

Replace the existing "select target" block:
```python
# Before (always routes to most-intimate contact):
target = all_rels[0]
target_chat_id = target["user_id"]
```
With:
```python
# Route follow_up to its intended target; other triggers use most-intimate contact
if trigger.get("type") == "follow_up":
    target_user_id = trigger.get("target_user_id", 0)
    target_chat_id = target_user_id   # private chat_id == user_id
    if not target_chat_id or target_chat_id not in self.proactive_callbacks:
        return  # target not registered — skip per spec boundary condition
else:
    target = all_rels[0]
    target_chat_id = target["user_id"]
```

After `await callback(messages)`, add:
```python
if trigger.get("type") == "follow_up":
    await self.bus.emit("proactive.follow_up_fired", {
        "thought_content": trigger.get("content", ""),
    })
```

- [ ] **Step 2: Write unit test for proactive.follow_up_fired emit**

```python
# tests/test_orchestrator_follow_up.py
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


def make_orchestrator(tmp_path):
    import yaml
    from orchestrator import Orchestrator
    cfg = yaml.safe_load(open("/Users/admin/Work/virtual-persona/config.yaml"))
    cfg["llm"]["expression"]["api_key"] = "test"
    cfg["llm"]["utility"]["api_key"] = "test"
    with patch("memory_hub.MemoryHub.__init__", lambda self, *a, **kw: None), \
         patch("vector_store.PersistentClient"):
        orch = Orchestrator.__new__(Orchestrator)
        orch.config = cfg
        orch.persona = cfg["persona"]
        orch.running = False
        from event_bus import EventBus
        orch.bus = EventBus()
        orch.inner_state = MagicMock()
        orch.proactive_callbacks = {}
        orch.relationships = MagicMock()
        orch.relationships.list_all.return_value = [{"user_id": 999}]
        orch.expression = MagicMock()
        orch.expression.compose_proactive = AsyncMock(return_value=[
            {"type": "text", "content": "嗯嗯", "delay": 1}
        ])
        orch.memory = MagicMock()
        orch.memory.add_message = MagicMock()
        return orch


def test_follow_up_fired_emitted_on_success(tmp_path):
    orch = make_orchestrator(tmp_path)
    from proactive_engine import ProactiveEngine
    from datetime import datetime, timedelta
    orch.proactive = ProactiveEngine(
        {"name": "测试", "personality": {"extraversion": 0.7}}, orch.bus
    )
    future = (datetime.now() + timedelta(hours=24)).isoformat()
    now_str = datetime.now().isoformat()
    # inject a follow_up trigger
    orch.proactive.pending_triggers = [{
        "type": "follow_up",
        "urgency": 0.8,
        "content": "想问小明进度",
        "target_user_id": 999,
        "source": "inner_state",
        "time": datetime.now(),
    }]

    messages_sent = []
    async def fake_callback(msgs):
        messages_sent.extend(msgs)
    orch.proactive_callbacks[999] = fake_callback

    emitted = []
    async def capture(event):
        emitted.append(event)
    orch.bus.subscribe("proactive.follow_up_fired", capture)

    asyncio.run(orch.handle_proactive_trigger())

    assert len(messages_sent) > 0
    assert any(e.data["thought_content"] == "想问小明进度" for e in emitted)


def test_follow_up_skipped_when_target_not_registered(tmp_path):
    orch = make_orchestrator(tmp_path)
    from proactive_engine import ProactiveEngine
    from datetime import datetime, timedelta
    orch.proactive = ProactiveEngine(
        {"name": "测试", "personality": {"extraversion": 0.7}}, orch.bus
    )
    orch.proactive.pending_triggers = [{
        "type": "follow_up",
        "urgency": 0.8,
        "content": "想问不在回调里的人",
        "target_user_id": 12345,   # not in proactive_callbacks
        "source": "inner_state",
        "time": datetime.now(),
    }]

    emitted = []
    async def capture(event):
        emitted.append(event)
    orch.bus.subscribe("proactive.follow_up_fired", capture)

    asyncio.run(orch.handle_proactive_trigger())
    assert len(emitted) == 0   # thought not fired, not deleted
```

- [ ] **Step 3: Run test to confirm failure**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_orchestrator_follow_up.py -v 2>&1 | head -30
```
Expected: FAIL (routing not yet implemented)

- [ ] **Step 4: Modify config.yaml**

Add to the `system:` section:
```yaml
  inner_state_interval_hours: [2, 3]   # 随机区间（小时）
  pending_thought_ttl_hours: 48        # pending_thoughts 默认过期时间
```

- [ ] **Step 5: Run orchestrator tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_orchestrator_follow_up.py -v
```
Expected: both tests PASS

- [ ] **Step 6: Smoke-test orchestrator imports and instantiation**

```bash
cd /Users/admin/Work/virtual-persona && python -c "
import yaml
from orchestrator import Orchestrator
cfg = yaml.safe_load(open('config.yaml'))
cfg['llm']['expression']['api_key'] = 'test'
cfg['llm']['utility']['api_key'] = 'test'
o = Orchestrator(cfg)
print('InnerStateManager:', type(o.inner_state).__name__)
print('expression inner_state:', type(o.expression.inner_state).__name__)
print('OK')
"
```
Expected output:
```
InnerStateManager: InnerStateManager
expression inner_state: InnerStateManager
OK
```

- [ ] **Step 7: Run all tests**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/ -v
```
Expected: all tests PASS

- [ ] **Step 8: Commit**

```bash
git add orchestrator.py config.yaml tests/test_orchestrator_follow_up.py
git commit -m "feat: wire InnerStateManager into Orchestrator with background loop and follow_up_fired emit"
```

---

## Task 10: Full test suite pass + final commit

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/ -v
```
Expected: all tests PASS, no errors

- [ ] **Step 2: Verify no regressions in existing test**

```bash
cd /Users/admin/Work/virtual-persona && python -m pytest tests/test_logging_setup.py -v
```
Expected: PASS

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete inner state manager implementation"
```
