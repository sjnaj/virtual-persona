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
