"""
关系管理器 —— 每个人在她心里都有一个独立的"印象档案"
她对不同人有不同的称呼、亲密度、聊天习惯、信任边界
"""
import json
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class RelationshipProfile:
    """她心中对某个人的全部认知"""
    user_id: int
    # ---- 基础信息（她了解到的） ----
    display_name: str = ""            # 她对这个人的称呼
    known_names: list = field(default_factory=list)  # 这个人用过的名字
    perceived_gender: str = "未知"
    perceived_age_range: str = "未知"
    
    # ---- 关系定位 ----
    role: str = "陌生人"               # 陌生人/网友/朋友/好朋友/暧昧对象/恋人/同事/长辈/...
    role_confidence: float = 0.5      # 她对这个定位的确信度
    nickname: str = ""                # 她给对方起的昵称/常用称呼
    how_they_met: str = ""            # 怎么认识的
    
    # ---- 关系维度（0~1） ----
    closeness: float = 0.1            # 亲密度
    trust: float = 0.2                # 信任度
    respect: float = 0.5              # 尊重/敬畏程度
    comfort: float = 0.1             # 相处舒适度
    attraction: float = 0.0           # 好感/吸引力
    dependency: float = 0.0           # 依赖程度
    irritation: float = 0.0           # 累积的烦感
    
    # ---- 互动模式 ----
    their_style: str = ""             # 对方的说话风格（她观察到的）
    typical_topics: list = field(default_factory=list)  # 常聊的话题
    boundaries: list = field(default_factory=list)      # 不能聊的话题/雷区
    inside_jokes: list = field(default_factory=list)    # 只有他们之间的梗
    unresolved: list = field(default_factory=list)      # 未解决的事/约定
    
    # ---- 统计 ----
    first_interaction: str = ""
    last_interaction: str = ""
    total_messages: int = 0
    their_messages: int = 0
    my_messages: int = 0
    interaction_days: int = 0
    avg_daily_messages: float = 0.0
    longest_silence_hours: float = 0.0
    
    # ---- 情感账本 ----
    positive_events: list = field(default_factory=list)  # 开心的事
    negative_events: list = field(default_factory=list)  # 不愉快的事
    emotional_debt: float = 0.0       # 负面积累（争吵/冷战）
    
    # ---- 她的主观标签 ----
    personal_tags: list = field(default_factory=list)    # ["话多","有趣","有点直男","深夜emo"]
    
    # ---- 元信息 ----
    is_ai: bool = False               # 她是否知道/怀疑对方是AI
    is_group_only: bool = False        # 是否只在群里见过

    def clamp(self):
        for attr in ["closeness", "trust", "respect", "comfort",
                      "attraction", "dependency", "irritation"]:
            val = getattr(self, attr)
            setattr(self, attr, max(0.0, min(1.0, val)))


class RelationshipManager:
    SAVE_PATH = "./data/relationships.json"

    def __init__(self, persona: dict, llm, event_bus):
        self.persona = persona
        self.llm = llm
        self.bus = event_bus
        self.profiles: Dict[int, RelationshipProfile] = {}
        self._load()

    def _load(self):
        path = Path(self.SAVE_PATH)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for uid_str, prof_data in data.items():
                uid = int(uid_str)
                self.profiles[uid] = RelationshipProfile(**{
                    k: v for k, v in prof_data.items()
                    if k in RelationshipProfile.__dataclass_fields__
                })
            logger.info(f"[Relationship] 加载了 {len(self.profiles)} 段关系")

    def _save(self):
        path = Path(self.SAVE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for uid, prof in self.profiles.items():
            data[str(uid)] = asdict(prof)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_or_create(self, user_id: int, display_name: str = "") -> RelationshipProfile:
        if user_id not in self.profiles:
            self.profiles[user_id] = RelationshipProfile(
                user_id=user_id,
                display_name=display_name or f"用户{user_id}",
                first_interaction=datetime.now().isoformat(),
            )
            logger.info(f"[Relationship] 新建关系档案: {display_name} ({user_id})")
        prof = self.profiles[user_id]
        if display_name and display_name != prof.display_name:
            if prof.display_name not in prof.known_names:
                prof.known_names.append(prof.display_name)
            prof.display_name = display_name
        return prof

    def get(self, user_id: int) -> Optional[RelationshipProfile]:
        return self.profiles.get(user_id)

    async def update_after_conversation(
        self, user_id: int, conversation_text: str, sentiment: float
    ):
        """
        对话结束后，让AI以她的视角更新对这个人的印象。
        不是每条消息都更新，而是一段对话后批量更新。
        """
        prof = self.get(user_id)
        if not prof:
            return

        prof.last_interaction = datetime.now().isoformat()
        prof.total_messages += 1
        prof.their_messages += 1

        # 计算互动天数
        try:
            first = datetime.fromisoformat(prof.first_interaction)
            prof.interaction_days = max(1, (datetime.now() - first).days)
            prof.avg_daily_messages = prof.total_messages / prof.interaction_days
        except:
            pass

        # 基础亲密度变化（互动本身增加亲密度，但有递减）
        closeness_gain = 0.002 * (1.0 - prof.closeness * 0.5)
        prof.closeness += closeness_gain

        # 情感影响
        if sentiment > 0.2:
            prof.trust += 0.003
            prof.comfort += 0.005
            prof.attraction += 0.002
            prof.irritation = max(0, prof.irritation - 0.005)
        elif sentiment < -0.3:
            prof.irritation += 0.01
            prof.comfort -= 0.003

        prof.clamp()

        # 定期深度更新（每20次对话）
        if prof.total_messages % 20 == 0:
            await self._deep_reflection(prof, conversation_text)

        self._save()

    async def _deep_reflection(self, prof: RelationshipProfile, recent_convo: str):
        """定期让AI深度反思对某人的印象"""
        current_tags = ", ".join(prof.personal_tags) if prof.personal_tags else "暂无"
        current_topics = ", ".join(prof.typical_topics) if prof.typical_topics else "暂无"

        prompt = f"""你是{self.persona['name']}。回顾你和"{prof.display_name}"的关系。

你们的互动数据：
- 认识天数：{prof.interaction_days}天
- 总消息数：{prof.total_messages}
- 当前关系：{prof.role}
- 亲密度：{prof.closeness:.2f}
- 你给ta的标签：{current_tags}
- 常聊话题：{current_topics}

最近的对话（"你"是你自己说的话，"{prof.display_name}"是对方说的话）：
{recent_convo[:1500]}

以你（{self.persona['name']}）的主观视角，更新你对ta的印象。
注意：new_tags 只能根据对方自己说的内容、提问、表达方式来推断ta的身份/职业/性格，不能把你自己说的内容误归为对方的特征，更新tag里面和对话内容明显冲突的地方。

输出JSON：
{{
  "role_update": "你觉得你们现在是什么关系（用一个自然的词）",
  "nickname": "你会怎么称呼ta（可以是名字、昵称、不变则填原来的）",
  "new_tags": ["对ta的印象标签，3-8个"],
  "new_topics": ["你们经常聊什么，3-8个"],
  "new_boundaries": ["有什么你不想和ta聊的"],
  "inside_jokes": ["你们之间的梗/暗号"],
  "their_style": "ta说话什么风格（简短描述）",
  "emotional_note": "你现在对ta的感觉（一句话内心独白）"
}}"""

        result = await self.llm.call_json(prompt, tier="utility")
        if not result:
            return

        if result.get("role_update"):
            prof.role = result["role_update"]
        if result.get("nickname"):
            prof.nickname = result["nickname"]
        if result.get("new_tags"):
            prof.personal_tags = result["new_tags"]
        if result.get("new_topics"):
            prof.typical_topics = result["new_topics"]
        if result.get("new_boundaries"):
            prof.boundaries = result["new_boundaries"]
        if result.get("inside_jokes"):
            prof.inside_jokes = result["inside_jokes"]
        if result.get("their_style"):
            prof.their_style = result["their_style"]

        logger.info(f"[Relationship] 深度更新 {prof.display_name}: "
                     f"role={prof.role}, tags={prof.personal_tags}")

    def get_social_context_for_prompt(self, user_id: int) -> str:
        """为表达层生成关系上下文描述"""
        prof = self.get(user_id)
        if not prof:
            return "你不认识这个人，这是第一次互动。对陌生人保持礼貌但有距离感。"

        lines = [f"对方：{prof.display_name}"]
        if prof.nickname:
            lines.append(f"你平时叫ta：{prof.nickname}")
        lines.append(f"你们的关系：{prof.role}")
        lines.append(f"亲密度：{prof.closeness:.1f}/1.0（{'很亲近' if prof.closeness>0.7 else '还行' if prof.closeness>0.4 else '不太熟'}）")
        lines.append(f"信任度：{prof.trust:.1f}/1.0")
        lines.append(f"舒适度：{prof.comfort:.1f}/1.0")

        if prof.attraction > 0.3:
            lines.append(f"你对ta有好感（{prof.attraction:.1f}）")
        if prof.irritation > 0.3:
            lines.append(f"你最近对ta有点烦（{prof.irritation:.1f}）")

        if prof.personal_tags:
            lines.append(f"你觉得ta是个：{'、'.join(prof.personal_tags)}")
        if prof.their_style:
            lines.append(f"ta说话风格：{prof.their_style}")
        if prof.typical_topics:
            lines.append(f"你们常聊：{'、'.join(prof.typical_topics)}")
        if prof.inside_jokes:
            lines.append(f"你们之间的梗：{'、'.join(prof.inside_jokes)}")
        if prof.boundaries:
            lines.append(f"你不想和ta聊：{'、'.join(prof.boundaries)}")
        if prof.unresolved:
            lines.append(f"还没聊完的事：{'、'.join(prof.unresolved)}")

        # 互动模式描述
        if prof.avg_daily_messages > 20:
            lines.append("你们每天聊很多，关系很好")
        elif prof.avg_daily_messages > 5:
            lines.append("你们经常聊天")
        elif prof.avg_daily_messages > 1:
            lines.append("你们偶尔聊聊")
        else:
            lines.append("你们不常聊天")

        return "\n".join(lines)

    def get_group_social_map(self, group_member_ids: List[int]) -> str:
        """获取群聊中所有人的关系概览"""
        lines = []
        for uid in group_member_ids:
            prof = self.get(uid)
            if prof:
                tag_str = f"（{'、'.join(prof.personal_tags[:2])}）" if prof.personal_tags else ""
                lines.append(
                    f"- {prof.nickname or prof.display_name}：{prof.role}，"
                    f"亲密度{prof.closeness:.1f}{tag_str}"
                )
            else:
                lines.append(f"- 用户{uid}：不认识")
        return "\n".join(lines) if lines else "群里的人你都不认识"

    def list_all(self) -> List[dict]:
        result = []
        for uid, prof in self.profiles.items():
            result.append({
                "user_id": uid,
                "name": prof.nickname or prof.display_name,
                "role": prof.role,
                "closeness": round(prof.closeness, 2),
                "trust": round(prof.trust, 2),
                "total_messages": prof.total_messages,
            })
        result.sort(key=lambda x: x["closeness"], reverse=True)
        return result