"""
表情包系统 —— 采集、管理、上下文选择
在 Telegram 中使用 sticker file_id 发送
"""
import json
import random
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# 表情包库 JSON 结构示例：
# {
#   "stickers": [
#     {
#       "id": "stk_001",
#       "file_id": "CAACAgIAAxkBAAI...",  <-- Telegram sticker file_id
#       "type": "sticker",                 <-- sticker / animation / photo
#       "tags": ["开心", "可爱", "猫"],
#       "mood": "happy",
#       "scene": ["打招呼", "收到好消息"],
#       "usage_count": 5,
#       "added_date": "2025-01-01",
#       "description": "一只橘猫开心地转圈"
#     }
#   ]
# }

MOOD_MAPPING = {
    "happy":            ["开心", "哈哈", "笑", "可爱"],
    "sad":              ["难过", "哭", "委屈", "伤心"],
    "annoyed":          ["无语", "白眼", "烦", "翻白眼"],
    "neutral_positive": ["嗯嗯", "好的", "OK", "收到"],
    "neutral":          ["摸鱼", "发呆", "日常"],
    "excited":          ["冲", "太棒了", "尖叫"],
    "shy":              ["害羞", "捂脸", "嘿嘿"],
    "angry":            ["生气", "打人", "哼"],
}


class StickerEngine:
    def __init__(self, persona: dict, llm, library_path: str = "./data/stickers.json"):
        self.persona = persona
        self.llm = llm
        self.library_path = Path(library_path)
        self.stickers: List[dict] = []
        self.recently_used: List[str] = []

        self._load_library()

    def _load_library(self):
        if self.library_path.exists():
            with open(self.library_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.stickers = data.get("stickers", [])
        else:
            self.library_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_library()
        logger.info(f"[Sticker] 已加载 {len(self.stickers)} 个表情包")

    def _save_library(self):
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump({"stickers": self.stickers}, f, ensure_ascii=False, indent=2)

    def add_sticker(
        self,
        file_id: str,
        sticker_type: str = "sticker",
        tags: List[str] = None,
        mood: str = "neutral",
        description: str = "",
    ) -> dict:
        """管理员通过bot添加表情包到库中"""
        sticker = {
            "id": f"stk_{len(self.stickers):04d}",
            "file_id": file_id,
            "type": sticker_type,
            "tags": tags or [],
            "mood": mood,
            "scene": [],
            "usage_count": 0,
            "added_date": datetime.now().strftime("%Y-%m-%d"),
            "description": description,
        }
        self.stickers.append(sticker)
        self._save_library()
        logger.info(f"[Sticker] 新增表情包: {sticker['id']} ({description})")
        return sticker

    async def should_use_sticker(self, draft_reply: str, emotion_style: dict) -> bool:
        """判断当前回复是否应该配表情包"""
        if not self.stickers:
            return False

        base_prob = 0.35  # 基础概率

        # 情绪影响
        mood = emotion_style.get("sticker_mood", "neutral")
        if mood in ("happy", "excited", "shy"):
            base_prob += 0.15
        elif mood in ("sad", "annoyed"):
            base_prob -= 0.1

        # 消息长度影响：短消息更容易配表情包
        if len(draft_reply) < 10:
            base_prob += 0.15
        elif len(draft_reply) > 80:
            base_prob -= 0.15

        # emoji频率设定影响
        freq = emotion_style.get("emoji_freq", "中")
        if freq == "高":
            base_prob += 0.15
        elif freq in ("低", "很低", "几乎没有"):
            base_prob -= 0.2

        return random.random() < max(0.05, min(0.7, base_prob))

    async def select_sticker(self, context: dict) -> Optional[dict]:
        """根据上下文选择最合适的表情包"""
        if not self.stickers:
            return None

        mood = context.get("mood", "neutral")
        reply_text = context.get("reply_text", "")

        # 第一步：按 mood 粗筛
        mood_keywords = MOOD_MAPPING.get(mood, [])
        candidates = []
        for stk in self.stickers:
            score = 0.0
            # mood 匹配
            if stk.get("mood") == mood:
                score += 3.0
            # tag 匹配
            for tag in stk.get("tags", []):
                if tag in mood_keywords:
                    score += 2.0
                if tag in reply_text:
                    score += 1.5

            # 使用频率偏好（真人有常用表情包）
            score += min(stk.get("usage_count", 0) * 0.1, 1.0)

            # 新表情包有新鲜感
            try:
                days_since_add = (datetime.now() - datetime.strptime(stk["added_date"], "%Y-%m-%d")).days
                if days_since_add < 7:
                    score += 0.5
            except:
                pass

            # 避免最近刚用过的
            if stk["id"] in self.recently_used[-3:]:
                score *= 0.2

            if score > 0:
                candidates.append((stk, score))

        if not candidates:
            # 没有好的匹配，随机选一个mood接近的或者不选
            general = [s for s in self.stickers if s.get("mood") in (mood, "neutral", "neutral_positive")]
            if general:
                candidates = [(random.choice(general), 1.0)]
            else:
                return None

        # 加权随机选择（不是选最高分的，加一点随机性）
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:5]
        weights = [c[1] for c in top]
        selected = random.choices(top, weights=weights, k=1)[0][0]

        # 更新使用记录
        selected["usage_count"] = selected.get("usage_count", 0) + 1
        self.recently_used.append(selected["id"])
        if len(self.recently_used) > 20:
            self.recently_used = self.recently_used[-20:]
        self._save_library()

        return selected

    async def ai_tag_sticker(self, description: str) -> dict:
        """用AI给新表情包自动打标签"""
        prompt = f"""给这个表情包打标签。
表情包描述：{description}

输出JSON：
{{"tags": ["标签1", "标签2"], "mood": "happy/sad/annoyed/neutral/neutral_positive/excited/shy/angry", "scenes": ["适用场景1", "适用场景2"]}}"""

        return await self.llm.call_json(prompt, tier="utility")