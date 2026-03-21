"""
群聊行为引擎 —— 决定她在群里何时说话、怎么说话

真人在群聊中的行为模式：
- 不是每条消息都回
- 被@了一定会回
- 感兴趣的话题会插嘴
- 和亲近的人互动更多
- 不熟的人说话她可能潜水
- 群里说话比私聊更克制
- 有时候就是发个表情包/哈哈
"""
import random
import logging
from datetime import datetime, timedelta
from typing import Optional, List

logger = logging.getLogger(__name__)


class GroupBehaviorEngine:
    def __init__(self, persona: dict, llm, relationship_mgr):
        self.persona = persona
        self.llm = llm
        self.rel = relationship_mgr
        self.extraversion = persona["personality"]["extraversion"]

    async def should_respond(
        self,
        chat_window,         # ChatWindow
        sender_id: int,
        message_text: str,
        mentioned_me: bool,
        emotion_state: dict,
    ) -> dict:
        """
        决定是否回复群消息。
        返回: {"should_reply": bool, "reply_mode": str, "reason": str}
        
        reply_mode:
          - "direct"    : 正式回复这条消息
          - "reaction"  : 轻回应（表情包/哈哈/+1）
          - "piggyback" : 借话题展开自己想说的
          - "ignore"    : 不回
        """
        # ===== 硬规则 =====
        
        # 被@了必须回
        if mentioned_me:
            return {
                "should_reply": True,
                "reply_mode": "direct",
                "reason": "被@了",
            }

        # 被直接叫名字
        my_names = [
            self.persona["name"],
            self.persona["name"][1:],  # 去姓
        ]
        for name in my_names:
            if name in message_text:
                return {
                    "should_reply": True,
                    "reply_mode": "direct",
                    "reason": "被叫名字了",
                }

        # ===== 概率规则 =====
        
        base_prob = 0.0
        reasons = []

        # 发送者的亲密度影响
        sender_prof = self.rel.get(sender_id)
        if sender_prof:
            closeness = sender_prof.closeness
            base_prob += closeness * 0.25
            if closeness > 0.6:
                reasons.append(f"和{sender_prof.nickname or sender_prof.display_name}比较熟")
        else:
            base_prob += 0.02  # 不认识的人，很低概率

        # 话题兴趣匹配
        interests = self.persona.get("interests", [])
        for interest in interests:
            keywords = interest.replace("（", " ").replace("）", " ").split()
            for kw in keywords:
                if len(kw) >= 2 and kw in message_text:
                    base_prob += 0.15
                    reasons.append(f"话题相关：{interest}")
                    break

        # 性格影响（外向的人更爱在群里说话）
        base_prob += self.extraversion * 0.08

        # 情绪影响
        valence = emotion_state.get("valence", 0)
        if valence > 0.3:
            base_prob += 0.08  # 心情好更爱说话
        elif valence < -0.3:
            base_prob -= 0.1   # 心情差就潜水

        # 连续沉默太久，概率升高（不能一直潜水）
        silence = chat_window.consecutive_silence
        if silence > 30:
            base_prob += 0.2
        elif silence > 15:
            base_prob += 0.1

        # 刚说过话不久，概率降低（避免刷屏）
        if chat_window.my_last_message_time:
            mins_since = (datetime.now() - chat_window.my_last_message_time).total_seconds() / 60
            if mins_since < 2:
                base_prob *= 0.2
            elif mins_since < 5:
                base_prob *= 0.5

        # 是否是对自己上一条消息的回应
        if chat_window.message_history:
            recent = chat_window.message_history[-3:]
            for m in reversed(recent):
                if m.get("is_me"):
                    # 有人接着我的话说了，更应该回
                    base_prob += 0.15
                    reasons.append("有人在接我的话")
                    break

        # 消息本身的"回应邀请度"
        invitation_markers = ["?", "？", "有人", "谁", "你们觉得", "求推荐", "哈哈哈"]
        for marker in invitation_markers:
            if marker in message_text:
                base_prob += 0.08

        # 最终决策
        prob = max(0.0, min(0.85, base_prob))
        should = random.random() < prob

        # 决定回复模式
        reply_mode = "ignore"
        if should:
            if prob > 0.5 or mentioned_me:
                reply_mode = "direct"
            elif prob > 0.3:
                reply_mode = random.choice(["direct", "reaction", "reaction"])
            else:
                reply_mode = "reaction"

        logger.debug(
            f"[GroupBehavior] 消息来自{sender_id}, prob={prob:.2f}, "
            f"should={should}, mode={reply_mode}, reasons={reasons}"
        )

        return {
            "should_reply": should,
            "reply_mode": reply_mode,
            "reason": "; ".join(reasons) if reasons else "随机参与",
            "probability": round(prob, 3),
        }

    def get_group_speaking_style(self, chat_window, emotion_style: dict) -> dict:
        """群聊中的说话风格调整"""
        # 群里比私聊更克制
        style = dict(emotion_style)
        
        # 消息长度缩短
        length_map = {
            "偏长": "正常",
            "正常": "偏短",
        }
        style["message_length"] = length_map.get(
            style.get("message_length", "正常"),
            style.get("message_length", "正常"),
        )
        
        # 减少分条数
        style["max_segments"] = 2  # 群里最多发2条
        
        # 不撒娇、不说太私人的话
        style["group_filter"] = True
        
        return style