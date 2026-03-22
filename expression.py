"""
增强版表达合成层 —— 根据关系+上下文调整表达
核心增加：
  1. 对不同的人说话风格不同
  2. 群聊中有隐私意识
  3. 群聊中可以对话题参与而非只回复一个人
"""
import re
import random
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class ExpressionSynthesizer:
    def __init__(self, persona, llm, memory, emotion, life_sim,
                 sticker_engine, browser, relationship_mgr, chat_ctx_mgr,
                 inner_state=None):
        self.persona = persona
        self.llm = llm
        self.memory = memory
        self.emotion = emotion
        self.life_sim = life_sim
        self.stickers = sticker_engine
        self.browser = browser
        self.rel = relationship_mgr
        self.chat_ctx = chat_ctx_mgr
        self.inner_state = inner_state

    async def compose_reply(
        self,
        user_message: str,
        user_id: int,
        chat_id: int,
        chat_type: str = "private",
        reply_mode: str = "direct",
    ) -> List[dict]:
        return await self._compose(
            user_message=user_message,
            user_id=user_id,
            chat_id=chat_id,
            chat_type=chat_type,
            reply_mode=reply_mode,
        )

    async def compose_proactive(self, trigger: dict, chat_id: int,
                                 user_id: int = 0) -> List[dict]:
        return await self._compose(
            proactive_trigger=trigger,
            user_id=user_id,
            chat_id=chat_id,
            chat_type="private",
        )

    async def _compose(
        self,
        user_message: str = None,
        proactive_trigger: dict = None,
        user_id: int = 0,
        chat_id: int = 0,
        chat_type: str = "private",
        reply_mode: str = "direct",
    ) -> List[dict]:
        now = datetime.now()
        is_group = chat_type in ("group", "supergroup")

        # ---- 收集状态 ----
        emotion_style = self.emotion.get_expression_style()
        life_status = self.life_sim.get_status()
        emotion_status = self.emotion.get_status()

        # ---- 关系上下文 ----
        relationship_desc = self.rel.get_social_context_for_prompt(user_id) if user_id else ""

        # 群聊时获取群内关系全景
        group_social_map = ""
        if is_group:
            window = self.chat_ctx.get_or_create(chat_id, chat_type)
            group_social_map = self.rel.get_group_social_map(window.group_members)

        # ---- 记忆检索 ----
        query = user_message or proactive_trigger.get("content", "")
        memory_context = "group" if is_group else "private"
        memories = await self.memory.recall(
            query, current_chat_context=memory_context
        ) if query else {"certain": [], "vague": [], "feelings": [], "private_hints": []}

        # ---- 记忆文本 ----
        memory_text = ""
        if memories["certain"]:
            memory_text += "你确定记得的：" + "；".join(memories["certain"]) + "\n"
        if memories["vague"]:
            memory_text += "隐约记得（可能有误）：" + "；".join(memories["vague"]) + "\n"
        if memories["feelings"]:
            memory_text += "你的感受：" + "；".join(memories["feelings"]) + "\n"
        if memories.get("private_hints") and is_group:
            memory_text += (
                "\n⚠️ 你从私聊中知道的（不要在群里直接说出来，但可以影响你的态度）：\n"
                + "；".join(memories["private_hints"]) + "\n"
            )
        if not memory_text:
            memory_text = "没有特别相关的记忆"

        # ---- 对话上下文 ----
        convo_text = self.chat_ctx.get_recent_context(chat_id, limit=15)

        # ---- 浏览内容 ----
        recent_browsing = self.browser.get_recent_interesting(3)
        browsing_text = "\n".join(
            f"- {c['summary']}（你觉得：{c['reaction']}）"
            for c in recent_browsing
        ) if recent_browsing else "最近没看到什么特别的"

        # ---- 内心独白 ----
        monologue_text = self.inner_state.get_monologue_text() if self.inner_state else ""
        monologue_section = f"\n## 你心里在转的\n{monologue_text}\n" if monologue_text else ""

        # ---- 构建 prompt ----
        # 群聊和私聊用不同的指令
        if is_group:
            context_instruction = f"""## 这是群聊
群名：{self.chat_ctx.get_or_create(chat_id, chat_type).group_title or '未知'}
群内你认识的人：
{group_social_map}

群聊行为准则：
1. 你说的话所有人都能看到，注意分寸
2. 不要暴露从私聊得知的他人隐私
3. 说话比私聊更简短、更casual
4. 不需要每条消息都回，可以只对感兴趣的话题参与
5. 可以@某人说话
6. 回复模式：{reply_mode}
   - direct = 正经回复
   - reaction = 轻回应（表情包/哈哈/简短感叹）
   - piggyback = 借话题说自己的
"""
        else:
            context_instruction = f"""## 这是你和对方的私聊
{relationship_desc}

私聊行为：
1. 根据亲密度调整语气
2. 亲密的人可以撒娇/吐槽/说心里话
3. 不太熟的人保持友好但有距离感
4. 你可以主动提起你们之间的梗/记忆
"""

        # ---- 关系特化的说话风格 ----
        rel_prof = self.rel.get(user_id)
        style_addon = ""
        if rel_prof:
            if rel_prof.closeness > 0.7:
                style_addon = "你和ta很亲近，可以随意说话、开玩笑、吐槽、撒娇。"
            elif rel_prof.closeness > 0.4:
                style_addon = "你和ta还算熟，说话轻松友好。"
            elif rel_prof.closeness > 0.2:
                style_addon = "你和ta不太熟，保持礼貌但不会太热情。"
            else:
                style_addon = "你和ta几乎不认识，保持客气距离。"
            if rel_prof.irritation > 0.5:
                style_addon += " 你最近对ta有点烦，回复可能不太耐烦。"
            if rel_prof.attraction > 0.5:
                style_addon += " 你对ta有好感，说话会有点在意形象。"

        # ---- 主动触发上下文 ----
        if proactive_trigger:
            ttype = proactive_trigger.get("type", "")
            if ttype == "content_share":
                seen = proactive_trigger.get("seen_content", "")
                reaction = proactive_trigger.get("reaction", "")
                share_how = proactive_trigger.get("content", "")
                proactive_context = (
                    f"你刚才刷手机看到：{seen}\n"
                    f"你的第一反应：{reaction}\n"
                    f"你想分享给对方，大概会这么说：{share_how}"
                )
            else:
                proactive_context = f"原因：{proactive_trigger.get('content', '')}"
        else:
            proactive_context = ""

        # ---- 人设特质 ----
        interests_str = "、".join(self.persona.get("interests", []))
        mbti = self.persona.get("mbti", "")
        _mbti_desc = {
            "ENFP": "热情开放、充满好奇心、喜欢分享、容易被新事物吸引",
            "INFP": "理想主义、善解人意、内敛但情感深厚",
            "ENFJ": "热情有感染力、善于倾听、天生领导者",
            "INTJ": "独立理性、目标明确、不喜欢废话",
        }
        mbti_desc = _mbti_desc.get(mbti, "")
        trait_section = (
            f"## 你的特质\n"
            f"年龄：{self.persona.get('age', '')}岁\n"
            f"MBTI：{mbti}{'（' + mbti_desc + '）' if mbti_desc else ''}\n"
            f"兴趣：{interests_str}\n"
        )

        # ---- 最终 prompt ----
        prompt = f"""你是{self.persona['name']}，一个真实的人在用手机聊天。

## 你是谁
{self.persona['background']}

{trait_section}
## 此刻状态
时间：{now.strftime("%H:%M")}（{'工作日' if now.weekday() < 5 else '周末'}）
正在做：{life_status['current_action']} @ {life_status['location']}
心情：{emotion_style['tone']}（valence={emotion_status['valence']:.1f}）
体力：{life_status['physical']['energy']}/100
{monologue_section}
## 说话风格
{self.persona['speaking_style']}
当前情绪下 → 语气{emotion_style['tone']}，长度{emotion_style['message_length']}，emoji{emotion_style['emoji_freq']}
{style_addon}

{context_instruction}

## 你记得的
{memory_text}

## 最近对话
{convo_text}

## 你刷手机看到的（不要强行提起）
{browsing_text}

## {'对方发来' if user_message else '你想主动说'}
{'"' + user_message + '"' if user_message else proactive_context}

## 输出要求
1. 用 ||| 分隔多条消息
2. 表情包用 [sticker:描述]
3. {'群聊中最多发1-2条，简短为主' if is_group else '可以分多条发'}
4. {'群聊中@某人用「@名字」格式' if is_group else ''}
5. 回复风格要匹配你和对方的关系亲密度
6. 不要说"作为AI"或暴露非人身份

直接输出消息："""

        system_prompt = (
            f"你正在扮演{self.persona['name']}。"
            f"直接以她的身份说话，不要有任何AI痕迹。"
        )

        raw = await self.llm.call(
            prompt, tier="expression",
            system_prompt=system_prompt,
            temperature=0.85,
        )

        if not raw:
            raw = "嗯嗯" if user_message else "在吗～"

        return await self._post_process(raw, emotion_style, is_group)

    async def _post_process(self, raw: str, emotion_style: dict,
                             is_group: bool = False) -> List[dict]:
        segments = [s.strip() for s in raw.split("|||") if s.strip()]
        if not segments:
            segments = [raw.strip()]

        # 群聊限制条数
        if is_group:
            max_seg = emotion_style.get("max_segments", 2)
            segments = segments[:max_seg]

        results = []
        for i, seg in enumerate(segments):
            sticker_match = re.search(r'\[sticker:(.+?)\]', seg)
            text_part = re.sub(r'\[sticker:.+?\]', '', seg).strip()

            if text_part:
                text_part = self._inject_typos(
                    text_part, emotion_style.get("typo_rate", 0.02)
                )
                results.append({
                    "type": "text",
                    "content": text_part,
                    "delay": self._calc_delay(i, text_part, emotion_style, is_group),
                })

            if sticker_match:
                sticker_desc = sticker_match.group(1)
                sticker = await self.stickers.select_sticker({
                    "mood": emotion_style.get("sticker_mood", "neutral"),
                    "reply_text": text_part,
                    "description": sticker_desc,
                })
                if sticker:
                    results.append({
                        "type": "sticker",
                        "file_id": sticker["file_id"],
                        "sticker_type": sticker.get("type", "sticker"),
                        "delay": random.uniform(0.8, 2.0),
                    })

        # 补表情包
        has_sticker = any(r["type"] == "sticker" for r in results)
        if not has_sticker and results:
            last_text = next(
                (r["content"] for r in reversed(results) if r["type"] == "text"), ""
            )
            if await self.stickers.should_use_sticker(last_text, emotion_style):
                sticker = await self.stickers.select_sticker({
                    "mood": emotion_style.get("sticker_mood", "neutral"),
                    "reply_text": last_text,
                })
                if sticker:
                    results.append({
                        "type": "sticker",
                        "file_id": sticker["file_id"],
                        "delay": random.uniform(0.8, 2.0),
                    })

        return results

    def _calc_delay(self, index, text, style, is_group=False):
        if index == 0:
            speed = style.get("response_speed", "normal")
            base = {
                "fast": random.uniform(2, 8),
                "normal": random.uniform(8, 40),
                "slow": random.uniform(40, 120),
                "very_slow": random.uniform(120, 400),
            }.get(speed, random.uniform(8, 40))
            # 群聊中回复更快（不用思考那么多）
            if is_group:
                base *= 0.5
            return base
        else:
            chars = len(text)
            return chars * random.uniform(0.08, 0.15) + random.uniform(0.5, 2.0)

    def _inject_typos(self, text, rate):
        if rate <= 0 or random.random() > rate * 5:
            return text
        typo_pairs = {"的": "地", "地": "的", "在": "再", "再": "在"}
        chars = list(text)
        for i, ch in enumerate(chars):
            if ch in typo_pairs and random.random() < rate:
                chars[i] = typo_pairs[ch]
                break
        return "".join(chars)