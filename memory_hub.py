"""
增强版记忆中枢 —— 记忆带有"来源上下文"标记
她知道什么是A私聊里说的，什么是群里公开说的
"""
import math
import json
import random
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import vector_store as chromadb

logger = logging.getLogger(__name__)


class MemoryHub:
    def __init__(self, persona: dict, llm, event_bus,
                 persist_dir: str = "./data/chroma"):
        self.persona = persona
        self.llm = llm
        self.bus = event_bus

        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.episodic = self.chroma.get_or_create_collection(
            name="episodic", metadata={"hnsw:space": "cosine"}
        )
        self.semantic = self.chroma.get_or_create_collection(
            name="semantic", metadata={"hnsw:space": "cosine"}
        )
        self.emotional = self.chroma.get_or_create_collection(
            name="emotional", metadata={"hnsw:space": "cosine"}
        )

        # 每个聊天窗口的缓冲区（分开管理）
        self.buffers: Dict[int, List[dict]] = {}  # chat_id → messages
        self.last_message_time: Optional[datetime] = None

    def add_message(self, chat_id: int, role: str, content: str,
                    user_id: int = 0, user_name: str = "",
                    context_type: str = "private"):
        """
        context_type: "private" / "group"
        """
        if chat_id not in self.buffers:
            self.buffers[chat_id] = []
        self.buffers[chat_id].append({
            "role": role,
            "content": content,
            "user_id": user_id,
            "user_name": user_name,
            "context_type": context_type,
            "time": datetime.now().isoformat(),
        })
        if role == "user":
            self.last_message_time = datetime.now()

    async def consolidate(self, chat_id: int = None):
        """
        整合记忆。
        关键增强：记忆会标记来源上下文（私聊/群聊）和涉及的人。
        """
        targets = [chat_id] if chat_id else list(self.buffers.keys())

        for cid in targets:
            buffer = self.buffers.get(cid, [])
            if len(buffer) < 4:
                continue

            context_type = buffer[0].get("context_type", "private") if buffer else "private"

            convo_text = "\n".join(
                f"{m.get('user_name', '对方') if m['role']=='user' else self.persona['name']}: {m['content']}"
                for m in buffer
            )

            # 识别对话中涉及的人
            participants = list(set(
                m.get("user_name", "") for m in buffer if m["role"] == "user"
            ))

            prompt = f"""从以下对话中提取记忆（{self.persona['name']}的视角）。

对话来源：{"群聊" if context_type == "group" else "私聊"}
{"参与者：" + "、".join(participants) if len(participants) > 1 else ""}

对话：
{convo_text}

输出JSON：
{{
  "facts": [
    {{"fact": "信息", "about_person": "关于谁", "confidence": 0.9, "source_privacy": "private/public"}}
  ],
  "episodes": [
    {{"summary": "事件概述", "people_involved": ["谁"], "emotional_weight": 0.5, "source_privacy": "private/public"}}
  ],
  "emotional_impressions": [
    {{"feeling": "感受", "about_person": "关于谁", "intensity": 0.5}}
  ]
}}

source_privacy说明：
- "private" = 这是在私聊中得知的，不应该在其他场合提起
- "public" = 这是在群里或公开场合说的，可以在其他场合提起"""

            result = await self.llm.call_json(prompt, tier="utility")
            if not result:
                continue

            now_str = datetime.now().isoformat()

            for i, fact in enumerate(result.get("facts", [])):
                self.semantic.add(
                    documents=[fact["fact"]],
                    metadatas=[{
                        "confidence": fact.get("confidence", 0.7),
                        "about_person": fact.get("about_person", ""),
                        "source_privacy": fact.get("source_privacy", "private"),
                        "source_chat_id": str(cid),
                        "created": now_str,
                        "type": "fact",
                    }],
                    ids=[f"fact_{cid}_{now_str}_{i}"],
                )

            for i, ep in enumerate(result.get("episodes", [])):
                self.episodic.add(
                    documents=[ep["summary"]],
                    metadatas=[{
                        "emotional_weight": ep.get("emotional_weight", 0.5),
                        "people_involved": json.dumps(ep.get("people_involved", [])),
                        "source_privacy": ep.get("source_privacy", "private"),
                        "source_chat_id": str(cid),
                        "created": now_str,
                        "accuracy": 1.0,
                        "type": "episode",
                    }],
                    ids=[f"episode_{cid}_{now_str}_{i}"],
                )

            for i, em in enumerate(result.get("emotional_impressions", [])):
                self.emotional.add(
                    documents=[em["feeling"]],
                    metadatas=[{
                        "intensity": em.get("intensity", 0.5),
                        "about_person": em.get("about_person", ""),
                        "created": now_str,
                        "type": "emotional",
                    }],
                    ids=[f"emotion_{cid}_{now_str}_{i}"],
                )

            self.buffers[cid].clear()
            logger.info(f"[Memory] 整合完成 chat_id={cid}")
            await self.bus.emit("memory.consolidated", {
                "chat_id": cid,
                "convo_text": convo_text[:1500],
            })

    async def recall(self, query: str, current_chat_context: str = "private",
                     about_person: str = "", n_results: int = 5) -> dict:
        """
        增强版记忆检索 —— 考虑隐私边界。
        在群聊中检索时，会过滤掉来自私聊的敏感记忆。
        """
        results = {"certain": [], "vague": [], "feelings": [], "private_hints": []}

        if self.semantic.count() > 0:
            n = min(n_results * 2, self.semantic.count())
            sem_results = self.semantic.query(query_texts=[query], n_results=n)
            for doc, meta in zip(sem_results["documents"][0], sem_results["metadatas"][0]):
                conf = meta.get("confidence", 0.7)
                created = datetime.fromisoformat(meta["created"])
                days_ago = (datetime.now() - created).days
                decay = max(0.3, 1.0 - days_ago * 0.01)
                effective_conf = conf * decay

                source_privacy = meta.get("source_privacy", "private")

                # 隐私过滤
                if current_chat_context == "group" and source_privacy == "private":
                    # 群聊中不直接使用私聊记忆，但标记为"你知道但不该说"
                    if effective_conf > 0.5:
                        results["private_hints"].append(doc)
                    continue

                if effective_conf > 0.7:
                    results["certain"].append(doc)
                elif effective_conf > 0.4:
                    results["vague"].append(doc)

        if self.episodic.count() > 0:
            n = min(n_results, self.episodic.count())
            ep_results = self.episodic.query(query_texts=[query], n_results=n)
            for doc, meta in zip(ep_results["documents"][0], ep_results["metadatas"][0]):
                ew = meta.get("emotional_weight", 0.5)
                created = datetime.fromisoformat(meta["created"])
                days_ago = (datetime.now() - created).days
                retention = 0.6 * math.exp(-0.05 * days_ago) + 0.4 * ew

                source_privacy = meta.get("source_privacy", "private")
                if current_chat_context == "group" and source_privacy == "private":
                    if retention > 0.4:
                        results["private_hints"].append(doc)
                    continue

                if retention > 0.5:
                    results["certain"].append(doc)
                elif retention > 0.3:
                    results["vague"].append(doc)

        if self.emotional.count() > 0:
            n = min(3, self.emotional.count())
            em_results = self.emotional.query(query_texts=[query], n_results=n)
            for doc in em_results["documents"][0]:
                results["feelings"].append(doc)

        return results

    async def forget_and_distort(self):
        if self.episodic.count() == 0:
            return
        all_episodes = self.episodic.get()
        for id_, meta in zip(all_episodes["ids"], all_episodes["metadatas"]):
            created = datetime.fromisoformat(meta["created"])
            days_ago = (datetime.now() - created).days
            accuracy = meta.get("accuracy", 1.0)
            new_accuracy = accuracy * (0.95 ** max(0, days_ago - 1))
            ew = meta.get("emotional_weight", 0.5)
            new_accuracy = min(1.0, new_accuracy + ew * 0.1)
            if new_accuracy < 0.15:
                self.episodic.delete(ids=[id_])
            else:
                self.episodic.update(
                    ids=[id_],
                    metadatas=[{**meta, "accuracy": round(new_accuracy, 3)}],
                )

    def get_status(self) -> dict:
        total_buffer = sum(len(b) for b in self.buffers.values())
        return {
            "episodic_count": self.episodic.count(),
            "semantic_count": self.semantic.count(),
            "emotional_count": self.emotional.count(),
            "buffer_size": total_buffer,
            "active_chats": len(self.buffers),
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
        }
    