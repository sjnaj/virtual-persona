"""
资讯浏览代理 —— 模拟真人刷手机看内容
支持两种模式：
1. RSS/API 模式（稳定，开箱即用）
2. browser-use 模式（真实浏览，需要额外配置）
"""
import random
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import httpx
import feedparser

logger = logging.getLogger(__name__)


class BrowserAgent:
    def __init__(self, persona: dict, llm, event_bus, feeds_config: list):
        self.persona = persona
        self.llm = llm
        self.bus = event_bus
        self.feeds = feeds_config

        # 已经"看过"的内容
        self.seen_content: List[dict] = []
        # 形成了观点的内容
        self.processed_content: List[dict] = []
        # 想分享的内容
        self.shareable_queue: List[dict] = []

        self.last_browse_time: Optional[datetime] = None

    async def should_browse(self, life_status: dict) -> bool:
        """判断当前是否应该"刷手机" """
        action = life_status.get("current_action", "")
        busy_actions = ["开会", "专注做设计", "洗澡", "睡觉", "做饭"]
        idle_actions = ["摸鱼", "坐地铁", "躺床上", "等外卖", "发呆", "赖床"]

        for ba in busy_actions:
            if ba in action:
                return False
        for ia in idle_actions:
            if ia in action:
                return random.random() < 0.6

        return random.random() < 0.15

    async def browse(self):
        """执行一次浏览"""
        self.last_browse_time = datetime.now()

        # 获取内容
        raw_content = await self._fetch_content()
        if not raw_content:
            return

        # 用AI模拟"刷到并形成反应"的过程
        opinions = await self._form_opinions(raw_content)

        for item in opinions:
            self.processed_content.append(item)
            if item.get("would_share"):
                self.shareable_queue.append(item)
                await self.bus.emit("browser.found_interesting", {
                    "content": item["summary"],
                    "reaction": item["reaction"],
                    "share_text": item.get("share_how", ""),
                })

        # 只保留最近的内容
        if len(self.processed_content) > 50:
            self.processed_content = self.processed_content[-50:]

        logger.info(f"[Browser] 浏览了 {len(raw_content)} 条内容, "
                     f"形成 {len(opinions)} 个观点, "
                     f"{sum(1 for o in opinions if o.get('would_share'))} 条想分享")

    async def _fetch_content(self) -> List[dict]:
        """从RSS获取内容（可替换为browser-use）"""
        items = []

        async with httpx.AsyncClient(timeout=15) as client:
            # 随机选1-2个源来"刷"
            selected_feeds = random.sample(
                self.feeds, min(2, len(self.feeds))
            )

            for feed_cfg in selected_feeds:
                try:
                    resp = await client.get(feed_cfg["url"])
                    parsed = feedparser.parse(resp.text)
                    for entry in parsed.entries[:10]:
                        items.append({
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", "")[:200],
                            "source": feed_cfg["name"],
                            "category": feed_cfg.get("category", ""),
                            "link": entry.get("link", ""),
                        })
                except Exception as e:
                    logger.warning(f"[Browser] Feed获取失败 {feed_cfg['name']}: {e}")

        # 随机选取5-8条（模拟真人不会看完所有推送）
        if items:
            count = min(random.randint(5, 8), len(items))
            items = random.sample(items, count)

        return items

    async def _form_opinions(self, raw_content: List[dict]) -> List[dict]:
        """以persona的视角对内容形成个人反应"""
        content_text = "\n".join(
            f"[{c['source']}] {c['title']}: {c['summary'][:100]}"
            for c in raw_content
        )

        prompt = f"""{self.persona['name']}（{self.persona['age']}岁，{self.persona['occupation']}）刚刚在刷手机，看到了这些内容：

{content_text}

以她的性格、年龄、兴趣（{', '.join(self.persona.get('interests', [])[:5])}），对每条内容的真实反应是什么？

要求：
- 大部分内容她只会扫一眼就划走（reaction填"划走了"）
- 只有1-2条会让她有明显反应
- 最多1条她想分享给朋友
- 她可能记错细节
- 不需要深刻见解，要口语化、自然

输出JSON数组：
[{{"summary":"简短描述看到了什么","reaction":"真实内心反应","would_share":false,"share_how":"如果要分享会怎么说（口语化，可为null）"}}]"""

        result = await self.llm.call(prompt, tier="utility", json_mode=True)
        try:
            import json
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return parsed
            return parsed.get("items", parsed.get("content", []))
        except:
            return []

    def pop_shareable(self) -> Optional[dict]:
        """取出一条想分享的内容"""
        if self.shareable_queue:
            return self.shareable_queue.pop(0)
        return None

    def get_recent_interesting(self, n: int = 3) -> List[dict]:
        """获取最近有反应的内容（供表达层参考）"""
        interesting = [
            c for c in self.processed_content
            if c.get("reaction") != "划走了"
        ]
        return interesting[-n:]

    def get_status(self) -> dict:
        return {
            "seen_count": len(self.seen_content),
            "processed_count": len(self.processed_content),
            "shareable_queue": len(self.shareable_queue),
            "last_browse": self.last_browse_time.isoformat() if self.last_browse_time else None,
        }