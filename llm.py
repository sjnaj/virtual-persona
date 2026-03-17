"""
LLM 调用封装 —— 支持不同Agent使用不同模型
"""
import json
import asyncio
import logging
from typing import Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: dict):
        # 表达层用强模型
        self.expression_client = AsyncOpenAI(
            api_key=config["expression"]["api_key"],
            base_url=config["expression"]["base_url"],
        )
        self.expression_model = config["expression"]["model"]

        # 工具Agent用便宜模型
        self.utility_client = AsyncOpenAI(
            api_key=config["utility"]["api_key"],
            base_url=config["utility"]["base_url"],
        )
        self.utility_model = config["utility"]["model"]

    async def call(
        self,
        prompt: str,
        tier: str = "utility",
        system_prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> str:
        client = self.expression_client if tier == "expression" else self.utility_client
        model = self.expression_model if tier == "expression" else self.utility_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                await asyncio.sleep(2 ** attempt)

        return ""

    async def call_json(self, prompt: str, tier: str = "utility", **kwargs) -> dict:
        raw = await self.call(prompt, tier=tier, json_mode=True, **kwargs)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            logger.error(f"Failed to parse JSON: {raw[:200]}")
            return {}