"""
LLM 调用封装 —— 支持不同Agent使用不同模型
"""
import json
import asyncio
import logging
from typing import Optional
import httpx
import numpy as np
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

        # 可选 embedding 配置
        emb_cfg = config.get("embedding", {})
        self._embed_url = emb_cfg.get("base_url", "").rstrip("/") + "/embeddings/multimodal" if emb_cfg else None
        self._embed_key = emb_cfg.get("api_key", "")
        self._embed_model = emb_cfg.get("model", "doubao-embedding-vision-251215")

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
        # if json_mode:
        #     kwargs["response_format"] = {"type": "json_object"}

        logger.debug(
            "[LLM-IN] tier=%s model=%s temp=%.2f\n%s",
            tier, model, temperature,
            "\n---\n".join(f"[{m['role']}] {m['content']}" for m in messages),
        )

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(**kwargs, timeout=30)
                output = resp.choices[0].message.content.strip()
                logger.debug("[LLM-OUT] tier=%s\n%s", tier, output)
                return output
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}")
                if attempt < 2:
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

    async def call_vision(
        self,
        prompt: str,
        images: list = None,
        tier: str = "expression",
        system_prompt: str = "",
        temperature: float = 0.85,
        max_tokens: int = 2000,
    ) -> str:
        """Send a multimodal request (text + base64 images). Falls back to plain
        text call when images is empty or None."""
        if not images:
            return await self.call(
                prompt, tier=tier, system_prompt=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )

        client = self.expression_client if tier == "expression" else self.utility_client
        model = self.expression_model if tier == "expression" else self.utility_model

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"},
            })

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        logger.debug(
            "[LLM-IN] tier=%s model=%s vision=True images=%d",
            tier, model, len(images),
        )

        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=30,
                )
                output = resp.choices[0].message.content.strip()
                logger.debug("[LLM-OUT] tier=%s\n%s", tier, output)
                return output
            except Exception as e:
                logger.warning(f"LLM vision call failed (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return ""

    def embed_sync(self, text: str) -> Optional[np.ndarray]:
        """同步调用 Doubao multimodal embedding 接口，返回 float32 向量。
        未配置 embedding 或调用失败时返回 None。"""
        if not self._embed_url or not self._embed_key:
            return None
        try:
            resp = httpx.post(
                self._embed_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._embed_key}",
                },
                json={
                    "model": self._embed_model,
                    "input": [{"type": "text", "text": text}],
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            # Doubao multimodal API returns {"data": {"embedding": [...]}}
            # not {"data": [{"embedding": [...]}]}
            data_field = data["data"]
            if isinstance(data_field, list):
                vec = data_field[0]["embedding"]
            else:
                vec = data_field["embedding"]
            if not vec:
                logger.warning(f"embed_sync: API returned empty embedding. Response: {data}")
                return None
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            logger.warning(f"embed_sync failed: {e}")
            return None