# tests/test_llm_vision.py
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Minimal 1×1 red pixel JPEG in base64
_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
    "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN"
    "DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA"
    "AAAAAAAAAAAAAP/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA"
    "/9oADAMBAAIRAxEAPwCwABmX/9k="
)

_TINY_WEBP_B64 = "UklGRlYAAABXRUJQVlA4IEoAAADQAQCdASoBAAEAAkA4JZACdAEO/gHOAAA="


def _make_llm():
    from llm import LLMClient
    cfg = {
        "expression": {"api_key": "test", "base_url": "https://api.test", "model": "test-model"},
        "utility": {"api_key": "test", "base_url": "https://api.test", "model": "cheap-model"},
    }
    return LLMClient(cfg)


def test_call_vision_builds_multimodal_content():
    """call_vision sends a content list with text + image_url nodes."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "看起来是红色"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    images = [{"mime_type": "image/jpeg", "base64": _TINY_JPEG_B64, "label": "photo"}]
    result = asyncio.run(llm.call_vision("这是什么颜色？", images=images, tier="expression"))

    assert result == "看起来是红色"
    msg = captured["messages"][-1]
    assert isinstance(msg["content"], list)
    text_nodes = [c for c in msg["content"] if c["type"] == "text"]
    image_nodes = [c for c in msg["content"] if c["type"] == "image_url"]
    assert len(text_nodes) == 1
    assert text_nodes[0]["text"] == "这是什么颜色？"
    assert len(image_nodes) == 1
    assert image_nodes[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_call_vision_label_field_ignored():
    """label field in image dict is not included in the API call."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    images = [{"mime_type": "image/webp", "base64": _TINY_WEBP_B64, "label": "sticker"}]
    asyncio.run(llm.call_vision("看一下", images=images, tier="expression"))

    image_node = [c for c in captured["messages"][-1]["content"] if c["type"] == "image_url"][0]
    assert "label" not in image_node
    assert "label" not in image_node["image_url"]


def test_call_vision_empty_images_falls_back_to_plain_call():
    """With no images, call_vision delegates to plain call() with same params."""
    llm = _make_llm()

    called_with = {}
    async def fake_create(**kwargs):
        called_with.update(kwargs)
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    asyncio.run(llm.call_vision("hello", images=[], tier="expression",
                                system_prompt="sys", temperature=0.5, max_tokens=100))

    msg = called_with["messages"][-1]
    # plain text fallback: content is a string, not a list
    assert isinstance(msg["content"], str)
    assert msg["content"] == "hello"
    assert called_with["temperature"] == 0.5
    assert called_with["max_tokens"] == 100


def test_call_vision_none_images_falls_back():
    """call_vision(images=None) also falls back to plain text."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    asyncio.run(llm.call_vision("hello", images=None, tier="expression"))

    msg = captured["messages"][-1]
    assert isinstance(msg["content"], str)


def test_call_vision_system_prompt_forwarded():
    """system_prompt appears in messages when provided."""
    llm = _make_llm()

    captured = {}
    async def fake_create(**kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    llm.expression_client.chat.completions.create = fake_create

    images = [{"mime_type": "image/jpeg", "base64": _TINY_JPEG_B64, "label": "photo"}]
    asyncio.run(llm.call_vision("q", images=images, tier="expression", system_prompt="你是林小晴"))

    system_msgs = [m for m in captured["messages"] if m["role"] == "system"]
    assert system_msgs and system_msgs[0]["content"] == "你是林小晴"
