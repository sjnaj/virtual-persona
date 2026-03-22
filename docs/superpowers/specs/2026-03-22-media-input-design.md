# Media Input Feature Design

**Date:** 2026-03-22
**Status:** Approved

## Overview

Add the ability for the bot to receive and visually understand user-sent photos, stickers, and GIF animations via base64-encoded image content passed to the vision-capable expression LLM. Videos are gracefully declined with a persona-appropriate reply.

## Supported Media Types

| Telegram Type | Handling | Notes |
|---|---|---|
| `photo` | Download largest available size → base64 JPEG | Standard image |
| `sticker` (static) | Download → base64 WEBP | `.webp` format, Claude supports it |
| `sticker` (animated `.tgs`) | Decline gracefully | TGS is Lottie/JSON, no image data |
| `animation` (GIF/MP4) | Download → base64 first frame or inline | `animation.file_id`, treat as image |
| `video` | Polite refusal reply | LLM does not support raw video |
| `document` (image mime) | Out of scope for this feature | Not implemented |

## Data Structure

A `media` field is threaded through the call chain as `list[dict] | None`:

```python
{
    "mime_type": "image/jpeg",  # image/png, image/webp, image/gif
    "base64": "<base64_string>",
    "label": "photo"            # photo | sticker | animation
}
```

## Architecture

### Call Chain

```
Telegram update
    ↓
bot.py: _handle_media_message()
    ↓  downloads file, encodes to base64
orchestrator.py: handle_message(text, ..., media=[...])
    ↓  passes through unchanged
expression.py: compose_reply(..., media=[...])
    ↓  builds prompt + calls vision LLM
llm.py: call_vision(prompt, images, tier)
    ↓  OpenAI multimodal content format
Claude Sonnet (vision)
```

### Video Flow

```
Telegram video update
    ↓
bot.py: _handle_video_message()
    ↓  no download needed
expression.py: compose_reply(text="[视频]", ...)
    ↓  prompt instructs persona to politely say she can't see videos
LLM → natural persona-style "video decline"
```

## File-by-File Changes

### `llm.py`

New method `call_vision(prompt, images, tier, ...)`:

```python
async def call_vision(
    self,
    prompt: str,
    images: list[dict],   # [{mime_type, base64}, ...]
    tier: str = "expression",
    system_prompt: str = "",
    temperature: float = 0.85,
    max_tokens: int = 2000,
) -> str:
```

- Builds `content` as a list: text node first, then one `image_url` node per image using `data:<mime>;base64,<b64>` URI scheme
- Falls back to plain `call()` if `images` is empty
- Only the `expression` tier is expected to be called with images (expression model = Claude Sonnet 4.6, which supports vision)

### `bot.py`

New methods:

- `_download_and_encode(bot, file_id, label) -> dict | None` — downloads file via `bot.get_file()`, reads bytes, base64-encodes, returns media dict
- `_handle_media_message(update, context)` — handles photo/sticker/animation; extracts text caption if any; builds `media` list; calls orchestrator
- `_handle_video_message(update, context)` — calls orchestrator with `text="[用户发来了视频]"` and no media (orchestrator passes this as a hint to expression to decline politely)

Registered handlers in `run()`:

```python
app.add_handler(MessageHandler(
    (filters.PHOTO | filters.Sticker.ALL | filters.ANIMATION) & ~filters.COMMAND,
    self._handle_media_message,
))
app.add_handler(MessageHandler(
    filters.VIDEO & ~filters.COMMAND,
    self._handle_video_message,
))
```

Animated stickers (`.tgs`): detected via `sticker.is_animated` → treated as video (polite decline).

### `orchestrator.py`

`handle_message()` signature change:

```python
async def handle_message(
    self,
    text: str,
    user_id: int,
    user_name: str,
    chat_id: int,
    chat_type: str = "private",
    group_title: str = "",
    mentioned_me: bool = False,
    media: list = None,          # NEW
) -> Optional[list]:
```

- `media` is passed through to `expression.compose_reply()`
- When recording to memory/chat context, if `media` is present: append a text label `[图片]` / `[表情包]` to the stored message text so memory/relationship layers remain text-only

### `expression.py`

`compose_reply()` and `_compose()` signature change:

```python
async def compose_reply(
    self,
    user_message: str,
    user_id: int,
    chat_id: int,
    chat_type: str = "private",
    reply_mode: str = "direct",
    media: list = None,          # NEW
) -> List[dict]:
```

In `_compose()`:
- If `media` is present, append to the "对方发来" section:
  ```
  ## 对方发来
  "<caption text>" [附带图片：1张photo]
  ```
- Call `self.llm.call_vision(prompt, images=media, tier="expression", ...)` instead of `self.llm.call(...)`
- If no media, continue using `self.llm.call(...)` unchanged

## Error Handling

- Download failure (network timeout, file too large): log warning, fall back to text-only reply (treat as if no media)
- Telegram file size limit: Telegram caps bot downloads at 20MB; files over this size will fail gracefully
- Unsupported mime type from Telegram: skip encoding, log debug
- LLM vision call failure: retry logic already in `llm.call()` is inherited; same 3-attempt pattern

## Memory & Relationship Impact

Media is **not** stored as binary in memory. The memory/relationship layers receive a text label (e.g., `[发来了图片]`) appended to the user message text. This keeps ChromaDB text-based and avoids bloating vector embeddings with non-semantic content.

## Testing Notes

- Unit-testable: `_download_and_encode` can be tested by mocking `bot.get_file()`
- `llm.call_vision` can be tested with a small 1x1 pixel base64 JPEG
- Integration: send a photo via Telegram and verify the reply references image content
