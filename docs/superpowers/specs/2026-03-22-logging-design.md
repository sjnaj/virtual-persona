# Logging System Design

**Date:** 2026-03-22
**Status:** Approved

## Overview

Add file-based logging to the virtual-persona bot. Currently all log output goes to the console only. This change adds two log files under a `log/` directory without modifying any existing logger declarations.

## Requirements

- `log/info.log` — captures all INFO and above messages (INFO, WARNING, ERROR, CRITICAL — full audit trail)
- `log/warn.log` — captures WARNING and above messages (WARNING, ERROR, CRITICAL — fast problem triage)
- No log rotation; file management is handled externally
- No changes to any file except `main.py`
- Bot must be launched from the project root directory (`python main.py`) for relative log paths to resolve correctly

## Directory Structure

```
virtual-persona/
└── log/
    ├── info.log
    └── warn.log
```

The `log/` directory is created automatically at startup via `os.makedirs("log", exist_ok=True)`.

## Implementation

### `main.py` — `setup_logging()` changes

Add two `FileHandler` instances to the root logger alongside the existing `StreamHandler`:

| Handler | Destination | Min Level | Format |
|---------|-------------|-----------|--------|
| `StreamHandler` | console | INFO | `HH:MM:SS [name] LEVEL: message` |
| `FileHandler("log/info.log")` | file | INFO | `YYYY-MM-DD HH:MM:SS [name] LEVEL: message` |
| `FileHandler("log/warn.log")` | file | WARNING | `YYYY-MM-DD HH:MM:SS [name] LEVEL: message` |

File handlers use `mode="a"` (append) and `encoding="utf-8"`.

Third-party noisy libraries (httpx, openai, telegram, httpcore, chromadb) remain suppressed at WARNING level.

### No changes to other files

All 15 modules already declare `logger = logging.getLogger(__name__)`. They inherit the root logger configuration automatically.

## Log Format

Console (short):
```
14:23:05 [bot] INFO: [Bot] 启动...
```

File (full date):
```
2026-03-22 14:23:05 [bot] INFO: [Bot] 启动...
2026-03-22 14:23:07 [orchestrator] WARNING: LLM call failed (attempt 1): timeout
```

## Scope

- **Files changed:** `main.py` only
- **Files added:** `log/.gitkeep` (to track the directory in git)
- **Files unchanged:** all other `.py` files
