# Design: Activate Unused Persona Config Fields

**Date:** 2026-03-22
**Status:** Approved

## Problem

`config.yaml` defines rich persona metadata that is largely ignored at runtime. Unused fields mean the simulated character behaves inconsistently with her stated identity.

### Unused Fields (before this change)

| Field | Value | Status |
|-------|-------|--------|
| `age` | 24 | ❌ unused |
| `gender` | 女 | ❌ unused (ignored per design decision) |
| `mbti` | ENFP | ❌ unused |
| `personality.agreeableness` | 0.75 | ❌ unused (ignored per design decision) |
| `personality.openness` | 0.8 | ❌ unused (ignored per design decision) |
| `interests` | 7 items | ❌ unused |
| `daily_patterns.work_start` | [9,10] | ❌ unused |
| `daily_patterns.work_end` | [18,19] | ❌ unused |
| `daily_patterns.lunch` | [11,13] | ❌ unused |
| `sticker_preferences` | 4 items | ❌ unused (ignored per design decision) |

### Per-field decisions

- **age** → inject into expression prompt
- **mbti** → inject into expression prompt (text context only, not numerical)
- **interests** → inject into expression prompt + life_simulator tick prompt
- **daily_patterns.work_start/work_end/lunch** → inject as dynamic schedule hint in life_simulator
- **gender / agreeableness / openness / sticker_preferences** → not used (already implicit in background/speaking_style or covered by other mechanisms)

## Approach: Structured Prompt Injection (Plan B)

Changes are confined to two files. No new data structures or agents required.

---

## Change 1 — `expression.py`: Add `## 你的特质` section

**Location:** Inside `_compose()`, immediately after the `## 你是谁` block.

**Content:**

```python
interests_str = "、".join(self.persona.get("interests", []))
mbti = self.persona.get("mbti", "")
mbti_desc = {
    "ENFP": "热情开放、充满好奇心、喜欢分享、容易被新事物吸引",
    # extend as needed
}.get(mbti, "")

trait_section = (
    f"## 你的特质\n"
    f"年龄：{self.persona.get('age', '')}岁\n"
    f"MBTI：{mbti}（{mbti_desc}）\n"
    f"兴趣：{interests_str}\n"
)
```

**Insertion point in prompt string:** The current prompt f-string in `_compose()` (line ~185) starts with:

```
你是{self.persona['name']}，一个真实的人在用手机聊天。

## 你是谁
{self.persona['background']}

## 此刻状态
```

Replace this block with:

```python
f"""你是{self.persona['name']}，一个真实的人在用手机聊天。

## 你是谁
{self.persona['background']}

{trait_section}
## 此刻状态
```

The variable `trait_section` is built just before the prompt f-string (same `_compose()` method scope) so it is available for interpolation.

**Rationale:** The expression model uses this section to colour reply tone, topic choices, and self-references. Age prevents anachronistic language. MBTI gives a one-line personality shorthand. Interests let the character organically reference her hobbies.

---

## Change 2 — `life_simulator.py`: Add interests to tick prompt

**Location:** Inside `tick()`, in the prompt f-string.

**Addition:** A new line after `她的职业：{self.persona.get('occupation', '')}` (line ~155 in `tick()`):

```python
interests_str = "、".join(self.persona.get("interests", []))
```

The resulting block in the prompt becomes:

```
她的职业：{self.persona.get('occupation', '')}
她的兴趣爱好：{interests_str}
她的性格：外向{...} 自律{...}
```

Note: `interests_str` is defined locally in `tick()`. A parallel definition also appears in `expression.py`'s `_compose()` (Change 1). The duplication is intentional — each method builds its own local string; no shared helper is needed.

**Rationale:** Without this, the LLM generating the next-15-minute activity has no knowledge of what she enjoys. It tends to produce generic actions. With interests, evenings and weekends become more varied and persona-consistent.

---

## Change 3 — `life_simulator.py`: Dynamic schedule hint in `_get_activity_hints()`

**Current behavior:** The method returns a hardcoded string from `_ACTIVITY_POOL_WEEKDAY` based on hour, using an early-return inside the loop.

**New behavior:** For weekdays, append a dynamic schedule annotation derived from `daily_patterns`. The loop is refactored from early-return to break+variable to enable the appending. The logic is equivalent; the structure change is intentional.

```python
def _get_activity_hints(self, hour: float, is_weekday: bool) -> str:
    pool = self._ACTIVITY_POOL_WEEKDAY if is_weekday else self._ACTIVITY_POOL_WEEKEND
    h = int(hour) % 24
    base_hint = "发呆、刷手机、失眠"
    for start, end, hints in pool:
        if start <= h < end:
            base_hint = hints
            break

    if not is_weekday:
        return base_hint

    # Append schedule context from config
    dp = self.persona.get("daily_patterns", {})
    ws = dp.get("work_start", [9, 10])
    we = dp.get("work_end", [18, 19])
    ln = dp.get("lunch", [11, 13])
    schedule = (
        f"（上班时间约{ws[0]}-{ws[1]}点，"
        f"午饭约{ln[0]}-{ln[1]}点，"
        f"下班约{we[0]}-{we[1]}点）"
    )
    return base_hint + "，" + schedule
```

The hardcoded `_ACTIVITY_POOL_WEEKDAY` is preserved as activity-type guidance; the schedule annotation adds time-boundary context on top.

---

## Affected Files

| File | Nature of change |
|------|-----------------|
| `expression.py` | Add `trait_section` variable and insert it into prompt f-string |
| `life_simulator.py` | Add interests line in tick prompt; update `_get_activity_hints()` |

## Out of Scope

- `emotion_engine.py` — MBTI/agreeableness/openness not wired numerically (per decision)
- `sticker_engine.py` — `sticker_preferences` not wired (per decision)
- `browser_agent.py` — interests-based scoring not in this iteration
- No new config fields, no schema changes
