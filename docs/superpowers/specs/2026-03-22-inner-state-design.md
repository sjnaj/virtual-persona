# Inner State Manager — 设计文档

**日期：** 2026-03-22
**目标：** 让虚拟人格表现出跨对话连续性和情绪不可预测性

---

## 背景与问题

当前系统存在两个让人格显得"机器味"的缺陷：

1. **跨对话没有连续性**：她不会主动跟进之前提过的事，每次对话像是全新开始。`relationship.py` 的 `unresolved` 字段存在但从未被填充或消费。
2. **情绪过于稳定可预测**：`emotion_engine` 的衰减是纯数学的，缺少无由来的坏心情、隔天情绪延续、以及"今天就是心情不好"这类真实的情绪底色。

---

## 解决方案：内心独白层（`inner_state.py`）

新建一个 `InnerStateManager`，负责：
- 每隔 2-4 小时生成一段内心独白快照
- 将情绪状态和未处理的想法持久化到磁盘
- 起床时用昨天的状态初始化今天的情绪基线
- 向 `proactive_engine` 提供"想跟进某件事"的触发器
- 向 `expression.py` 提供当前心理底色

---

## 数据结构

### `InnerMonologue`
每次生成独白时产生的快照：

```python
@dataclass
class InnerMonologue:
    text: str           # "最近总有点心不在焉，也不知道为什么..."
    mood_baseline: float   # -1~1，当前情绪底色
    mood_reason: str    # "昨晚睡得很好" / "工作上有件事没处理完"
    generated_at: str   # ISO时间戳
```

### `PendingThought`
她脑子里"挂着"的事，是跨对话连续性的核心载体：

```python
@dataclass
class PendingThought:
    content: str        # "想问小明那个项目后来怎样了"
    target_user_id: int # 0=不针对某人，>0=想找某人说
    urgency: float      # 0~1
    source: str         # "conversation" / "life_event" / "browsing"
    created_at: str
    expires_at: str     # 过期自动遗忘（默认 48 小时）
```

### 持久化格式（`./data/inner_state.json`）
```json
{
  "current_monologue": { ... },
  "yesterday_final_valence": 0.2,
  "yesterday_final_arousal": -0.1,
  "pending_thoughts": [ ... ]
}
```

---

## 独白生成流程

**触发时机：** 后台 loop，每 `inner_state.interval_hours`（默认 2~3 小时随机）触发一次，且仅在非睡眠状态下运行。

**输入上下文：**
- 当前情绪状态（valence, arousal, irritability）
- 当前生活状态（正在做什么、体力）
- 最近生活事件（过去 3 小时的 notable events）
- 最近对话摘要（由 `memory.consolidated` 事件提供）
- 当前 `pending_thoughts` 列表

**LLM 调用（utility tier）输出 JSON：**
```json
{
  "monologue": "她的一句内心独白",
  "mood_baseline": 0.1,
  "mood_reason": "今天有点累",
  "new_pending_thoughts": [
    {
      "content": "想问小明那个项目后来怎样了",
      "target_user_id": 12345,
      "urgency": 0.6,
      "source": "conversation"
    }
  ],
  "resolved_thoughts": ["content of thought that was resolved"]
}
```

生成完成后 emit `inner_state.updated` 事件。

---

## 情绪跨日连续性

**起床时触发（订阅 `life.sleeping` 解除后的第一次 tick）：**

```
今日 mood_baseline =
    yesterday_final_valence × 0.4    # 延续感，但不完全继承
  + 随机扰动 uniform(-0.15, +0.15)   # 无由来的情绪波动
  + 星期权重（周一 -0.05，周五 +0.05，周末 +0.03）
  + sleep_quality 修正（sleep_debt > 30 时 -0.1）
```

`EmotionEngine` 新增 `set_daily_baseline(value: float)` 接口，由 `InnerStateManager` 在起床事件时调用，设置当天情绪的"地板"——其他情绪事件在这个基础上叠加。

---

## 与现有模块的集成

### expression.py（最小改动）
在 prompt 的"此刻状态"区块后追加：

```
## 你心里在转的
{inner_state.current_monologue.text}
```

不改 prompt 结构，仅新增一段。LLM 自然地将心理底色融入回复语气。

### proactive_engine.py
- 订阅 `inner_state.updated` 事件
- 将 `pending_thoughts` 中 `target_user_id > 0` 的条目转成 proactive 触发器
- 触发类型标记为 `"follow_up"`，urgency 直接继承
- 与现有的 `missing` / `mood_share` / `life_event` 触发器并存，不替换

### memory_hub.py
- `consolidate()` 完成后 emit `memory.consolidated` 事件，携带对话原文（截断至 1500 字符）
- `InnerStateManager` 订阅此事件，在下次独白生成时作为"最近对话"输入

### emotion_engine.py
- 新增 `set_daily_baseline(value: float)` 方法
- `passive_decay()` 衰减到基线而非零：`valence → baseline × (1 - decay_factor) + valence × decay_factor`

### orchestrator.py
- 实例化 `InnerStateManager`，注入 emotion、life_sim、memory 引用
- 新增后台 loop：`_inner_state_loop()`，间隔从 config 读取
- 在 `_life_loop()` 的起床判断处调用 `inner_state.on_wake_up()`

### config.yaml
```yaml
system:
  inner_state_interval_hours: [2, 3]   # 随机区间，单位小时
  pending_thought_ttl_hours: 48        # pending_thoughts 默认过期时间
```

---

## 边界条件

| 情况 | 处理方式 |
|------|---------|
| 首次启动，无历史数据 | `yesterday_final_valence = 0`，随机扰动正常工作 |
| LLM 调用失败 | 保留上一次 monologue，不更新 pending_thoughts |
| `pending_thought` 过期 | 在每次生成独白时清理 expires_at < now 的条目 |
| 睡眠中触发 loop | 跳过生成，不 emit 事件 |
| target_user_id 不在 proactive_callbacks | 跳过该 pending_thought，不触发 |

---

## 不在此次范围内

- 独白内容不对外暴露（不加新 `/status` 字段）
- 不修改群聊行为逻辑
- 不改变记忆检索的隐私过滤规则
- `pending_thoughts` 不跨用户共享（每条绑定单一 target）
