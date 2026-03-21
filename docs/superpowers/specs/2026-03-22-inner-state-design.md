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
    mood_tint: float    # -1~1，本次独白的情绪色调（仅用于 prompt 注入，不作为衰减地板）
    mood_reason: str    # "昨晚睡得很好" / "工作上有件事没处理完"
    generated_at: str   # ISO时间戳
```

> **注意**：`mood_tint` 是独白的主观色调，仅供 `expression.py` prompt 使用，与 `EmotionEngine` 的 `daily_baseline`（衰减地板）是两个不同概念，不可互换。

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
  "mood_tint": 0.1,
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

**起床时触发：**

`LifeSimulator.tick()` 在 `is_sleeping` 从 `True` 变为 `False` 时（第 69-75 行）新增 emit `life.woke_up` 事件（无 payload）。`InnerStateManager` 订阅此事件并调用 `on_wake_up()`。

```
今日 mood_baseline =
    yesterday_final_valence × 0.4           # 延续感，但不完全继承
  + 随机扰动 uniform(-0.15, +0.15)          # 无由来的情绪波动
  + 星期权重（周一 -0.05，周五 +0.05，周末 +0.03）
  + (life_sim.physical.sleep_debt > 30 ? -0.1 : 0)  # 睡眠债影响
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
- 将 `pending_thoughts` 中 `target_user_id > 0` 的条目转成 proactive 触发器，类型为 `"follow_up"`，urgency 直接继承
- 与现有的 `missing` / `mood_share` / `life_event` 触发器并存，不替换
- 触发器被选中后返回给 `Orchestrator`，**由 `Orchestrator` 在 `await callback(messages)` 成功执行后**（即消息实际发出后）emit `proactive.follow_up_fired` 事件，payload：`{"thought_content": trigger["content"]}`
- `InnerStateManager` 订阅 `proactive.follow_up_fired`，从 `pending_thoughts` 中移除 `content` 匹配的条目并持久化
- 若 callback 不存在或失败，不 emit 事件，thought 保留直至自然过期或下次独白生成时由 LLM 标记为 resolved

### memory_hub.py
- `consolidate()` 对**每个** `chat_id` 处理完成后，各自 emit 一次 `memory.consolidated` 事件，payload：`{"chat_id": cid, "convo_text": convo_text[:1500]}`
- `InnerStateManager` 订阅此事件，将最新收到的 `convo_text` 追加到 `_recent_convos` 缓冲（最多保留 3 条），在下次独白生成时作为"最近对话"输入

### emotion_engine.py
- 新增字段 `daily_baseline: float = 0.0`（构造函数初始化为 0，确保首日行为与现在一致）
- 新增 `set_daily_baseline(value: float)` 方法，仅在 `life.woke_up` 时由 `InnerStateManager` 调用一次，**不在每次独白更新时调用**
- `passive_decay()` 中 `valence` 改为衰减到基线：`valence = daily_baseline + (valence - daily_baseline) × 0.98`
- `arousal` 和 `irritability` 继续按原逻辑衰减到零（`×0.95` / `×0.97`），不受 `daily_baseline` 影响

### life_simulator.py
- 在 `tick()` 的起床判断块（第 65-76 行）中，`self.is_sleeping = False` 之后新增：`await self.bus.emit("life.woke_up", {})`

### orchestrator.py
- 实例化 `InnerStateManager`，注入 emotion、life_sim、memory 引用
- 新增后台 loop：`_inner_state_loop()`，间隔从 config 读取
- 在主动消息发送成功后（`await callback(messages)` 完成后）emit `proactive.follow_up_fired` 事件（若触发类型为 `"follow_up"`）
- **不**直接调用 `inner_state.on_wake_up()`；起床事件由 `InnerStateManager` 通过订阅 `life.woke_up` 自行响应

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
| 首次启动，无历史数据 | `yesterday_final_valence = 0`，`EmotionEngine.daily_baseline = 0`，随机扰动正常工作 |
| LLM 调用失败 | 保留上一次 monologue，不更新 pending_thoughts |
| `pending_thought` 过期 | 在每次生成独白时清理 expires_at < now 的条目 |
| 睡眠中触发 loop | 跳过生成，不 emit 事件 |
| 入睡时 | 订阅 `life.sleeping`，快照当前 `emotion.state.valence` 到 `yesterday_final_valence` 并持久化 |
| target_user_id 不在 proactive_callbacks | 跳过该 pending_thought，不触发 |

---

## 不在此次范围内

- 独白内容不对外暴露（不加新 `/status` 字段）
- 不修改群聊行为逻辑
- 不改变记忆检索的隐私过滤规则
- `pending_thoughts` 不跨用户共享（每条绑定单一 target）
