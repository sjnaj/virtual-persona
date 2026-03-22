# LifeSimulator 重构设计文档

**日期**: 2026-03-22
**文件**: `life_simulator.py`
**状态**: 待实现

---

## 问题背景

当前 `LifeSimulator` 存在两个已知 Bug 和三个待增强方向：

### Bug
1. **重启丢失状态**：`__init__` 每次都用默认值初始化（`is_sleeping=True`、`energy=80` 等），无持久化机制。
2. **行为自强化循环**：LLM 生成的 `current_action`（如"刷小红书"）被放入下一轮 prompt 的 `recent_actions`，模型看到全是相同行为后倾向于继续生成相同行为，形成正反馈锁死。

### 待增强
3. 行为多样性不足，缺乏时段感。
4. 天气/季节信息硬编码，与真实日期脱节。
5. 没有猫咪年糕的互动，缺少角色特色。

---

## 设计方案（方案 A：最小改动）

所有改动限定在 `life_simulator.py` 内，不修改其他模块的接口。

---

## 一、状态持久化

### 存储位置
`./data/life_state.json`

### 写入时机
每次有效 `tick()`（产生了新的 action 的 tick）结束后，异步写入（`asyncio.create_task`），不阻塞主流程。

### 保存字段
```json
{
  "physical": { "energy": 72, "hunger": 35, "comfort": 68, "sleep_debt": 5 },
  "location": "家",
  "current_action": "刷剧",
  "current_detail": "在看脱口秀第三集",
  "is_sleeping": false,
  "woke_up_today": true,
  "daily_weather": { "condition": "小雨", "temp": 18, "date": "2026-03-22" },
  "yearago": { "location": "沙发", "mood": "赖床" },
  "event_log": [ ...最近25条 LifeEvent 序列化... ],
  "saved_at": "2026-03-22T21:30:00"
}
```

### 恢复策略
启动时调用 `_load_state()`：
- 文件存在 **且** `saved_at` 在过去 24 小时内 → 恢复全部字段
- 文件不存在或超过 24h → 使用默认值（长时间停机，状态应重置）
- JSON 解析失败 → 忽略，使用默认值，记录 warning

### 新增方法
- `_load_state()` — `__init__` 末尾调用
- `_save_state()` — 每次有效 tick 末尾调用（`asyncio.create_task` 包装）
- `async def _do_save()` — 实际写文件（`Path.write_text` + `json.dumps`）

---

## 二、行为多样性（提示词压制）

### 高频行为黑名单
在构建 prompt 前，统计 `event_log` 最近 8 条中出现 ≥2 次的 action，将其列入黑名单并注入 prompt：
```
最近这些行为已经出现太多次了，这轮请避免选择：["刷小红书", "刷手机"]
```
若无重复行为，则不添加此段。

### 时段活动池
按当前小时分段，向 prompt 提供候选行为范围（软约束，AI 可自由选择其他合理活动）：

| 时段 | 工作日候选行为 |
|------|--------------|
| 7–9  | 起床洗漱、吃早饭、通勤 |
| 9–12 | 开会、做设计、摸鱼、喝咖啡 |
| 12–14 | 吃午饭、午休、散步 |
| 14–18 | 做设计、改稿、摸鱼、下午茶 |
| 18–20 | 通勤回家、买晚饭、做饭 |
| 20–23 | 看剧、刷手机、练字、洗澡、陪年糕玩、发呆 |

| 时段 | 周末候选行为 |
|------|------------|
| 7–10 | 睡懒觉、赖床、慢慢起床 |
| 10–12 | 吃早午饭、逛超市、出门买奶茶 |
| 12–18 | 出门逛街、做美甲、看展、朋友聚餐、午睡 |
| 18–23 | 买晚饭、看综艺、刷视频、洗澡、和朋友聊天 |

候选池通过 `_get_activity_hints(hour, is_weekday)` 方法返回。

### Prompt 注入位置
在现有 prompt 末尾、JSON 格式要求之前，插入两段：
```
最近活动有点单调，这轮请避免：{blacklist}（如果列表非空）
这个时间段她比较可能做的事情有：{hints}，也可以有其他合理安排。
```

---

## 三、天气 / 季节感知

### 城市来源
1. `persona.city`（`config.yaml` 新增可选字段）
2. 无此字段则默认 `北京`

### 获取时机
每次 tick 时检查 `daily_weather["date"]` 是否等于今天（`datetime.now().date().isoformat()`）：
- 不等于 → 调用 `_fetch_weather()` 更新
- 等于 → 直接复用缓存（同天不重复请求）

### API
```
GET https://wttr.in/{city}?format=j1
```
取 `current_condition[0]` 中的 `temp_C` 和 `weatherDesc[0].value`，映射为简体中文描述（"Sunny" → "晴"，"Rain" → "雨"，等）。

### 失败兜底
网络请求失败或解析异常时，用 `_season_default_weather()` 生成季节默认天气（按月份区间，见下表）并记录 warning，不抛出异常。

| 月份 | 季节 | 默认天气示例 |
|------|------|------------|
| 3–5  | 春   | 多云 18°C |
| 6–8  | 夏   | 晴热 32°C |
| 9–11 | 秋   | 晴 22°C |
| 12–2 | 冬   | 阴 6°C |

### Prompt 注入
将原 prompt 中硬编码的 `天气：晴 22°C（简化）` 替换为：
```
今天天气：{condition} {temp}°C
```

### 新增方法
- `async def _fetch_weather(city: str) -> dict` — 请求 wttr.in，返回 `{condition, temp, date}`
- `_season_default_weather() -> dict` — 按当前月份返回季节默认值

---

## 四、年糕（猫咪）互动

### 数据结构
新增 `YearGaoState` dataclass（定义在 `life_simulator.py` 顶部）：
```python
@dataclass
class YearGaoState:
    location: str = "沙发"   # 沙发/床上/窗台/猫窝/厨房/消失了
    mood: str = "发呆"        # 赖床/玩耍/讨食/发呆/黏人/暴走
```

### 状态更新
`_update_yearago(hour)` 方法，每次 tick 时调用，纯规则 + 随机，无 LLM 调用：

| 时段 | 高概率 mood（权重最高） |
|------|----------------------|
| 6–8  | 讨食（60%）、黏人（30%） |
| 9–17 | 赖床（50%）、发呆（40%） |
| 17–19 | 讨食（60%）、玩耍（30%） |
| 20–23 | 玩耍（40%）、暴走（30%）、黏人（20%） |
| 0–6  | 发呆（50%）、消失了（40%） |

location 随 mood 联动（讨食→厨房，赖床/发呆→沙发/床上/猫窝，暴走→消失了）。每次 tick 有 30% 概率切换状态（猫咪不会每15分钟都变），避免抖动。

### Prompt 注入
在现有状态说明区块中追加：
```
你的猫年糕现在：在{location}{mood}
```

### 对 notable 的影响
当年糕处于"讨食"或"暴走"时，在 prompt 中追加提示：
```
（年糕现在比较闹腾，相关的事情比平时更值得分享）
```

### 持久化
`YearGaoState` 序列化进 `life_state.json` 的 `yearago` 字段，随其他状态一起恢复。

---

## 五、config.yaml 变更

新增一个可选字段：
```yaml
persona:
  city: "上海"   # 用于获取真实天气，不填则默认北京
```

---

## 六、依赖变更

- 新增 `httpx`（已在 `browser_agent.py` 中使用，无需额外安装）
- 无新增外部依赖

---

## 七、接口变更

`get_status()` 返回值扩展，新增字段：
```python
{
    ...,
    "weather": { "condition": "小雨", "temp": 18 },
    "yearago": { "location": "沙发", "mood": "发呆" },
}
```
`Orchestrator` 的 `get_full_status()` 通过 `life_sim.get_status()` 自动获得这些字段，无需修改。

---

## 八、测试要点

- 重启后状态恢复（`saved_at` 在 24h 内）
- 重启后超过 24h 重置状态
- 连续相同 action 触发黑名单，下一 tick prompt 包含禁止项
- 天气 API 失败时使用季节默认值，不抛异常
- 年糕状态按时段更新，持久化后恢复正确
