# 写进 MySQL 的东西：换 5 类模型之后哪些变了

这个服务**没有任何人调它**——定时从 TDengine 取项圈上报的原始 IMU，
推理完写进 MySQL，后端（Java）定期去表里查有没有新结果。
整条链上只通过数据库交接，所以**库里的取值范围就是接口契约**。

换成 5 类模型（活动/睡觉/抓挠/未佩戴/甩身体）之后，有一处变化必须
提前跟后端说，还有一处是 bug、已经修了。

---

## ① `pet_dog_behavior.d_{device_id}.behavior` 多了一个取值：**4**

| 编码 | behavior_label | 什么时候出现 |
|---|---|---|
| 0 | 未知 | 模型不确定（`CONFIDENCE_THRESHOLD` > 0 时才有，默认 0 = 不产生） |
| 1 | 活动 | 一直有。**甩身体也并进这一类** |
| 2 | 睡觉 | 一直有 |
| 3 | 抓挠 | 一直有 |
| **4** | **未佩戴** | **5 类模型才有，新增** |

**这是唯一一处取值范围的变化。** 表结构（`behavior SMALLINT`、
`behavior_label VARCHAR(8)`）没动，没有加列也没有改类型。

如果后端那边是 `switch (behavior)` 或者枚举 0..3，碰到 4 会走进 default
分支——表现可能是"多了一类不认识的数据"，也可能是静默丢掉。
**这个要提前说一声**，不是库的格式坏了。

### 为什么不把它并进"未知"

以前没有这个类别时，摘下项圈那段会被模型判成**睡觉**（三个类别里必须挑一个），
于是 `sleep_min` 虚高，而看数据完全看不出来——跟真的在睡长得一模一样。
并进"未知"的话，"模型不确定"和"项圈没戴"又分不开了。

### 想临时退回 3 类

`MODEL_PATH` 指回 `weights/ml_rf.pkl`（老的 3 类模型还在）。
类别表从模型自己的 `.json` 读，不写死，所以退回去之后就不再产生 4。

---

## ② 已修：「未佩戴」的时间被算成了「佩戴」

`pet_dog_skin_assessment` 的 `wear_minutes` 和
`pet_dog_daily_summary` 的一串列都受影响。

原来算佩戴时长是"把当天所有行为事件的时长加起来"——3 类的时候没问题，
每个事件都意味着项圈戴在身上。有了 behavior=4 之后照旧全加，
**方向正好反了**：没戴的时间被算成戴着。

一处错、四处连带，而且一个报错都没有：

| 受影响 | 怎么错 |
|---|---|
| `wear_minutes` | 虚高。该标无效天（`data_quality=1`）的没标 |
| `sleep_ratio` / `active_ratio` | 分母偏大 → 比值偏低 → `sleep_status`/`move_status` 判错 |
| `off_min` | `1440 - wear - loose`，wear 虚高所以 off 偏小 |
| `sleep_min + move_min + scratch_min` | **不再约等于 `wear_min`** |

最后那条大概就是后端那边"看着格式有点不对"的来源：以前这三列加起来
约等于 `wear_min`，现在中间凭空少了一块（未佩戴那段既在 wear 里、
又不属于任何一列）。

**修法**：算佩戴时长时排掉 `behavior = 4`。3 类模型不产生 4，所以
这个条件对历史数据是空操作，不会改写老的天。
SQL 在 `modules/assessment/queries.py:wear_ms_sql`，
测试在 `tests/unit/test_wear_minutes.py`（真建表真执行，不是比字符串）。

修完之后：

```
sleep_min + move_min + scratch_min  ≈  wear_min        （等式恢复）
off_min = 1440 - wear_min - loose_min                  （现在包含未佩戴那段）
```

### ⚠ 已经写进去的那些天不会自动改

修的是往后算的逻辑。中间跑过 5 类模型的那几天，库里的
`wear_minutes` / `sleep_ratio` / `off_min` / 状态灯仍然是旧算法的结果。
要纠正就重跑那几天的评估（`backfill/`），**否则那几天的数会跟前后不一致**。

---

## 没有变的东西

- 表结构：没加列、没删列、没改类型
- `pet_dog_behavior`：`ts_start` 唯一键、`INSERT IGNORE` 的幂等行为
- `duration_sec` `DECIMAL(10,2)`、`confidence` `DECIMAL(5,3)` 的精度
- 时区处理、`local_start` / `local_end` 的格式
- 抓挠相关的所有列（`scratch_*`、`night_scratch_count`）——抓挠还是 3
- 几何（16Hz / 1s 窗 / 0.5s 步）由模型 `.json` 决定，换模型不用改配置
