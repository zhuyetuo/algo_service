# 离线工具

设备端链路还没打通时用的两个工具：

| 脚本 | 作用 |
|------|------|
| `diagnose_signal.py` | 核对设备上报的 IMU 单位是否与模型训练单位一致 |
| `run_backfill.py` | 把历史 IMU 数据跑一遍推理，结果写进 MySQL 行为表 |

**先跑 `diagnose_signal.py` 再跑回补** —— 单位不对的话回补出来的结果是错的，
还得删库重来。

---

# 一、信号量级诊断 `diagnose_signal.py`

imu_train 训练时**不做量纲统一**，训练 CSV 是什么单位模型就学什么量级。而特征里
mean / std / rms / range / SMA / 模长 / jerk 全是**有量纲的绝对量**，
单位差一个量级这些特征就整体平移出训练分布。

> 频谱熵、能量占比、相关系数是无量纲的不受影响，所以单位错了通常不是"全错"，
> 而是**置信度莫名偏低、某几类死活预测不出来**，比全错更难排查。

```bash
# 逐台设备看量级
docker exec algo_service python backfill/diagnose_signal.py

# 只看某台，或指定某天
docker exec algo_service python backfill/diagnose_signal.py --device-sn EA:CB:3E:CF:00:11
docker exec algo_service python backfill/diagnose_signal.py --device-sn EA:CB:3E:CF:00:11 --date 2026-08-19
```

判断依据是**静止时 `|acc|` 必然等于重力常数**：中位数 ≈9.8 就是 m/s²，≈1.0 就是 g，
工具会直接给结论。角速度没有这种固定锚点，工具打印量级由你判断——
犬只日常活动 deg/s 是几十~几百，换成 rad/s 则是零点几~几。

确认后在 `docker-compose.yml` 或 `.env` 里设置：

```bash
IMU_DEVICE_ACC_UNIT=ms2     # ms2 | g
IMU_DEVICE_GYRO_UNIT=rads   # dps | rads
# docker compose down && docker compose up -d 生效
```

启动日志的「量纲统一」一行会打印最终生效的换算系数，可核对：

```
量纲统一 : 加速度 m/s²→m/s² (×1)   角速度 rad/s→deg/s (×57.2958)
```

新模型建议在 `ml_rf.json` 里写上 `acc_unit` / `gyro_unit`，服务会优先采用，
不用再靠配置猜。

---

# 二、离线回补 `run_backfill.py`

把 **TDengine 里已有的历史 IMU 数据**跑一遍推理，
把行为识别结果写进 MySQL 行为表，先把"数据 → 算法 → 数据库"这条链路验证起来。

跟线上调度器的区别：

| | 线上推理周期（`scheduler/jobs.py`） | 离线回补（本工具） |
|---|---|---|
| 触发方式 | 每 15 秒自动跑 | 手动执行 |
| 数据范围 | 增量（`device_sync_state.last_processed_ts` 之后） | 指定日期 / 日期区间 |
| 断点 | 会推进断点 | **不碰断点**，不影响线上增量流程 |
| 重复执行 | — | 安全，行为表 `ts_start` 唯一键会 `INSERT IGNORE` 去重 |

---

## 用法

在容器内执行（依赖已装好，且能连到 MySQL 和 TDengine）：

```bash
# 1. 先看看 TDengine 里有哪些设备、数据覆盖哪段时间
docker exec algo_service python backfill/run_backfill.py --list-devices

# 2. 试跑：只推理、打印结果分布，不写库
docker exec algo_service python backfill/run_backfill.py --date 2026-08-19 --dry-run

# 3. 正式回补 8月19日 这一天的全部设备
docker exec algo_service python backfill/run_backfill.py --date 2026-08-19

# 4. 回补一段区间，并顺带跑每天的皮肤评估
docker exec algo_service python backfill/run_backfill.py \
    --start 2026-08-19 --end 2026-08-21 --assess

# 5. 只回补指定设备
docker exec algo_service python backfill/run_backfill.py --date 2026-08-19 --devices 70,72
```

### 参数

| 参数 | 说明 |
|------|------|
| `--date` | 回补单个日期（`YYYY-MM-DD`） |
| `--start` / `--end` | 回补日期区间，`--end` 含当天；不填 `--end` 等价于只跑 `--start` 那天 |
| `--devices` | 只处理指定 `device_id`，逗号分隔，如 `70,72` |
| `--assess` | 写完行为事件后再跑一遍当天皮肤评估（写评估表） |
| `--dry-run` | 只推理并打印行为分布，不写数据库 |
| `--list-devices` | 列出 TDengine 中的设备及数据时间范围后退出 |

> 日期按**各设备用户自己的时区**解释（取自 `hiccpet_petos.user.timezone`）。
> 同一个 `--date 2026-08-19`，北京时区的设备和 UTC 时区的设备拉取的 UTC 区间不同，
> 与线上推理周期的分日口径保持一致。

---

## 输出

逐设备逐天打印进度，结束后给一张汇总表：

```
==============================================================================
回补汇总    耗时 40.6s
==============================================================================
    设备ID  device_sn                  采样点    天数     事件数  行为分布
  --------------------------------------------------------------------------
      70  EA:CB:3E:CF:00:11        90000     1       2  活动=1  睡觉=1
  --------------------------------------------------------------------------
      合计                           90000             2
==============================================================================
```

写入目标表：`pet_dog_behavior.d_{device_id}`（表不存在会自动建）。
加 `--assess` 时另外写 `pet_dog_skin_assessment.d_{device_id}`。

查回补结果：

```bash
docker exec local-mysql8 mysql -h 192.168.33.253 -P 30100 -u root -pHicc-pet-mysql-2026 \
  -e "SELECT bind_id, behavior, behavior_label, duration_sec, local_start, local_end
      FROM pet_dog_behavior.d_70 ORDER BY ts_start LIMIT 20;" 2>/dev/null
```

---

## 常见问题

**「区间内无 IMU 数据」**
先用 `--list-devices` 确认该设备在 TDengine 里的数据时间范围，再对照 `--date` 是否落在范围内。
注意 `--list-devices` 打印的是 **UTC** 时间，而 `--date` 按用户本地时区解释。

**「没有匹配的设备」**
工具优先从 `hiccpet_petos.device_bind_history`（`bind_status=1`）读设备清单，
业务库不可用时退回 `device_sync_state`。两处都为空时就没有设备可跑——
先确认设备绑定关系已经写进业务库。

**重复跑会不会写重？**
不会。行为表对 `ts_start` 建了唯一键，写入用 `INSERT IGNORE`，重复回补同一天是幂等的。
但如果中途换了模型或改了窗口参数，旧结果不会被覆盖——需要先手动删掉那段时间的记录再回补。
