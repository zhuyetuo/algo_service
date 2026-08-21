# 离线工具

设备端链路还没打通时用的两个工具：

| 脚本 | 作用 |
|------|------|
| `diagnose_signal.py` | 核对设备上报的 IMU 单位是否与模型训练单位一致 |
| `run_backfill.py` | 把 TDengine 历史 IMU 数据跑一遍推理，结果写进 MySQL 行为表 |
| `import_infer.py` | 把 **imu_train 那边跑出来的推理结果**导入 MySQL 行为表 |

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


---

# 三、导入 imu_train 推理结果 `import_infer.py`

设备链路没打通、但 imu_train 那边已经拿录制数据跑出识别结果时用这个。
和 `run_backfill.py` 的区别：回补是在**本服务里做推理**，这个是**直接吃别处算好的结果**。

## 输入格式

### 首选：imu_train 原生产物 `*_infer.json`（不用额外做转换）

`run_review_bins_all_days.sh` 跑完，`RESULT_ROOT/{day}/` 下每个 CSV 都有一份
`{stem}_infer.json`，`infer_csv_scratch.py` 写的，结构是：

```json
{
  "csv_basename": "xxx_IMU1.csv",
  "n_windows": 3600,
  "windows": [
    {"ts": "2026-08-19 12:36:20.000", "label": "睡觉", "conf": 0.98,
     "probs": {"抓挠": 0.01, "活动": 0.01, "睡觉": 0.98}}
  ],
  "scratch_segments": [...]
}
```

工具只用 `windows` 里的 `ts` / `label` / `conf`，其余字段忽略。
**把整个 `RESULT_ROOT/{day}/` 目录拷过来即可，不需要转成 CSV。**

### 备选：CSV

需要自己出 CSV 时，两种表头都支持，按表头自动识别：

```csv
# 窗口级（推荐，语义与线上推理一致，由本工具负责合并成事件）
device_sn,ts,label,conf
EA:CB:3E:CF:00:11,2026-08-19 12:36:20.000,睡觉,0.9812
```

```csv
# 事件级（已经合并好的片段，原样导入）
device_sn,start_ts,end_ts,label,conf
EA:CB:3E:CF:00:11,2026-08-19 12:36:20.000,2026-08-19 12:41:20.000,睡觉,0.95
```

| 列 | 必填 | 说明 |
|----|------|------|
| `device_sn` | 否 | 填了就按它匹配设备；不填则按**文件名**匹配映射表 |
| `ts` / `start_ts` / `end_ts` | 是 | `YYYY-MM-DD HH:MM:SS[.mmm]`、ISO8601（可带 `+08:00`）或 epoch 毫秒 |
| `label` | 是 | 必须是 `活动` / `睡觉` / `抓挠` 三者之一，其它值整行跳过 |
| `conf` | 是 | 0~1 小数 |

> 时间戳不带时区偏移时按 `--tz` 解释（默认 `Asia/Shanghai`）。
> 想彻底避免歧义，直接写成 `2026-08-19T12:36:20.000+08:00`。

## 设备映射表

imu_train 那边的标识是 `IMU1` / `task496_imu1` 这类录制分组键，不是 `device_id`，
所以要给一张映射表（`match` 对文件名做子串匹配，大小写不敏感，多条命中取最长的）：

```csv
match,device_sn,device_id,bind_id,timezone
IMU1,EA:CB:3E:CF:00:11,,,Asia/Shanghai
IMU2,EA:CB:3E:CF:00:12,,,Asia/Shanghai
```

`device_sn` 和 `device_id` **至少填一个**：只给 `device_sn` 时会去业务库反查
`device_id` / `bind_id` / 用户时区；业务库查不到就必须直接填 `device_id`。
模板见 `backfill/device_map.csv.example`。

## 用法

```bash
# 1. 先 dry-run，看看每台设备解析出多少事件、行为分布合不合理
docker exec algo_service python backfill/import_infer.py \
    --input infer_result_majority/2026_8_19 \
    --device-map backfill/device_map.csv --dry-run

# 2. 确认无误后写库
docker exec algo_service python backfill/import_infer.py \
    --input infer_result_majority/2026_8_19 \
    --device-map backfill/device_map.csv

# 3. 另一个模型变体导到带后缀的表，方便并排比较，不污染正式表
docker exec algo_service python backfill/import_infer.py \
    --input infer_result_majority_syn/2026_8_19 \
    --device-map backfill/device_map.csv --table-suffix _syn
```

| 参数 | 说明 |
|------|------|
| `--input` | 结果目录（递归找 `*_infer.json` / `*.csv`）或单个文件 |
| `--device-map` | 设备映射表 CSV |
| `--tz` | 时间戳不带时区时按此时区解释（默认 `Asia/Shanghai`） |
| `--window-sec` | 推理窗口长度，用于算事件结束时间（默认 `2.0`） |
| `--max-gap-sec` | 相邻窗口间隔超过该值就断开（默认按 2.5 倍步长自动推断） |
| `--table-suffix` | 目标表后缀，如 `_syn` → 写入 `d_70_syn` |
| `--dry-run` | 只解析统计，不写库 |

## 两个注意点

**录制中断会自动断开事件。** 相邻窗口间隔超过 `--max-gap-sec`（默认 2.5 倍步长）
就不合并——中间隔了半小时没数据的两段"睡觉"，不该被算成一段连续 30 分钟的睡眠。

**两个模型变体不能都往正式表里导。** 行为表 `ts_start` 有唯一键，
`majority` 和 `majority_syn` 时间戳完全一样，后导的会被 `INSERT IGNORE` 全部丢掉，
看起来"导入成功"但一条没进。要并排比较就用 `--table-suffix`。
