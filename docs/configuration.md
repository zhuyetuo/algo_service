# 配置项说明

所有配置项均可通过环境变量或项目根目录的 `.env` 文件覆盖（变量名为大写下划线形式）。

复制模板后按实际环境修改：

```bash
cp .env.example .env
```

---

## MySQL

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DB_HOST` | `192.168.33.253` | 主机地址，同服务器部署填 `host.docker.internal` |
| `DB_PORT` | `30100` | 端口 |
| `DB_NAME` | `algo` | 数据库名 |
| `DB_USER` | `root` | 用户名 |
| `DB_PASSWORD` | `Hicc-pet-mysql-2026` | 密码 |

Schema 名称（通常无需修改）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `PG_SCHEMA_BEHAVIOR` | `pet_dog_behavior` | 行为事件表所在 schema |
| `PG_SCHEMA_ASSESSMENT` | `pet_dog_skin_assessment` | 日评估结果表所在 schema |
| `PG_SCHEMA_ENVIRONMENT` | `pet_dog_environment` | 环境数据表所在 schema |
| `PG_SCHEMA_BASELINE` | `pet_dog_scratch_baseline` | 个体基线表所在 schema |

---

## TDengine

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TD_HOST` | `tdengine` | 主机地址，同服务器部署填 `host.docker.internal` |
| `TD_PORT` | `6041` | REST API 端口 |
| `TD_USER` | `root` | 用户名 |
| `TD_PASSWORD` | `taosdata` | 密码 |
| `TD_DATABASE` | `pet_collar_raw` | 数据库名 |
| `TD_SUPERTABLE` | `imu_raw` | IMU 超级表名 |
| `TD_BATCH_SIZE` | `50000` | 每次单设备最大拉取行数 |

---

## 模型与推理

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MODEL_PATH` | `weights/behavior_lgbm.pkl` | LightGBM 模型文件路径 |
| `IMU_SAMPLE_RATE` | `50` | IMU 采样率（Hz） |
| `WINDOW_SECONDS` | `3` | 滑动窗口时长（秒） |
| `WINDOW_OVERLAP` | `0.5` | 滑动窗口重叠比例 |
| `FETCH_INTERVAL_MIN` | `15` | 推理调度间隔（分钟） |

---

## 调度

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `BASELINE_UPDATE_CRON` | `0 2 * * *` | 基线更新 cron 表达式（UTC） |
| `ASSESSMENT_CRON` | `0 3 * * *` | 日评估 cron 表达式（UTC） |

---

## 评估阈值

三个阶段的告警触发条件，根据设备基线成熟度自动切换。

| 配置项 | 默认值 | 适用阶段 | 说明 |
|--------|--------|---------|------|
| `PHASE1_THRESHOLD_Z` | `4.0` | Phase 1（第 3–13 天） | Z-score 单日阈值 |
| `PHASE1_THRESHOLD_CONSEC` | `5` | Phase 1 | 连续异常天数 |
| `PHASE1_THRESHOLD_AVGZ` | `5.0` | Phase 1 | 滑动窗口平均 Z |
| `PHASE2_THRESHOLD_Z` | `3.5` | Phase 2（第 14–29 天） | Z-score 单日阈值 |
| `PHASE2_THRESHOLD_CONSEC` | `4` | Phase 2 | 连续异常天数 |
| `PHASE2_THRESHOLD_AVGZ` | `4.0` | Phase 2 | 滑动窗口平均 Z |
| `PHASE3_THRESHOLD_Z` | `2.5` | Phase 3（第 30 天起） | Z-score 单日阈值 |
| `PHASE3_THRESHOLD_CONSEC` | `3` | Phase 3 | 连续异常天数 |
| `PHASE3_THRESHOLD_AVGZ` | `3.5` | Phase 3 | 滑动窗口平均 Z |

---

## 其他

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `BASELINE_STD_FLOOR` | `2.0` | 抓挠计数基线标准差下限（防除零） |
| `BASELINE_STD_FLOOR_WPEB` | `1.0` | W-PEB 基线标准差下限 |
| `MIN_WEAR_MINUTES` | `480` | 每天最少有效佩戴分钟数（低于此值标记为无效天） |
| `NIGHT_HOUR_START` | `22` | 夜间窗口起始小时（本地时间） |
| `NIGHT_HOUR_END` | `6` | 夜间窗口结束小时（本地时间） |
| `LOG_LEVEL` | `info` | 日志级别（debug / info / warning / error） |
