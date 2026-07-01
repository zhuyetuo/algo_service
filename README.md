# algo_service

智能宠物项圈算法服务。基于 6 轴 IMU 数据对犬猫行为进行分类，重点检测抓挠行为，并结合个体基线动态评估皮肤健康风险。

---

## 快速启动

**前置条件**：MySQL 和 TDengine 均使用远端服务器（192.168.33.253）。

```bash
# 1. 拉取代码
git clone <repo> && cd algo_service

# 2. 配置（按实际环境修改 .env）
cp .env.example .env

# 3. 训练模型（首次运行约 20-25 秒，生成 weights/behavior_lgbm.pkl）
python train/train.py

# 4. 启动服务
docker compose up -d --build

# 5. 确认两个数据库连接正常
curl http://localhost:8000/health
# 期望返回：{"status":"ok","mysql":"ok","tdengine":"ok"}
```

```bash
# 查看最新日志（实时）
docker logs algo_service --tail 50 -f

# 查看最新 50 条日志（不跟随）
docker logs algo_service --tail 50

# 开启逐窗口推理详情（每秒一行，含片上时间和置信度）
VERBOSE_INFERENCE=true docker compose up -d
docker logs algo_service -f | grep "片上"

# 重启服务（拉取最新代码后重建）
docker compose down && git pull && docker compose up -d --build

# 停止服务
docker compose down
```

---

## 数据基础设施

| 系统 | 地址 | 用途 |
|------|------|------|
| MySQL | 192.168.33.253:3306 | algo 服务自身数据库（行为事件、评估结果、基线、同步断点） |
| MySQL | 192.168.33.253:3306 | 业务库 `hiccpet_petos`（设备绑定、用户时区） |
| TDengine | 192.168.33.253:6041 | 时序原始数据（IMU、环境、体温） |

**TDengine 超级表结构：**

| 超级表 | 字段 | 说明 |
|--------|------|------|
| `imu_data` | `accel_x/y/z`, `gyro_x/y/z`, TAG:`device_sn` | 6 轴 IMU，50Hz |
| `env_data` | `temperature`, `humidity`, `body_temp`, TAG:`device_sn` | 环境温湿度 + 体温 |
| `battery_data` | `battery_level`, `charging`, TAG:`device_sn` | 电量与充电状态 |

**查询 TDengine 数据量：**

```bash
# 查看设备列表
curl -s -u root:taosdata \
  -d "SELECT DISTINCT device_sn FROM hiccpet_device.imu_data" \
  http://192.168.33.253:6041/rest/sql | python3 -m json.tool

# 查看指定设备最新数据
curl -s -u root:taosdata \
  -d "SELECT LAST(ts), COUNT(*) FROM hiccpet_device.imu_data WHERE device_sn='EA:CB:3E:CF:00:11'" \
  http://192.168.33.253:6041/rest/sql | python3 -m json.tool
```

**查看算法服务写入的数据：**

```bash
# 同步断点（含 bind_id）
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT device_id, device_sn, bind_id, user_timezone, last_processed_ts, last_env_ts FROM algo.device_sync_state;" 2>/dev/null

# 行为事件（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT bind_id, behavior, duration_sec, local_start, local_end FROM pet_dog_behavior.d_70 ORDER BY ts_start DESC LIMIT 10;" 2>/dev/null

# 环境数据（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT bind_id, local_date, env_temp, env_humidity, neck_temp FROM pet_dog_environment.d_70;" 2>/dev/null

# 每日行为汇总（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT bind_id, local_date, sleep_min, move_min, scratch_count, sleep_status, move_status, scratch_status FROM pet_dog_daily_summary.d_70 ORDER BY stat_date_ts DESC LIMIT 10;" 2>/dev/null
```

**一键清空 MySQL 数据（调试用）：**

```bash
# 删除所有分表 + 同步断点，重启服务后自动重建
bash scripts/reset_db.sh
```

> **bind_id 说明**：一台设备可先后绑定不同宠物（如原主人的狗去世后更换新狗）。所有数据表均包含 `bind_id` 字段，对应 `hiccpet_petos.device_bind_history.bind_id`，可按绑定期过滤历史数据，避免不同宠物的数据混淆。

---

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [算法流程](#算法流程)
- [特征说明（78 维）](#特征说明78-维)
- [配置项说明](#配置项说明)
- [测试与验证](#测试与验证)

---

## 项目概述

| 项目 | 说明 |
|------|------|
| 语言 | Python 3.11+ |
| 框架 | FastAPI + APScheduler |
| 模型 | LightGBM（CPU，无 GPU 依赖） |
| 时序数据库 | TDengine（taosrest HTTP 连接器，192.168.33.253:6041） |
| 数据库 | MySQL（aiomysql + SQLAlchemy async，192.168.33.253:3306） |
| 部署 | Docker Compose |

核心功能：

1. **行为识别**：每 15 秒（默认）从 TDengine 拉取最新 IMU 数据，滑动窗口分割后提取 78 维特征（与 imu_train 特征空间对齐），LightGBM 分类为 MOVEMENT(1) / SLEEP(2) / SCRATCH(3)，写入 `pet_dog_behavior.d_{device_id}` 表（含本地时间、用户时区、bind_id）。
2. **皮肤健康日评估**：每天凌晨 03:00 UTC 汇总抓挠次数，与个体基线对比计算 Z-score，三层阈值触发分级告警，写入评估表。
3. **基线更新**：每天凌晨 02:00 UTC 用过去 30 天有效数据更新个体基线（EWMA + 软冻结），写入 `pet_dog_scratch_baseline.pet_skin_baseline` 表。
4. **环境数据同步**：每个推理周期同步 TDengine `env_data` 的环境温湿度和体温，按本地日期聚合后写入 `pet_dog_environment.d_{device_id}` 表。

---

## 目录结构

```
algo_service/
├── main.py                      FastAPI 入口，注册路由，启动/关闭调度器
├── config.py                    全局配置（pydantic-settings，支持 .env）
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── db/
│   ├── client.py                MySQL 连接池（aiomysql + SQLAlchemy async）
│   └── tdengine.py              TDengine REST 连接与数据拉取（同步，线程池执行）
│
├── modules/
│   ├── inference/
│   │   ├── model.py             特征提取（78 维）、滑动窗口、LightGBM 推理
│   │   └── handler.py           POST /api/v1/inference/predict 端点
│   ├── assessment/
│   │   └── evaluator.py         日评估引擎、GET /api/v1/assessment/report/{device_id}
│   └── baseline/
│       └── updater.py           基线更新引擎
│
├── scheduler/
│   └── jobs.py                  APScheduler 三个定时任务
│
├── weights/
│   └── ml_lgbm.pkl              imu_train 训练好的模型（joblib 格式，不纳入版本库）
│
├── database_infra/              Git 子模块：数据库基础设施（DDL 等）
├── imu_train/                   Git 子模块：IMU 行为分类模型训练项目
│
├── train/
│   └── train.py                 旧版模型训练脚本（已由 imu_train 子模块替代）
│
└── tests/
    └── unit/                    单元测试（无需真实数据库，88 个测试）
```

---

## 算法流程

### 1. 行为识别（推理模块）

```
TDengine imu_data (accel_x/y/z, gyro_x/y/z, 50 Hz)
  └─ 滑动窗口分割 (3 s / 50% 重叠 → 每窗口 150 样本)
       └─ 特征提取 (78 维 = 6 轴 × 13 维：9 时域 + 4 Welch PSD 频域)
            └─ LightGBM 分类 → MOVEMENT(1) / SLEEP(2) / SCRATCH(3)
                 └─ 合并连续同标签窗口 → 行为事件 (start_time, end_time, confidence)
                      └─ 写入 pet_dog_behavior.d_{device_id}（含 local_start、user_timezone、bind_id）
```

### 2. 日评估（评估模块）

```
当天抓挠事件 → 汇总 scratch_count
  └─ 读取个体基线 (baseline_mean, baseline_std, temp_coef)
       └─ 温度修正 Z-score：z = (count - mean - coef*(temp-20)) / std
            └─ 三层阈值判断（根据基线置信度阶段动态调整）：
                 Phase 0 (valid_days < 3)  : 预热期，不评估
                 Phase 1 (valid_days 3-13) : z>4.0, 连续≥5天, avg_z>5.0
                 Phase 2 (valid_days 14-29): z>3.5, 连续≥4天, avg_z>4.0
                 Phase 3 (valid_days ≥ 30) : z>2.5, 连续≥3天, avg_z>3.5
```

> 温度修正在 Z-score 计算层进行，**不修改原始抓挠计数**。

### 3. 基线更新（基线模块）

```
过去 30 天有效数据
  └─ 指数加权移动平均更新均值：
       正常天 weight=0.05，异常天 weight=0.01（慢渗透，不完全排除）
  └─ 正常天计数重算标准差（≥3 天才更新）
  └─ 温度-抓挠相关系数最小二乘拟合（≥20 对才更新，clip 到 [0, 0.4]）
  └─ 置信度 = min(valid_days / 30, 1.0)
```

---

## 特征说明（78 维）

特征空间与 `imu_train` 子模块保持完全一致，共 78 维：

| 轴 | 时域（9 维） | 频域（4 维，Welch PSD） |
|----|------------|----------------------|
| ax, ay, az, gx, gy, gz | 均值、标准差、最小值、最大值、极差、RMS、偏度、峰度、过零率 | 频谱均值、频谱标准差、主频、频谱熵 |

详见 [modules/inference/README.md](modules/inference/README.md)。

---

## 配置项说明

主要环境变量（均可在 `.env` 或 `docker-compose.yml` 中覆盖）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_HOST` | `192.168.33.253` | MySQL 地址 |
| `DB_PORT` | `3306` | MySQL 端口 |
| `DB_NAME` | `algo` | algo 服务自身数据库名 |
| `DB_USER` / `DB_PASSWORD` | — | MySQL 认证 |
| `BIZ_SCHEMA` | `hiccpet_petos` | 业务库（设备绑定来源） |
| `TD_HOST` | `192.168.33.253` | TDengine 地址 |
| `TD_DATABASE` | `hiccpet_device` | TDengine 数据库名 |
| `FETCH_INTERVAL_SEC` | `15` | 推理周期间隔（秒） |
| `LOG_LEVEL` | `info` | 日志级别 |

详见 [docs/configuration.md](docs/configuration.md)。

---

## 模型训练

模型由 `imu_train` 子模块训练，使用 LightGBM，输出 `ml_lgbm.pkl`（joblib 格式）。

```bash
# 初始化子模块（首次克隆后执行）
git submodule update --init --recursive

# 训练（在 imu_train 目录下执行，详见子模块 README）
cd imu_train && python train.py

# 将训练好的模型复制到 weights/
cp imu_train/output/ml_lgbm.pkl weights/
```

模型文件 `weights/ml_lgbm.pkl` 不纳入版本库，需训练后手动放入。

**部署时更新模型：**

```bash
# 将新模型文件复制到容器内
docker cp weights/ml_lgbm.pkl algo_service:/app/weights/ml_lgbm.pkl
docker restart algo_service
```

---

## 测试与验证

本项目包含两类测试，覆盖不同层面：

### 算法准确率评估

评估分类模型在不同行为场景下的准确率，**一键运行并生成测试报告**：

```bash
# 标准运行（使用已缓存的合成数据，约 30 秒）
docker exec algo_service python tests/run_evaluation.py

# 强制重新生成合成数据后评估（约 3 分钟）
docker exec algo_service python tests/run_evaluation.py --fresh
```

评估方案：5 个行为场景，每个场景分别用**训练内**（原始场景分布）和**训练外**（高抓挠分布，模型从未见过）两套数据测试，共 10 组：

| 场景 | 说明 | 训练内抓挠比 | 训练外抓挠比 |
|------|------|------------|------------|
| S1 Normal | 普通健康犬只 | ~3% | ~15% |
| S2 Active | 活跃型犬只 | ~2% | ~15% |
| S3 Calm | 安静型犬只 | ~2% | ~15% |
| S4 Mild skin | 轻度皮肤问题 | ~9% | ~30% |
| S5 Severe skin | 重度皮肤问题 | ~20% | ~50% |

- 训练内（_in）：使用原始场景分布，其中 S1/S2/S3 与模型训练数据分布完全一致
- 训练外（_out）：同一运动/睡眠模式但抓挠比例大幅提升，模型训练时从未见过

**输出文件：**

| 文件 | 说明 |
|------|------|
| `docs/test_report.md` | 完整测试报告（含所有指标、混淆矩阵、特征重要性）|
| `tests/evaluation/model/scenarios_summary.csv` | 各场景准确率汇总 |
| `tests/evaluation/model/classification_report.csv` | 各类别精确率/召回率/F1 |
| `tests/evaluation/model/confusion_matrix.csv` | 各场景混淆矩阵 |
| `tests/evaluation/model/feature_importance.csv` | 特征重要性排名 |

> ⚠️ **说明**：当前使用合成 IMU 数据（已参照真实传感器采集规律校准）。上线前须使用真实标注数据集替换 `tests/evaluation/` 中的结果文件，重新运行 `--fresh` 以验证实际准确率。

---

### 服务单元测试

针对服务各模块的功能正确性验证，在容器内运行（依赖已安装，无需真实数据库）：

```bash
docker exec algo_service python -m pytest tests/unit/ -v
```

88 个测试，覆盖：特征提取、评分函数、基线算法、TDengine 工具函数、`/health` 接口。
