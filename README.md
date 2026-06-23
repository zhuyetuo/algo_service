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
# 查看日志
docker logs algo_service -f

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
# 同步断点
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT device_id, device_sn, user_timezone, last_processed_ts, last_env_ts FROM algo.device_sync_state;" 2>/dev/null

# 行为事件（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT behavior, duration_sec, local_start, local_end FROM pet_dog_behavior.d_70 ORDER BY ts_start DESC LIMIT 10;" 2>/dev/null

# 环境数据（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -u root -pHicc-mysql-2026 \
  -e "SELECT local_date, env_temp, env_humidity, neck_temp FROM pet_dog_environment.d_70;" 2>/dev/null
```

---

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [算法流程](#算法流程)
- [特征说明（93 维）](#特征说明93-维)
- [配置项说明](#配置项说明)
- [单元测试](#单元测试)

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

1. **行为识别**：每 15 秒（默认）从 TDengine 拉取最新 IMU 数据，滑动窗口分割后提取 93 维特征，LightGBM 分类为 MOVEMENT / SLEEP / SCRATCH / UNKNOWN，写入 `pet_dog_behavior.d_{device_id}` 表（含本地时间和用户时区）。
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
│   │   ├── model.py             特征提取、滑动窗口、LightGBM 推理
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
│   └── behavior_lgbm.pkl        训练好的模型（由 train/train.py 生成，不纳入版本库）
│
├── train/
│   └── train.py                 模型训练脚本（合成数据 + LightGBM）
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
       └─ 特征提取 (93 维，见下表)
            └─ LightGBM 分类 → MOVEMENT(1) / SLEEP(2) / SCRATCH(3)
                 └─ 合并连续同标签窗口 → 行为事件 (start_time, end_time, confidence)
                      └─ 写入 pet_dog_behavior.d_{device_id}（含 local_start、user_timezone）
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

## 特征说明（93 维）

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

模型文件 `weights/behavior_lgbm.pkl` 不纳入版本库，需在本机训练后再启动服务。

```bash
# 在项目根目录执行（需要先安装 requirements.txt）
python train/train.py
```

首次运行约 20–25 秒；训练数据缓存在 `train/data/`，再次运行约 5 秒。

---

## 单元测试

单元测试在容器内运行（依赖已安装，无需真实数据库）：

```bash
docker exec algo_service python -m pytest tests/unit/ -v
```

88 个测试，覆盖：特征提取、评分函数、基线算法、TDengine 工具函数、`/health` 接口。
