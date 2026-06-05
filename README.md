# algo_service

智能宠物项圈算法服务。基于 6 轴 IMU 数据对犬猫行为进行分类，重点检测抓挠行为，并结合个体基线动态评估皮肤健康风险。

---

## 快速启动

**前置条件**：PostgreSQL 和 TDengine 已在同服务器运行。

```bash
# 1. 拉取代码
git clone <repo> && cd algo_service

# 2. 配置
cp .env.example .env

# 3. 训练模型（首次运行约 20-25 秒，生成 weights/behavior_lgbm.pkl）
python train/train.py

# 4. 启动服务
docker compose up -d --build

# 5. 确认两个数据库连接正常
curl http://localhost:8000/health
```

```bash
# 查看日志
docker logs algo_service -f

# 清空日志（先停容器，删完再启动）
docker compose down
rm -f logs/algo_service*.log logs/algo_service*.log.zip
docker compose up -d

# 停止服务
docker compose stop

# 停止并删除容器
docker compose down

# 停止并删除容器 + 数据卷（慎用，会清空 model_weights）
docker compose down -v
```

---

## 数据基础设施

测试数据由独立仓库 [`database_infra`](database_infra/) 提供（已作为 git submodule 引入）。

```bash
# 首次 clone 时同步 submodule
git submodule update --init

# 启动 PostgreSQL + TDengine
cd database_infra && docker compose up -d

# 生成 180 天历史数据（确认 imu_raw_db.py 里 DAYS=180）
python imu_raw_db.py
python behavior_db.py
python environment_db.py
python pg_seed.py

# 或一键重置并重新生成
./reset_and_load.sh
```

**查询 TDengine 数据量**（无需 algo_service 容器在线，直接查 TDengine 容器）：

```bash
# 查看指定设备的 IMU 数据范围
curl -s -u root:taosdata \
  -d "SELECT COUNT(*), FIRST(ts), LAST(ts) FROM pet_collar_raw.imu_raw WHERE device_id=1" \
  http://localhost:6041/rest/sql | python3 -m json.tool

# 查看所有设备 IMU 数据量汇总
curl -s -u root:taosdata \
  -d "SELECT device_id, COUNT(*) FROM pet_collar_raw.imu_raw GROUP BY device_id ORDER BY device_id" \
  http://localhost:6041/rest/sql | python3 -m json.tool
```

> 详细说明见 [docs/deployment.md](docs/deployment.md)

---

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [算法流程](#算法流程)
- [特征说明（93 维）](#特征说明93-维)
- [数据文件说明](#数据文件说明)
- [配置项说明](#配置项说明)
- [测试模块](#测试模块)

---

## 项目概述

| 项目 | 说明 |
|------|------|
| 语言 | Python 3.11+ |
| 框架 | FastAPI + APScheduler |
| 模型 | LightGBM（CPU，无 GPU 依赖） |
| 时序数据库 | TDengine（taosrest HTTP 连接器） |
| 数据库 | PostgreSQL（asyncpg + SQLAlchemy async） |
| 部署 | Docker Compose |

核心功能：

1. **行为识别**：每 15 分钟从 TDengine 拉取最新 IMU 数据，滑动窗口分割后提取特征，LightGBM 分类为 MOVEMENT / SLEEP / SCRATCH / UNKNOWN，按设备写入 `behavior.{device_sn}` 表。
2. **皮肤健康日评估**：每天凌晨 03:00 UTC 汇总抓挠次数，与个体基线对比计算 Z-score，三层阈值（Z 值、连续天数、平均 Z）触发分级告警，写入 `skin_assessment.{device_sn}` 表。
3. **基线更新**：每天凌晨 02:00 UTC 用过去 30 天的有效数据更新个体基线（EWMA + 软冻结），写入 `baseline.{device_sn}` 表。

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
│   ├── client.py                AsyncSessionLocal（asyncpg）
│   └── tdengine.py              TDengine REST 连接与数据拉取（同步）
│
├── modules/
│   ├── inference/
│   │   ├── model.py             特征提取、滑动窗口、LightGBM 推理
│   │   └── handler.py           POST /predict  端点
│   ├── assessment/
│   │   └── evaluator.py         日评估引擎、GET /report/{device_sn}
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
│   ├── train.py                 模型训练脚本（合成数据 + LightGBM）
│   └── data/                    训练缓存 CSV（不纳入版本库）
│
└── tests/
    ├── data/                    测试生成的 CSV 数据（不纳入版本库）
    ├── test_1_inference.py      行为识别测试（生成数据 + 训练 + 评估）
    ├── test_2_assessment.py     皮肤健康评估测试（5 场景 × 180 天）
    └── test_3_baseline.py       基线更新单元测试（4 个子测试）
```

---

## 算法流程

### 1. 行为识别（推理模块）

```
IMU 原始数据 (N×6, 50 Hz)
  └─ 滑动窗口分割 (3 s / 50% 重叠 → 每窗口 150 样本)
       └─ 特征提取 (93 维，见下表)
            └─ LightGBM 分类 → MOVEMENT(1) / SLEEP(2) / SCRATCH(3)
                 └─ 合并连续同标签窗口 → 行为事件 (start_time, end_time, confidence)
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

## 数据文件说明

详见 [tests/data/README.md](tests/data/README.md)。

---

## 配置项说明

详见 [docs/configuration.md](docs/configuration.md)。

---

## 测试模块

详见 [tests/README.md](tests/README.md)。

---

## 模型训练

模型文件 `weights/behavior_lgbm.pkl` 不纳入版本库，需在本机训练后再启动服务。

```bash
# 在项目根目录执行（需要先安装 requirements.txt）
cd algo_service
python train/train.py
```

**训练过程：**
1. 生成合成 IMU 数据（S1 正常 / S2 活跃 / S3 安静，各 180 天）
2. 提取 93 维特征，80/20 划分训练集/验证集
3. 训练 LightGBM（300 棵树，早停 30 轮）
4. 保存模型到 `weights/behavior_lgbm.pkl`

首次运行约 20–25 秒；训练数据缓存在 `train/data/`，再次运行约 5 秒。删除 `train/data/` 可强制重新生成数据。

> 如果需要更高精度的模型，也可用 `tests/test_1_inference.py` 替代——它会额外输出 5 场景评估报告和混淆矩阵，模型同样保存到 `weights/behavior_lgbm.pkl`。

---

## 单元测试

```bash
# 安装测试依赖
pip install -r requirements-dev.txt

# 运行全部单元测试（无需真实数据库）
python -m pytest tests/unit/ -v
```

88 个测试，覆盖特征提取、评分函数、基线算法、TDengine 工具函数、`/health` 接口。
