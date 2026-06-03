# algo_service

智能宠物项圈算法服务。基于 6 轴 IMU 数据对犬猫行为进行分类，重点检测抓挠行为，并结合个体基线动态评估皮肤健康风险。

---

## 快速启动

**前置条件**：PostgreSQL 和 TDengine 已在同服务器运行。

```bash
# 1. 拉取代码
git clone <repo> && cd algo_service

# 2. 配置（默认值已对齐本地环境，通常无需改动）
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

# 停止服务
docker compose stop

# 停止并删除容器
docker compose down

# 停止并删除容器 + 数据卷（慎用，会清空 model_weights）
docker compose down -v
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

IMU 窗口为 `(150, 6)` 数组，列顺序：`ax ay az gx gy gz`（加速度计 + 陀螺仪，各 3 轴）。

特征按以下顺序拼接，共 93 维：

### 每轴特征（12 维 × 6 轴 = 72 维）

对 6 个通道（ax / ay / az / gx / gy / gz）各自独立计算：

| 轴内偏移 | 特征名 | 说明 |
|----------|--------|------|
| +0  | mean             | 均值 |
| +1  | std              | 标准差 |
| +2  | min              | 最小值 |
| +3  | max              | 最大值 |
| +4  | peak_to_peak     | 峰峰值（max − min） |
| +5  | rms              | 均方根值 √(mean(x²)) |
| +6  | zero_crossing    | 过零次数（信号符号变化次数） |
| +7  | skewness         | 偏度（分布不对称程度） |
| +8  | kurtosis         | 峰度（分布尖锐程度） |
| +9  | dominant_freq    | 主频率（Hz，功率谱最大分量对应频率） |
| +10 | spectral_energy  | 频谱能量（∑|X(f)|²） |
| +11 | spectral_entropy | 频谱熵（−∑p·log p，p 为归一化谱） |

完整索引映射：

| 特征索引 | 通道 | 特征 |
|----------|------|------|
| feat_000 – feat_011 | ax | 上表 12 项 |
| feat_012 – feat_023 | ay | 上表 12 项 |
| feat_024 – feat_035 | az | 上表 12 项 |
| feat_036 – feat_047 | gx | 上表 12 项 |
| feat_048 – feat_059 | gy | 上表 12 项 |
| feat_060 – feat_071 | gz | 上表 12 项 |

### 合力特征（9 维 × 2 = 18 维）

对加速度计合力 `‖[ax,ay,az]‖` 和陀螺仪合力 `‖[gx,gy,gz]‖` 各计算上表中的时域 9 项（无频域）：

| 特征索引 | 通道 | 特征 |
|----------|------|------|
| feat_072 – feat_080 | acc_mag（加速度计合力） | mean / std / min / max / peak_to_peak / rms / zero_crossing / skewness / kurtosis |
| feat_081 – feat_089 | gyr_mag（陀螺仪合力） | 同上 |

### 跨轴相关系数（3 维）

加速度计三轴两两 Pearson 相关系数：

| 特征索引 | 说明 |
|----------|------|
| feat_090 | corr(ax, ay) |
| feat_091 | corr(ax, az) |
| feat_092 | corr(ay, az) |

---

### 不同行为的典型特征值

| 特征 | MOVEMENT（运动） | SLEEP（睡眠） | SCRATCH（抓挠） |
|------|----------|-------|---------|
| ax 主频 (feat_009) | 1.5–2.5 Hz | 0.2–0.4 Hz | 4–8 Hz |
| ax RMS (feat_005) | 0.4–0.8 | ~0.02 | 1.5–3.0 |
| acc_mag std (feat_073) | 中等 | 极小 | 大 |
| corr(ax,ay) (feat_090) | 中等正相关 | 接近 0 | 强相关 |

---

## 数据文件说明

### `tests/data/imu_pool.csv`

IMU 原始窗口池，由 `test_1_inference.py` 第一次运行时生成。

| 列名 | 类型 | 说明 |
|------|------|------|
| label | str | 行为类别：movement / sleep / scratch |
| window_id | int | 窗口编号（0–1999，每类 2000 个） |
| sample_idx | int | 窗口内采样点索引（0–149，共 150 点） |
| ax | float | 加速度计 X 轴（m/s²） |
| ay | float | 加速度计 Y 轴（m/s²） |
| az | float | 加速度计 Z 轴（m/s²，静止时约 9.8） |
| gx | float | 陀螺仪 X 轴（rad/s） |
| gy | float | 陀螺仪 Y 轴（rad/s） |
| gz | float | 陀螺仪 Z 轴（rad/s） |

行数：3 类 × 2000 窗口 × 150 采样点 = **900,000 行**

### `tests/data/features_pool.csv`

从 `imu_pool.csv` 提取的特征，每行对应一个窗口。

| 列名 | 说明 |
|------|------|
| label | 行为类别字符串 |
| window_id | 窗口编号 |
| feat_000 – feat_092 | 93 维特征（详见上方特征表） |

行数：3 类 × 2000 窗口 = **6,000 行**，列数：**95**（label + window_id + 93 特征）

### `tests/data/features_{scenario}_{split}.csv`

按场景和数据划分保存的特征数据集，训练集（`train`）基于 180 天，测试集（`test`）基于 30 天。

| 列名 | 说明 |
|------|------|
| label | 行为类别字符串 |
| label_int | 行为类别整数（1=movement, 2=sleep, 3=scratch） |
| feat_000 – feat_092 | 93 维特征 |

五个场景：

| 文件名前缀 | 场景 | 说明 |
|-----------|------|------|
| features_S1_Normal | 正常犬 | 运动 55% / 睡眠 40% / 抓挠 5% |
| features_S2_Active | 活跃犬 | 运动 72% / 睡眠 25% / 抓挠 3% |
| features_S3_Calm | 安静犬 | 运动 20% / 睡眠 77% / 抓挠 3% |
| features_S4_Mild_skin | 轻度皮肤病（训练未见） | 抓挠 15% |
| features_S5_Severe_skin | 重度皮肤病（训练未见） | 抓挠 30% |

---

## 配置项说明

所有配置项均可通过环境变量或 `.env` 文件覆盖（大写下划线形式）。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DB_HOST` | `postgres` | **PostgreSQL 主机地址（必填）** |
| `DB_PORT` | `5432` | PostgreSQL 端口 |
| `DB_NAME` | `algo` | 数据库名 |
| `DB_USER` | `algo` | 数据库用户 |
| `DB_PASSWORD` | `algo` | 数据库密码 |
| `TD_HOST` | `tdengine` | **TDengine 主机地址（必填）** |
| `TD_PORT` | `6041` | TDengine REST API 端口 |
| `TD_USER` | `root` | TDengine 用户名 |
| `TD_PASSWORD` | `taosdata` | TDengine 密码 |
| `TD_DATABASE` | `pet_collar_raw` | TDengine 数据库名 |
| `TD_SUPERTABLE` | `imu_raw` | IMU 超级表名 |
| `TD_BATCH_SIZE` | `50000` | 每次拉取最大行数 |
| `MODEL_PATH` | `weights/behavior_lgbm.pkl` | 模型文件路径 |
| `IMU_SAMPLE_RATE` | `50` | IMU 采样率（Hz） |
| `WINDOW_SECONDS` | `3` | 分类滑动窗口时长（秒） |
| `WINDOW_OVERLAP` | `0.5` | 滑动窗口重叠比例 |
| `FETCH_INTERVAL_MIN` | `15` | 推理调度间隔（分钟） |
| `BASELINE_UPDATE_CRON` | `0 2 * * *` | 基线更新 cron（UTC） |
| `ASSESSMENT_CRON` | `0 3 * * *` | 日评估 cron（UTC） |
| `PHASE1_THRESHOLD_Z` | `4.0` | 早期阶段 Z-score 阈值 |
| `PHASE2_THRESHOLD_Z` | `3.5` | 中期阶段 Z-score 阈值 |
| `PHASE3_THRESHOLD_Z` | `2.5` | 稳定阶段 Z-score 阈值 |
| `BASELINE_STD_FLOOR` | `2.0` | 基线标准差下限（防除零） |
| `MIN_WEAR_MINUTES` | `480` | 每天最少佩戴时长（分钟） |

---

## 测试模块

### test_1_inference.py — 行为识别

```bash
python tests/test_1_inference.py
```

- 生成 5 场景 × 180 天合成 IMU 数据，保存至 `tests/data/`
- 用 S1/S2/S3 训练 LightGBM，对全部 5 场景评估（含 S4/S5 未见场景）
- 输出准确率、分类报告、混淆矩阵、抓挠 F1
- **再次运行直接加载 CSV，不重新生成**

### test_2_assessment.py — 皮肤健康评估

```bash
python tests/test_2_assessment.py
```

5 个场景验证评估逻辑：

| 场景 | 内容 | 预期 |
|------|------|------|
| A 正常 | 无疾病 | 0 告警 |
| B 皮肤病 | 第 45–75 天发病（+10 抓挠） | ≥1 告警，恢复后消除 |
| C 季节性 | 夏季温度升高 | 0 误报（温度修正生效） |
| D 缓慢增加 | 抓挠 180 天线性增加 | 基线跟随，≤1 告警 |
| E 数据缺口 | 多段设备离线 + 疾病期 | 无缺口误报 |

输出诊断表格 + 折线图 `tests/test_2_assessment.png`

### test_3_baseline.py — 基线更新

```bash
python tests/test_3_baseline.py
```

| 子测试 | 验证内容 | 通过条件 |
|--------|----------|----------|
| T1 正常收敛 | 90 天正常数据 | 最终误差 < 1.0 |
| T2 异常渗透 | 30 天疾病期 | 基线偏移 < 3.0 |
| T3 温度系数 | 120 天温度相关 | 系数误差 < 0.08 |
| T4 阶段转换 | 180 天完整周期 | 置信度达到 1.0，均值误差 < 1.5 |

输出 PASS/FAIL + 图表 `tests/test_3_baseline.png`

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
