# algo_service

智能宠物项圈算法服务。基于 6 轴 IMU 数据对犬猫行为进行分类，重点检测抓挠行为，并结合个体基线动态评估皮肤健康风险。

---

## 快速启动

**前置条件**：MySQL 和 TDengine 均使用远端服务器（192.168.33.253）。

```bash
# 1. 拉取代码（含模型文件 weights/ml_rf.pkl，无需单独训练）
git clone <repo> && cd algo_service

# 2. 配置（按实际环境修改 .env，默认指向 192.168.33.253）
cp .env.example .env

# 3. 启动服务（首次自动构建镜像，约 1-2 分钟）
docker compose up -d --build

# 4. 确认两个数据库连接正常
curl http://localhost:8000/health
# 期望返回：{"status":"ok","mysql":"ok","tdengine":"ok"}
```

> `docker compose up -d --build` 会检测本地是否已有镜像；若不存在或 Dockerfile/依赖有变化则自动重新构建，构建完成后启动容器。适用于单机部署或开发环境。

**K8s 部署（生产环境）：**

K8s 不使用 docker compose，需先将镜像推送到私有镜像仓库：

```bash
# 构建并推送镜像（替换 <registry> 为实际仓库地址，如 registry.hiccpet.com）
docker build -t <registry>/algo_service:v1.0 .
docker push <registry>/algo_service:v1.0

# K8s 通过 Deployment 拉取镜像，环境变量通过 ConfigMap / Secret 注入
# 参考 docker-compose.yml 中的 environment 块配置对应的 K8s ConfigMap
```

K8s Deployment 中将 `image` 设为推送的镜像地址，并通过 `envFrom` 引用 ConfigMap / Secret 传入 `DB_HOST`、`TD_HOST`、`DB_PASSWORD` 等变量，与 docker-compose.yml 中的 `environment` 块一一对应。

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
| MySQL | 192.168.33.253:30100 | algo 服务自身数据库（行为事件、评估结果、基线、同步断点） |
| MySQL | 192.168.33.253:30100 | 业务库 `hiccpet_petos`（设备绑定、用户时区） |
| TDengine | 192.168.33.253:6041 | 时序原始数据（IMU、环境、体温） |

**TDengine 超级表结构：**

| 超级表 | 字段 | 说明 |
|--------|------|------|
| `imu_data` | `accel_x/y/z`, `gyro_x/y/z`, TAG:`device_sn` | 6 轴 IMU，设备上报 25Hz |
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
docker exec local-mysql8 mysql -h 192.168.33.253 -P 30100 -u root -pHicc-pet-mysql-2026 \
  -e "SELECT device_id, device_sn, bind_id, user_timezone, last_processed_ts, last_env_ts FROM algo.device_sync_state;" 2>/dev/null

# 行为事件（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -P 30100 -u root -pHicc-pet-mysql-2026 \
  -e "SELECT bind_id, behavior, duration_sec, local_start, local_end FROM pet_dog_behavior.d_70 ORDER BY ts_start DESC LIMIT 10;" 2>/dev/null

# 环境数据（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -P 30100 -u root -pHicc-pet-mysql-2026 \
  -e "SELECT bind_id, local_date, env_temp, env_humidity, neck_temp FROM pet_dog_environment.d_70;" 2>/dev/null

# 每日行为汇总（设备 70）
docker exec local-mysql8 mysql -h 192.168.33.253 -P 30100 -u root -pHicc-pet-mysql-2026 \
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
- [特征说明](#特征说明)
- [离线回补（设备链路未打通时用）](#离线回补设备链路未打通时用)
- [IMU 量纲统一](#imu-量纲统一)
- [配置项说明](#配置项说明)
- [测试与验证](#测试与验证)

---

## 项目概述

| 项目 | 说明 |
|------|------|
| 语言 | Python 3.11+ |
| 框架 | FastAPI + APScheduler |
| 模型 | RandomForest（scikit-learn，CPU，无 GPU 依赖） |
| 时序数据库 | TDengine（taosrest HTTP 连接器，192.168.33.253:6041） |
| 数据库 | MySQL（aiomysql + SQLAlchemy async，192.168.33.253:30100） |
| 部署 | Docker Compose |

核心功能：

1. **行为识别**：每 15 秒（默认）从 TDengine 拉取最新 IMU 数据，按 2s/50% 重叠滑动窗口分割，重力轴对齐后提取 171/193 维特征（与 imu_train 特征空间逐行对齐），RandomForest 分类为 MOVEMENT(1) / SLEEP(2) / SCRATCH(3)，写入 `pet_dog_behavior.d_{device_id}` 表（含本地时间、用户时区、bind_id、behavior_label 中文标签）。
2. **皮肤健康日评估**：每天凌晨 03:00 UTC 汇总抓挠次数，与个体基线对比计算 Z-score，三层阈值触发分级告警，写入评估表。
3. **基线更新**：每天凌晨 02:00 UTC 用过去 30 天有效数据更新个体基线（EWMA + 软冻结），写入 `pet_dog_scratch_baseline.pet_skin_baseline` 表。
4. **环境数据同步**：每个推理周期同步 TDengine `env_data` 的环境温湿度和体温，按本地日期聚合后写入 `pet_dog_environment.d_{device_id}` 表。

---

## 目录结构

```
algo_service/
├── main.py                      FastAPI 入口，注册路由，启动/关闭调度器
├── config.py                    全局配置（pydantic-settings，支持 .env）
├── timezones.py                 时区名归一（CST/PRC/+08:00 → IANA）
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
│   │   ├── features.py          特征提取（imu_train src/ml/features.py 的逐行移植）
│   │   ├── gravity.py           重力轴对齐 + 原始姿态角（imu_train gravity_align.py 移植）
│   │   ├── units.py             量纲统一：设备上报单位 → 模型训练单位
│   │   ├── model.py             滑动窗口、特征布局自动识别、RandomForest 推理、事件合并
│   │   └── handler.py           POST /api/v1/inference/predict 端点
│   ├── assessment/
│   │   └── evaluator.py         日评估引擎、GET /api/v1/assessment/report/{device_id}
│   └── baseline/
│       └── updater.py           基线更新引擎
│
├── scheduler/
│   └── jobs.py                  APScheduler 三个定时任务
│
├── backfill/
│   ├── run_backfill.py          离线回补：历史 IMU 数据 → 推理 → 写库
│   ├── diagnose_signal.py       信号量级诊断：核对 IMU 单位是否与训练单位一致
│   └── import_infer.py          导入 imu_train 那边跑出来的推理结果
│
├── weights/
│   ├── ml_rf.pkl                RandomForest 模型（joblib，25Hz，window=2s，gravity_aligned）
│   └── ml_rf.json               训练元数据（采样率、窗口参数、类别、精度）
│
├── database_infra/              Git 子模块：数据库基础设施（DDL 等）
├── imu_train/                   Git 子模块：IMU 行为分类模型训练项目
│
└── tests/
    └── unit/                    单元测试（无需真实数据库，151 个测试）
```

---

## 算法流程

### 1. 行为识别（推理模块）

```
TDengine imu_data (accel_x/y/z, gyro_x/y/z, 25 Hz)
  └─ 量纲统一（设备上报单位 → 模型训练单位，见「IMU 量纲统一」）
       └─ 滑动窗口分割 (2 s / 50% 重叠 / 步长 1 s → 每窗口 50 样本)
            └─ 原始 acc 算 pitch/roll → 重力轴对齐 → 姿态角拼到通道末尾（顺序与训练侧一致）
                 └─ 特征提取 (6通道171维 / 8通道193维，与 imu_train 完全一致)
                      └─ RandomForest 分类 → MOVEMENT(1) / SLEEP(2) / SCRATCH(3)
                           └─ 多数票平滑（±2 帧，消除单窗口随机噪声翻转）
                                └─ 合并连续同标签窗口 → 行为事件
                                     └─ 写入 pet_dog_behavior.d_{device_id}
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

## 特征说明

特征提取与 `imu_train/src/ml/features.py` **逐行对齐**（`modules/inference/features.py`
是它的直接移植，改动前必须先同步 imu_train）。通道顺序：

```
0:3  acc_x/y/z    重力对齐后
3:6  gyr_x/y/z    重力对齐后
6:8  pitch, roll  原始未对齐姿态角（必须在重力对齐前算）
```

| 特征组 | 维度 | 内容 |
|--------|------|------|
| 时域（逐通道） | 通道数 × 11 | 均值、标准差、最小值、最大值、极差、RMS、偏度、峰度、**均值穿越率**、IQR、峰值计数 |
| 频域（仅前 6 通道） | 6 × 8 | 频谱均值、频谱标准差、主频、频谱熵 + 4 个分频段能量占比 |
| 全局跨通道 | 8 | acc/gyro 的 SMA，各自三轴两两相关系数 |
| 模长 | 38 | acc 模长、gyro 模长各自的时域(11)+频域(8) |
| Jerk | 11 | 加加速度模长的时域统计量 |

合计：**6 通道 → 171 维，8 通道 → 193 维**。

服务启动时会读模型的 `n_features_in_` **自动识别**该用哪套特征（193 / 171 / 旧版 78），
并在日志里打印识别结果；无法识别时直接报错退出，不会静默用错特征。

> 均值穿越率统计的是穿越**窗口自身均值**的次数，而不是绝对 0——重力对齐后 acc_z 恒有
> 直流偏置，穿越绝对 0 没有意义。

详见 [modules/inference/README.md](modules/inference/README.md)。

---

## 离线回补（设备链路未打通时用）

设备还没接入时，可以把 TDengine 里已有的历史数据跑一遍推理写进数据库：

```bash
# 看看 TDengine 里有哪些设备、覆盖哪段时间
docker exec algo_service python backfill/run_backfill.py --list-devices

# 试跑（不写库），确认结果分布合理
docker exec algo_service python backfill/run_backfill.py --date 2026-08-19 --dry-run

# 正式回补 8月19日，并跑当天皮肤评估
docker exec algo_service python backfill/run_backfill.py --date 2026-08-19 --assess
```

不会推进 `device_sync_state` 断点，不影响线上增量推理；重复回补同一天是幂等的。

如果推理是在 **imu_train 那边**跑的（录制数据 + `run_review_bins_all_days.sh`），
直接把 `RESULT_ROOT/{day}/` 目录拷过来导入，不需要本服务再算一遍：

```bash
docker exec algo_service python backfill/import_infer.py \
    --input infer_result_majority/2026_8_19 \
    --device-map backfill/device_map.csv --dry-run
```

完整说明见 [backfill/README.md](backfill/README.md)。

---

## IMU 量纲统一

**imu_train 训练时不做量纲统一**（`loader_custom.py`：「单位不限（训练时不做量纲统一）」），
训练 CSV 是什么单位，模型学到的就是什么量级。而特征里绝大多数是**有量纲的绝对量**——
mean / std / min / max / range / rms / iqr、SMA、acc/gyro 模长、jerk——
单位差一个量级，这些特征整体平移，直接落到训练分布之外。

> 频谱熵、分频段能量占比、相关系数是无量纲的，不受影响。所以单位错了往往不是"全错"，
> 而是**置信度莫名偏低、某几类死活预测不出来**，比全错更难查。

已知的单位约定：

| | 训练侧（imu_train） | 设备侧（TDengine） |
|---|---|---|
| 加速度 | `labelstudio_to_custom.py` 统一换算到 **m/s²** | m/s²（静止 \|acc\| ≈ 9.6~9.8） |
| 角速度 | **原样透传不换算** → WitMotion 原始输出为 **deg/s** | 需实测确认（TF 固件文档为 rad/s） |

**上线前务必用真实数据确认，不要照搬上表**（设备固件可能改过）：

```bash
docker exec algo_service python backfill/diagnose_signal.py --device-sn EA:CB:3E:CF:00:11
```

工具会打印 `|acc|` / `|gyro|` 的量级并给出判断——静止时 `|acc|` 必然等于重力常数，
中位数 ≈9.8 就是 m/s²，≈1.0 就是 g。角速度没有这种固定锚点，按经验量级判断：
犬只日常活动 deg/s 是几十~几百，换成 rad/s 则是零点几~几。

确认后在 `docker-compose.yml` 或 `.env` 设置，例如角速度实际是 rad/s：

```bash
IMU_DEVICE_GYRO_UNIT=rads
# docker compose down && docker compose up -d 生效
```

默认四个单位全部一致（换算系数 = 1.0，不改变原有行为）。启动日志的
「量纲统一」一行会打印最终生效的换算系数，可核对。

新模型建议在 `ml_rf.json` 里写上 `acc_unit` / `gyro_unit`，服务会优先采用，
不用再靠配置猜；缺失时启动会打警告提醒。

---

## 配置项说明

主要环境变量（均可在 `.env` 或 `docker-compose.yml` 中覆盖）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_HOST` | `192.168.33.253` | MySQL 地址 |
| `DB_PORT` | `30100` | MySQL 端口 |
| `DB_NAME` | `algo` | algo 服务自身数据库名 |
| `DB_USER` / `DB_PASSWORD` | — | MySQL 认证 |
| `BIZ_SCHEMA` | `hiccpet_petos` | 业务库（设备绑定来源） |
| `TD_HOST` | `192.168.33.253` | TDengine 地址 |
| `TD_DATABASE` | `hiccpet_device` | TDengine 数据库名 |
| `FETCH_INTERVAL_SEC` | `15` | 推理周期间隔（秒） |
| `IMU_SAMPLE_RATE` | `25` | IMU 采样率（Hz），须与模型训练参数一致 |
| `WINDOW_SECONDS` | `2.0` | 滑动窗口长度（秒） |
| `WINDOW_OVERLAP` | `0.5` | 窗口重叠比例（0.5 = 步长为窗口一半） |
| `CONFIDENCE_THRESHOLD` | `0.0` | 置信度阈值，低于此值标记为 UNKNOWN（0.0 = 禁用） |
| `SMOOTH_WINDOW` | `5` | 逐窗口多数票平滑的窗口数（奇数，1 = 关闭） |
| `IMU_DEVICE_ACC_UNIT` | `ms2` | 设备上报的加速度单位：`ms2`=m/s²，`g`=重力单位 |
| `IMU_DEVICE_GYRO_UNIT` | `dps` | 设备上报的角速度单位：`dps`=deg/s，`rads`=rad/s |
| `IMU_MODEL_ACC_UNIT` | `ms2` | 模型训练时的加速度单位（`ml_rf.json` 有 `acc_unit` 时以它为准） |
| `IMU_MODEL_GYRO_UNIT` | `dps` | 模型训练时的角速度单位（`ml_rf.json` 有 `gyro_unit` 时以它为准） |
| `VERBOSE_INFERENCE` | `false` | 开启后每个推理窗口输出一行详细日志（含片上时间、置信度） |
| `CST_TIMEZONE` | `America/New_York` | `"CST"` 按哪个时区解释（全球有歧义；生产库里 "CST" 用户与真实美国用户同批出现，经业务确认按美国东部时间） |
| `LOG_LEVEL` | `info` | 日志级别 |

修改环境变量后执行 `docker compose down && docker compose up -d` 生效（无需重新 `--build`）。

---

## 模型训练

模型由 `imu_train` 子模块训练，使用 RandomForest，输出 `ml_rf.pkl`（joblib 格式）。**`weights/ml_rf.pkl` 和 `weights/ml_rf.json` 已随代码库提交，服务启动时直接加载，无需手动训练。**

若需重新训练（例如更换训练数据或调整超参），在 `imu_train` 目录下完成训练后：

```bash
# 将新模型文件覆盖到 weights/
cp imu_train/output/ml_rf.pkl weights/
cp imu_train/output/ml_rf.json weights/

# 重启服务（无需重新构建镜像）
docker compose restart
```

服务启动时会自动读取 `weights/ml_rf.json` 中的训练参数（采样率、窗口长度、步长、是否重力对齐），并与当前推理参数对比，不一致时打印警告及修复建议。

**K8s 环境更新模型：**

```bash
# 将新模型文件复制到正在运行的 Pod 内（pod-name 替换为实际 Pod 名）
kubectl cp weights/ml_rf.pkl <namespace>/<pod-name>:/app/weights/ml_rf.pkl
kubectl cp weights/ml_rf.json <namespace>/<pod-name>:/app/weights/ml_rf.json
kubectl rollout restart deployment/algo-service -n <namespace>
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

151 个测试，覆盖：特征提取与维度、重力对齐、姿态角、量纲换算与单位诊断、滑动窗口、事件合并、标签平滑、
评分函数、基线算法、TDengine 工具函数、`/health` 接口。
