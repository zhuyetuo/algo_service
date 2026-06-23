# 测试数据文件说明

本目录存放由测试脚本自动生成的 CSV 缓存文件，**不纳入版本库**（已加入 `.gitignore`）。

首次运行 `python tests/test_1_inference.py` 时自动生成；再次运行直接读缓存，不重新生成。

一键评估脚本 `tests/run_evaluation.py --fresh` 会自动清空本目录并重新生成。

---

## `imu_pool.csv`

IMU 原始窗口池。每类行为生成 2000 个窗口，每窗口 150 个采样点（3 秒 × 50 Hz）。

| 列名 | 类型 | 说明 |
|------|------|------|
| label | str | 行为类别：`movement` / `sleep` / `scratch` |
| window_id | int | 窗口编号（0–1999，每类独立编号） |
| sample_idx | int | 窗口内采样点索引（0–149） |
| ax | float | 加速度计 X 轴（m/s²） |
| ay | float | 加速度计 Y 轴（m/s²） |
| az | float | 加速度计 Z 轴（m/s²，犬只躺卧时约 8.6，活动时因设备倾角变化而变化） |
| gx | float | 陀螺仪 X 轴（rad/s） |
| gy | float | 陀螺仪 Y 轴（rad/s） |
| gz | float | 陀螺仪 Z 轴（rad/s） |

行数：3 类 × 2000 窗口 × 150 采样点 = **900,000 行**

---

## `features_pool.csv`

从 `imu_pool.csv` 提取的特征池，每行对应一个窗口。

| 列名 | 说明 |
|------|------|
| label | 行为类别字符串 |
| window_id | 窗口编号 |
| feat_000 – feat_092 | 93 维特征 |

行数：3 类 × 2000 窗口 = **6,000 行**，列数：**95**（label + window_id + 93 特征）

特征维度说明见 [`modules/inference/README.md`](../../modules/inference/README.md)。

---

## `features_{scenario}_{split}.csv`

按场景和数据集划分保存的特征文件。训练集（`train`）基于 180 天，测试集（`test`）基于 30 天。

| 列名 | 说明 |
|------|------|
| label | 行为类别字符串 |
| label_int | 行为类别整数（1=movement, 2=sleep, 3=scratch） |
| feat_000 – feat_092 | 93 维特征 |

五个场景：

| 文件名前缀 | 场景 | 行为分布 |
|-----------|------|---------|
| `features_S1_Normal_*.csv` | 正常犬 | 运动 55% / 睡眠 40% / 抓挠 5% |
| `features_S2_Active_*.csv` | 活跃犬 | 运动 72% / 睡眠 25% / 抓挠 3% |
| `features_S3_Calm_*.csv` | 安静犬 | 运动 20% / 睡眠 77% / 抓挠 3% |
| `features_S4_Mild_skin_*.csv` | 轻度皮肤病（训练未见） | 抓挠 15% |
| `features_S5_Severe_skin_*.csv` | 重度皮肤病（训练未见） | 抓挠 30% |

S1/S2/S3 用于训练，S4/S5 仅用于评估泛化能力。
