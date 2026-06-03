# 特征说明（93 维）

`extract_features(window, fs)` 接收形状为 `(150, 6)` 的 IMU 窗口（3 秒 × 50 Hz），
列顺序为 `ax ay az gx gy gz`（加速度计 + 陀螺仪，各 3 轴），返回 93 维特征向量。

---

## 每轴特征（12 维 × 6 轴 = 72 维）

对 6 个通道（ax / ay / az / gx / gy / gz）各自独立计算时域 9 项 + 频域 3 项：

| 轴内偏移 | 特征名 | 类型 | 说明 |
|----------|--------|------|------|
| +0  | mean             | 时域 | 均值 |
| +1  | std              | 时域 | 标准差 |
| +2  | min              | 时域 | 最小值 |
| +3  | max              | 时域 | 最大值 |
| +4  | peak_to_peak     | 时域 | 峰峰值（max − min） |
| +5  | rms              | 时域 | 均方根值 √(mean(x²)) |
| +6  | zero_crossing    | 时域 | 过零次数（信号符号变化次数） |
| +7  | skewness         | 时域 | 偏度（分布不对称程度） |
| +8  | kurtosis         | 时域 | 峰度（分布尖锐程度） |
| +9  | dominant_freq    | 频域 | 主频率（Hz，功率谱最大分量对应频率） |
| +10 | spectral_energy  | 频域 | 频谱能量（∑\|X(f)\|²） |
| +11 | spectral_entropy | 频域 | 频谱熵（−∑p·log p，p 为归一化谱） |

完整索引映射：

| 特征索引 | 通道 |
|----------|------|
| feat_000 – feat_011 | ax |
| feat_012 – feat_023 | ay |
| feat_024 – feat_035 | az |
| feat_036 – feat_047 | gx |
| feat_048 – feat_059 | gy |
| feat_060 – feat_071 | gz |

---

## 合力特征（9 维 × 2 = 18 维）

对加速度计合力 `‖[ax,ay,az]‖` 和陀螺仪合力 `‖[gx,gy,gz]‖` 各计算时域 9 项（无频域）：

| 特征索引 | 通道 | 特征 |
|----------|------|------|
| feat_072 – feat_080 | acc_mag（加速度计合力） | mean / std / min / max / peak_to_peak / rms / zero_crossing / skewness / kurtosis |
| feat_081 – feat_089 | gyr_mag（陀螺仪合力） | 同上 |

---

## 跨轴相关系数（3 维）

加速度计三轴两两 Pearson 相关系数：

| 特征索引 | 说明 |
|----------|------|
| feat_090 | corr(ax, ay) |
| feat_091 | corr(ax, az) |
| feat_092 | corr(ay, az) |

---

## 不同行为的典型特征值

| 特征 | MOVEMENT（运动） | SLEEP（睡眠） | SCRATCH（抓挠） |
|------|-----------------|--------------|----------------|
| ax 主频 (feat_009) | 1.5–2.5 Hz | 0.2–0.4 Hz | 4–8 Hz |
| ax RMS (feat_005) | 0.4–0.8 | ~0.02 | 1.5–3.0 |
| acc_mag std (feat_073) | 中等 | 极小 | 大 |
| corr(ax,ay) (feat_090) | 中等正相关 | 接近 0 | 强相关 |

---

## 相关代码

- 特征提取实现：[`model.py`](model.py) — `_time_features()` / `_freq_features()` / `extract_features()`
- 单元测试：[`tests/unit/test_inference_model.py`](../../tests/unit/test_inference_model.py)
