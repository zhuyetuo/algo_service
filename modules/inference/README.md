# 推理模块

特征提取与预处理是 `imu_train` 的**逐行移植**，任何改动都必须先同步 imu_train，
否则推理特征会偏离模型训练时的分布，置信度和准确率都会掉。

| 本模块文件 | 对应 imu_train 源文件 |
|-----------|----------------------|
| `features.py` | `src/ml/features.py` |
| `gravity.py` | `src/data/gravity_align.py` |
| `model.py` 的窗口/对齐顺序 | `src/data/preprocess.py` 的 `process_split()` |

---

## 处理顺序

顺序不能调换 —— `raw_tilt` 必须用**原始未对齐**的加速度计算：
`gravity_align` 会把每个窗口的平均加速度旋到 +Z，这会把窗口的绝对倾角抹成 ~0，
"趴着 / 坐着 / 站着"的姿态信息就没了。训练侧就是先算姿态角再对齐的。

```
(T, 6) 原始窗口  [ax ay az gx gy gz]
  │
  ├─ raw_tilt(acc)          → (T, 2) pitch/roll   ← 用原始 acc
  └─ gravity_align(window)  → (T, 6) 重力对齐到 +Z
        │
        └─ concat → (T, 8)  [acc对齐 ×3, gyr对齐 ×3, pitch, roll]
              └─ extract_one() → 193 维特征
```

---

## 特征布局

`extract_one()` 的拼接顺序，与 `imu_train` 的 `_extract_one()` 完全一致：

| 顺序 | 特征组 | 维度（8 通道） | 维度（6 通道） |
|------|--------|---------------|---------------|
| 1 | 时域，逐通道 | 8 × 11 = 88 | 6 × 11 = 66 |
| 2 | 频域，仅前 6 通道 | 6 × 8 = 48 | 6 × 8 = 48 |
| 3 | 全局跨通道 | 8 | 8 |
| 4 | 模长（acc / gyro） | 38 | 38 |
| 5 | Jerk 模长 | 11 | 11 |
| | **合计** | **193** | **171** |

姿态角通道不参与频域特征 —— 它不是振荡信号，做 PSD 没有意义。

### 时域 11 项（`_time_stats_1d`）

| 偏移 | 特征 | 说明 |
|------|------|------|
| +0 | mean | 均值 |
| +1 | std | 标准差 |
| +2 | min | 最小值 |
| +3 | max | 最大值 |
| +4 | range | 极差（max − min） |
| +5 | rms | 均方根 √(mean(x²)) |
| +6 | skew | 偏度（std ≤ 1e-8 时钳为 0） |
| +7 | kurtosis | 峰度（同上） |
| +8 | **mcr** | **均值**穿越率 —— 穿越窗口自身均值的次数 |
| +9 | iqr | 四分位距（Q3 − Q1），比 std 抗突发尖峰 |
| +10 | peak_count | 局部极值个数，直接量化"动了几次" |

> mcr 统计的是穿越**均值**而非绝对 0：重力对齐后 acc_z 恒有直流偏置，
> 穿越绝对 0 会恒为 0，没有区分度。peak_count 与 mcr 不重复 —— 波形不对称时两者不等价。

### 频域 8 项（`_freq_stats_1d`，Welch PSD，`nperseg=min(len(x), 32)`）

| 偏移 | 特征 |
|------|------|
| +0 | 频谱均值 |
| +1 | 频谱标准差 |
| +2 | 主频 |
| +3 | 频谱熵 |
| +4…+7 | 4 个分频段能量占比，频段按 Nyquist 比例切分：`[0, .125) [.125, .375) [.375, .75) [.75, 1)` |

> 分频段用 Nyquist 的**比例**而不是绝对 Hz，这样同一套特征在 20/25/50Hz 下含义一致。
> 4 段占比之和通常略小于 1 —— 最高段是半开区间，正好落在 Nyquist 上的 bin 不计入。

### 全局 8 项（`_global_features`）

acc 与 gyro 各自的 SMA（三轴绝对值之和的均值），加上各自三轴两两相关系数
（xy / yz / xz），共 2 + 3 + 3 = 8。

### 模长 38 项 / Jerk 11 项

acc 模长和 gyro 模长各取时域 11 + 频域 8 = 19，合计 38。模长旋转不变，抗项圈朝向漂移。
Jerk 是 `diff(acc) × hz` 的模长再取时域 11 项，捕捉动作的"突然性"——
平缓行走 jerk 小，甩头、抓挠回位这类瞬时动作会有明显尖峰。

---

## 特征布局自动识别

`BehaviorClassifier` 启动时读模型的 `n_features_in_`，反推该用哪套特征：

| `n_features_in_` | 判定 | 说明 |
|------------------|------|------|
| 193 | v2 / 8 通道 | 当前 imu_train 训练管线 |
| 171 | v2 / 6 通道 | 未附加姿态角的 imu_train 模型 |
| 78 | legacy | 旧版特征，仅为兼容仓库内已提交的模型，会打警告 |
| 其它 | 报错退出 | 不会静默用错特征 |

换模型时把 `ml_rf.pkl` 和 `ml_rf.json` 一起放进 `weights/` 重启即可，无需改代码。
`ml_rf.json` 的 `classes` 决定类别顺序（按中文名映射到 `BehaviorLabel`，不写死下标），
`hz` / `window_s` / `stride_s` / `gravity_aligned` 会与推理参数比对，不一致时打印修复建议。

---

## 相关代码

- 特征提取：[`features.py`](features.py)
- 重力对齐与姿态角：[`gravity.py`](gravity.py)
- 分类器封装：[`model.py`](model.py)
- 单元测试：[`tests/unit/test_inference_model.py`](../../tests/unit/test_inference_model.py)
