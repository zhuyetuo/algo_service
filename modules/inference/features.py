"""
手工特征提取 —— 与 imu_train/src/ml/features.py 逐行对齐。

⚠️  改动本文件前请先同步 imu_train，任何与训练侧不一致的改动都会让推理结果
    偏离训练分布（特征顺序、维度、统计量定义都必须完全一致）。

输入通道约定（固定顺序）：
  0:3  acc_x, acc_y, acc_z   （重力对齐后）
  3:6  gyr_x, gyr_y, gyr_z   （重力对齐后）
  6:8  pitch, roll           （原始未对齐姿态角，由 gravity.raw_tilt 计算）

特征维度：
  6 通道 → 171 维    8 通道 → 193 维
"""

import numpy as np
from scipy import stats, signal
from scipy.signal import find_peaks

# 频段边界，按 Nyquist 频率的比例给出，兼容不同采样率
FREQ_BANDS = [(0.0, 0.125), (0.125, 0.375), (0.375, 0.75), (0.75, 1.0)]


def _time_stats_1d(x: np.ndarray) -> list:
    """单个一维信号的 11 个时域统计量，供逐通道特征和模长/Jerk 特征共用。"""
    std = np.std(x)
    q1, q3 = np.percentile(x, [25, 75])
    x_centered = x - np.mean(x)
    return [
        np.mean(x),
        std,
        np.min(x),
        np.max(x),
        np.max(x) - np.min(x),                       # 极差
        np.sqrt(np.mean(x ** 2)),                    # RMS
        stats.skew(x) if std > 1e-8 else 0.0,
        stats.kurtosis(x) if std > 1e-8 else 0.0,
        np.sum(np.diff(np.sign(x_centered)) != 0),   # 均值穿越率(mcr)：穿越窗口自身均值
                                                     # 的次数，而非绝对 0
        q3 - q1,                                     # IQR，比 std 更抗突发尖峰噪声
        len(find_peaks(x)[0]),                       # 局部极值(峰值)计数
    ]


def _freq_stats_1d(x: np.ndarray, hz: int) -> list:
    """单个一维信号的频域统计量：4 个统计量 + 4 个分频段能量占比。"""
    freqs, psd = signal.welch(x, fs=hz, nperseg=min(len(x), 32))
    psd_norm = psd / (psd.sum() + 1e-8)
    spec_mean = np.sum(freqs * psd_norm)
    feats = [
        spec_mean,                                                # 频谱均值
        np.sqrt(np.sum((freqs - spec_mean) ** 2 * psd_norm)),     # 频谱标准差
        freqs[np.argmax(psd)],                                    # 主频
        -np.sum(psd_norm * np.log(psd_norm + 1e-8)),              # 频谱熵
    ]
    nyq = hz / 2.0
    for lo_frac, hi_frac in FREQ_BANDS:
        mask = (freqs >= lo_frac * nyq) & (freqs < hi_frac * nyq)
        feats.append(float(psd_norm[mask].sum()))                 # 分频段能量占比
    return feats


def _time_features(window: np.ndarray) -> np.ndarray:
    """window: (window_size, n_channels) → 1D 特征向量，逐通道提取。"""
    feats = []
    for ch in range(window.shape[1]):
        feats.extend(_time_stats_1d(window[:, ch]))
    return np.array(feats, dtype=np.float32)


def _freq_features(window: np.ndarray, hz: int, n_ch: int) -> np.ndarray:
    """频域特征：只对前 n_ch 个通道（acc+gyro）提取，姿态角通道跳过（非振荡信号）。"""
    feats = []
    for ch in range(min(n_ch, window.shape[1])):
        feats.extend(_freq_stats_1d(window[:, ch], hz))
    return np.array(feats, dtype=np.float32)


def _sma(triplet: np.ndarray) -> float:
    """信号幅值面积：三轴绝对值之和的均值，衡量整体运动能量（不区分方向）。"""
    return float(np.mean(np.sum(np.abs(triplet), axis=1)))


def _cross_axis_corr(triplet: np.ndarray) -> list:
    """三轴两两相关系数，捕捉不同轴之间的协同运动模式。"""
    feats = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        xi, xj = triplet[:, i], triplet[:, j]
        if np.std(xi) > 1e-8 and np.std(xj) > 1e-8:
            c = float(np.corrcoef(xi, xj)[0, 1])
            c = 0.0 if np.isnan(c) else c
        else:
            c = 0.0
        feats.append(c)
    return feats


def _global_features(window: np.ndarray) -> np.ndarray:
    """跨通道的全局特征：SMA + 三轴相关系数，加速度与角速度各一份。"""
    feats = []
    acc  = window[:, 0:3]
    gyro = window[:, 3:6]
    feats.append(_sma(acc))
    feats.append(_sma(gyro))
    feats.extend(_cross_axis_corr(acc))
    feats.extend(_cross_axis_corr(gyro))
    return np.array(feats, dtype=np.float32)


def _magnitude(triplet: np.ndarray) -> np.ndarray:
    """三轴合成模长 sqrt(x²+y²+z²)，旋转不变，抗项圈朝向漂移。"""
    return np.sqrt(np.sum(triplet ** 2, axis=1))


def _magnitude_features(window: np.ndarray, hz: int) -> np.ndarray:
    """acc 模长、gyro 模长各自的时域+频域特征（各 11+8=19 维）。"""
    feats = []
    for triplet in (window[:, 0:3], window[:, 3:6]):
        mag = _magnitude(triplet)
        feats.extend(_time_stats_1d(mag))
        feats.extend(_freq_stats_1d(mag, hz))
    return np.array(feats, dtype=np.float32)


def _jerk_features(window: np.ndarray, hz: int) -> np.ndarray:
    """加速度的加加速度（Jerk）模长的时域统计量，捕捉动作的"突然性"。"""
    acc = window[:, 0:3]
    jerk = np.diff(acc, axis=0) * hz  # 近似导数：Δacc / Δt
    jerk_mag = _magnitude(jerk)
    return np.array(_time_stats_1d(jerk_mag), dtype=np.float32)


def extract_one(window: np.ndarray, hz: int) -> np.ndarray:
    """
    单窗口特征提取。拼接顺序必须与 imu_train 完全一致：
      时域(全部通道) → 频域(仅前6通道) → 全局 → 模长 → jerk
    """
    parts = [_time_features(window), _freq_features(window, hz, n_ch=6)]
    if window.shape[1] >= 6:
        parts.append(_global_features(window))
        parts.append(_magnitude_features(window, hz))
        parts.append(_jerk_features(window, hz))
    return np.concatenate(parts)


def extract_features(X: np.ndarray, hz: int) -> np.ndarray:
    """X: (N, window_size, n_channels) → (N, n_features)。"""
    if len(X) == 0:
        return np.empty((0, feature_dim(X.shape[1] if X.ndim == 3 else 10,
                                        X.shape[2] if X.ndim == 3 else 6, hz)),
                        dtype=np.float32)
    return np.stack([extract_one(X[i], hz) for i in range(len(X))])


def feature_dim(window_size: int, n_channels: int, hz: int) -> int:
    """给定窗口形状，返回特征维度（用于与模型 n_features_in_ 比对）。"""
    dummy = np.zeros((window_size, n_channels), dtype=np.float32)
    return len(extract_one(dummy, hz))


CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "pitch", "roll"]

TIME_FEAT_NAMES = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis",
                   "mcr", "iqr", "peak_count"]
FREQ_FEAT_NAMES = ["spec_mean", "spec_std", "peak_freq", "spec_entropy",
                   "band_energy_0", "band_energy_1", "band_energy_2", "band_energy_3"]
GLOBAL_FEAT_NAMES = ["sma_acc", "sma_gyro",
                     "corr_acc_xy", "corr_acc_yz", "corr_acc_xz",
                     "corr_gyro_xy", "corr_gyro_yz", "corr_gyro_xz"]
MAG_FEAT_NAMES = ([f"acc_mag_{f}" for f in TIME_FEAT_NAMES]
                  + [f"acc_mag_{f}" for f in FREQ_FEAT_NAMES]
                  + [f"gyro_mag_{f}" for f in TIME_FEAT_NAMES]
                  + [f"gyro_mag_{f}" for f in FREQ_FEAT_NAMES])
JERK_FEAT_NAMES = [f"acc_jerk_mag_{f}" for f in TIME_FEAT_NAMES]


def feature_names(n_channels: int) -> list:
    """特征名列表，顺序与 extract_one 的拼接顺序完全一致。"""
    names = []
    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        names.extend(f"{ch_name}_{feat}" for feat in TIME_FEAT_NAMES)
    for ch in range(min(6, n_channels)):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        names.extend(f"{ch_name}_{feat}" for feat in FREQ_FEAT_NAMES)
    if n_channels >= 6:
        names.extend(GLOBAL_FEAT_NAMES)
        names.extend(MAG_FEAT_NAMES)
        names.extend(JERK_FEAT_NAMES)
    return names


# ---------------------------------------------------------------------------
# 旧版 78 维特征（兼容仓库中已提交的 weights/ml_rf.pkl）
# 新模型训练完成后可整段删除
# ---------------------------------------------------------------------------

def _legacy_time_features(x: np.ndarray) -> np.ndarray:
    return np.array([
        np.mean(x), np.std(x), np.min(x), np.max(x),
        np.max(x) - np.min(x), np.sqrt(np.mean(x ** 2)),
        stats.skew(x), stats.kurtosis(x),
        float(np.sum(np.diff(np.sign(x)) != 0)),
    ])


def _legacy_freq_features(x: np.ndarray, fs: int) -> np.ndarray:
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 32))
    psd_norm = psd / (psd.sum() + 1e-8)
    spectral_mean = np.sum(freqs * psd_norm)
    spectral_std  = np.sqrt(np.sum((freqs - spectral_mean) ** 2 * psd_norm))
    return np.array([
        spectral_mean, spectral_std,
        freqs[np.argmax(psd)],
        -np.sum(psd_norm * np.log(psd_norm + 1e-8)),
    ])


def extract_one_legacy(window: np.ndarray, fs: int) -> np.ndarray:
    """旧版 78 维：先全部 6 轴时域（54），再全部 6 轴频域（24）。"""
    time_parts, freq_parts = [], []
    for i in range(6):
        col = window[:, i]
        time_parts.append(_legacy_time_features(col))
        freq_parts.append(_legacy_freq_features(col, fs))
    return np.concatenate(time_parts + freq_parts)
