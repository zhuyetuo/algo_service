from enum import IntEnum
from pathlib import Path

import joblib
import numpy as np
from scipy import stats, signal

from config import settings


class BehaviorLabel(IntEnum):
    UNKNOWN  = 0
    MOVEMENT = 1
    SLEEP    = 2
    SCRATCH  = 3


# imu_train 输出标签 → BehaviorLabel 映射
# processed_custom/20hz/ml_rf.json classes 顺序：抓挠=0, 活动=1, 睡觉=2
_LABEL_MAP: dict[int, int] = {
    0: BehaviorLabel.SCRATCH,
    1: BehaviorLabel.MOVEMENT,
    2: BehaviorLabel.SLEEP,
}


# ---------------------------------------------------------------------------
# 特征提取（与 training/imu_train/src/ml/features.py 保持一致，共 78 维）
# ---------------------------------------------------------------------------

def _time_features(x: np.ndarray) -> np.ndarray:
    """单轴时域特征 9 维：均值、标准差、最小值、最大值、极差、RMS、偏度、峰度、过零率。"""
    return np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.max(x) - np.min(x),
        np.sqrt(np.mean(x ** 2)),
        stats.skew(x),
        stats.kurtosis(x),
        float(np.sum(np.diff(np.sign(x)) != 0)),
    ])


def _freq_features(x: np.ndarray, fs: int) -> np.ndarray:
    """单轴频域特征 4 维（Welch PSD）：频谱均值、频谱标准差、主频、频谱熵。"""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), 32))
    psd_norm = psd / (psd.sum() + 1e-8)
    spectral_mean = np.sum(freqs * psd_norm)
    spectral_std  = np.sqrt(np.sum((freqs - spectral_mean) ** 2 * psd_norm))
    dominant_freq = freqs[np.argmax(psd)]
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-8))
    return np.array([spectral_mean, spectral_std, dominant_freq, spectral_entropy])


def extract_features(window: np.ndarray, fs: int) -> np.ndarray:
    """
    window : (n_samples, 6) float32，列顺序为 [ax, ay, az, gx, gy, gz]
    返回：78 维特征向量（6 轴 × 13 维），与 imu_train 特征空间一致。
    """
    parts = []
    for i in range(6):
        col = window[:, i]
        parts.append(_time_features(col))   # 9 维
        parts.append(_freq_features(col, fs))  # 4 维
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 滑动窗口分段
# ---------------------------------------------------------------------------

def segment(data: np.ndarray, window_samples: int, step_samples: int) -> list[np.ndarray]:
    """对数据进行滑动窗口分段，返回窗口列表。"""
    windows, start = [], 0
    while start + window_samples <= len(data):
        windows.append(data[start: start + window_samples])
        start += step_samples
    return windows


# ---------------------------------------------------------------------------
# 行为事件合并：将连续相同标签的窗口合并为一个事件
# ---------------------------------------------------------------------------

def windows_to_events(
    labels: np.ndarray,
    confidences: np.ndarray,
    window_samples: int,
    step_samples: int,
    fs: int,
    base_ts_ms: int,
) -> list[dict]:
    """
    将逐窗口预测结果合并为行为事件。

    base_ts_ms：批次第一个采样点的 UTC 毫秒时间戳。
    返回字典列表，包含字段：behavior_type、start_time、end_time、confidence
    """
    if len(labels) == 0:
        return []

    events = []
    cur_label = int(labels[0])
    cur_conf_sum = float(confidences[0])
    cur_conf_cnt = 1
    cur_start_sample = 0

    for idx in range(1, len(labels)):
        lbl = int(labels[idx])
        if lbl == cur_label:
            # 同一标签，累加置信度
            cur_conf_sum += float(confidences[idx])
            cur_conf_cnt += 1
        else:
            # 标签切换，输出当前事件
            end_sample = idx * step_samples + window_samples
            events.append({
                "behavior_type": cur_label,
                "start_time": base_ts_ms + int(cur_start_sample / fs * 1000),
                "end_time":   base_ts_ms + int(end_sample / fs * 1000),
                "confidence": round(cur_conf_sum / cur_conf_cnt, 4),
            })
            cur_label = lbl
            cur_conf_sum = float(confidences[idx])
            cur_conf_cnt = 1
            cur_start_sample = idx * step_samples

    # 输出最后一段事件
    end_sample = (len(labels) - 1) * step_samples + window_samples
    events.append({
        "behavior_type": cur_label,
        "start_time": base_ts_ms + int(cur_start_sample / fs * 1000),
        "end_time":   base_ts_ms + int(end_sample / fs * 1000),
        "confidence": round(cur_conf_sum / cur_conf_cnt, 4),
    })
    return events


# ---------------------------------------------------------------------------
# 逐窗口标签平滑：滑动多数票，消除决策边界附近的帧间跳变
# ---------------------------------------------------------------------------

def _majority_smooth(labels: np.ndarray, k: int = 5) -> np.ndarray:
    """
    对长度为 N 的标签序列做半径 k//2 的滑动多数票平滑。
    k=5 意味着每个位置参考前后各 2 帧，5 帧里少数类被多数类覆盖。
    """
    if len(labels) < k:
        return labels
    half = k // 2
    smoothed = labels.copy()
    for i in range(len(labels)):
        window = labels[max(0, i - half): i + half + 1]
        vals, counts = np.unique(window, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return smoothed


# ---------------------------------------------------------------------------
# 模型封装
# ---------------------------------------------------------------------------

class BehaviorClassifier:
    def __init__(self):
        import logging
        logger = logging.getLogger(__name__)
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        self._model = joblib.load(path)

        self._fs   = settings.imu_sample_rate
        self._win  = int(settings.window_seconds * self._fs)
        self._step = int(self._win * (1 - settings.window_overlap))

        model_type = type(self._model).__name__
        logger.info(
            "行为分类器已加载 model=%s type=%s fs=%dHz window=%.1fs step=%ds",
            path, model_type, self._fs, settings.window_seconds, self._step // self._fs,
        )

    def predict(
        self,
        data: np.ndarray,
        base_ts_ms: int,
    ) -> list[dict]:
        """
        data       : (N, 6) float32，按时间顺序排列的 IMU 采样数据
        base_ts_ms : data[0] 对应的 UTC 毫秒时间戳

        返回行为事件列表（参见 windows_to_events）。
        """
        windows = segment(data, self._win, self._step)
        if not windows:
            return []

        X = np.stack([extract_features(w, self._fs) for w in windows])
        raw_labels = self._model.predict(X)

        # imu_train 输出 0/1/2，映射到 BehaviorLabel（SLEEP=2/MOVEMENT=1/SCRATCH=3）
        labels = np.array([_LABEL_MAP.get(int(l), BehaviorLabel.UNKNOWN) for l in raw_labels])

        # predict_proba 返回每个类别的概率分布，最大值即为置信度
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            confidences = proba.max(axis=1)
        else:
            confidences = np.ones(len(labels))

        # 滑动多数票平滑（窗口 5），消除模型在决策边界附近的逐帧跳变
        labels = _majority_smooth(labels, k=5)

        return windows_to_events(
            labels, confidences, self._win, self._step, self._fs, base_ts_ms
        )


# 单例模式，启动时加载一次
_classifier: BehaviorClassifier | None = None


def get_classifier() -> BehaviorClassifier:
    """获取全局分类器单例，首次调用时加载模型。"""
    global _classifier
    if _classifier is None:
        _classifier = BehaviorClassifier()
    return _classifier
