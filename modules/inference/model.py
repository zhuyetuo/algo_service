import pickle
from enum import IntEnum
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq

from config import settings


class BehaviorLabel(IntEnum):
    UNKNOWN  = 0
    MOVEMENT = 1
    SLEEP    = 2
    SCRATCH  = 3


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

def _time_features(x: np.ndarray) -> np.ndarray:
    """提取单轴时域特征：均值、标准差、最小值、最大值、极差、RMS、过零率、偏度、峰度。"""
    rms = np.sqrt(np.mean(x ** 2))
    return np.array([
        x.mean(),
        x.std(),
        x.min(),
        x.max(),
        x.max() - x.min(),
        rms,
        float(np.sum(np.diff(np.sign(x)) != 0)),
        stats.skew(x),
        stats.kurtosis(x),
    ])


def _freq_features(x: np.ndarray, fs: int) -> np.ndarray:
    """提取单轴频域特征：主频、能量、谱熵。"""
    spectrum = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0 / fs)
    if spectrum.sum() == 0:
        return np.zeros(3)
    dominant = freqs[np.argmax(spectrum)]
    energy = float(np.sum(spectrum ** 2))
    prob = spectrum / spectrum.sum()
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
    return np.array([dominant, energy, entropy])


def extract_features(window: np.ndarray, fs: int) -> np.ndarray:
    """
    window : (n_samples, 6) float32，列顺序为 [ax, ay, az, gx, gy, gz]
    返回：一维特征向量（约 93 维）
    """
    parts = []
    # 对六轴分别提取时域和频域特征
    for i in range(6):
        col = window[:, i]
        parts.append(_time_features(col))
        parts.append(_freq_features(col, fs))

    # 加速度合力特征
    acc_mag = np.linalg.norm(window[:, :3], axis=1)
    parts.append(_time_features(acc_mag))

    # 角速度合力特征
    gyr_mag = np.linalg.norm(window[:, 3:], axis=1)
    parts.append(_time_features(gyr_mag))

    # 加速度轴间相关系数
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
        parts.append(np.array([corr]))

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
# 模型封装
# ---------------------------------------------------------------------------

class BehaviorClassifier:
    def __init__(self):
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")
        with open(path, "rb") as f:
            self._model = pickle.load(f)

        self._fs   = settings.imu_sample_rate
        self._win  = int(settings.window_seconds * self._fs)
        self._step = int(self._win * (1 - settings.window_overlap))

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
        labels = self._model.predict(X)

        # predict_proba 返回每个类别的概率分布，最大值即为置信度
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            confidences = proba.max(axis=1)
        else:
            confidences = np.ones(len(labels))

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
