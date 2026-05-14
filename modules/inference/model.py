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
# Feature extraction
# ---------------------------------------------------------------------------

def _time_features(x: np.ndarray) -> np.ndarray:
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
    window : (n_samples, 6) float32  [ax, ay, az, gx, gy, gz]
    returns: 1-D feature vector (~93 dims)
    """
    parts = []
    for i in range(6):
        col = window[:, i]
        parts.append(_time_features(col))
        parts.append(_freq_features(col, fs))

    acc_mag = np.linalg.norm(window[:, :3], axis=1)
    parts.append(_time_features(acc_mag))

    gyr_mag = np.linalg.norm(window[:, 3:], axis=1)
    parts.append(_time_features(gyr_mag))

    for i, j in [(0, 1), (0, 2), (1, 2)]:
        corr = np.corrcoef(window[:, i], window[:, j])[0, 1]
        parts.append(np.array([corr]))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Sliding window segmentation
# ---------------------------------------------------------------------------

def segment(data: np.ndarray, window_samples: int, step_samples: int) -> list[np.ndarray]:
    windows, start = [], 0
    while start + window_samples <= len(data):
        windows.append(data[start: start + window_samples])
        start += step_samples
    return windows


# ---------------------------------------------------------------------------
# Behavior event: merge consecutive windows with same label into one event
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
    Convert per-window predictions into merged behavior events.

    base_ts_ms : UTC millisecond timestamp of the first sample in the batch.
    Returns list of dicts with keys:
        behavior_type, start_time, end_time, confidence
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
            cur_conf_sum += float(confidences[idx])
            cur_conf_cnt += 1
        else:
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

    # flush last segment
    end_sample = (len(labels) - 1) * step_samples + window_samples
    events.append({
        "behavior_type": cur_label,
        "start_time": base_ts_ms + int(cur_start_sample / fs * 1000),
        "end_time":   base_ts_ms + int(end_sample / fs * 1000),
        "confidence": round(cur_conf_sum / cur_conf_cnt, 4),
    })
    return events


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class BehaviorClassifier:
    def __init__(self):
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
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
        data       : (N, 6) float32 — chronological IMU samples
        base_ts_ms : UTC ms timestamp of data[0]

        Returns list of behavior events (see windows_to_events).
        """
        windows = segment(data, self._win, self._step)
        if not windows:
            return []

        X = np.stack([extract_features(w, self._fs) for w in windows])
        labels = self._model.predict(X)

        # predict_proba gives per-class scores; max score = confidence
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)
            confidences = proba.max(axis=1)
        else:
            confidences = np.ones(len(labels))

        return windows_to_events(
            labels, confidences, self._win, self._step, self._fs, base_ts_ms
        )


# Singleton — loaded once at startup
_classifier: BehaviorClassifier | None = None


def get_classifier() -> BehaviorClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BehaviorClassifier()
    return _classifier
