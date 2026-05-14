import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq

from config import settings

# Axis order expected in the input array (columns)
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _time_features(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2))
    return np.array([
        x.mean(),
        x.std(),
        x.min(),
        x.max(),
        x.max() - x.min(),          # peak-to-peak
        rms,
        np.sum(np.diff(np.sign(x)) != 0),  # zero-crossing count
        stats.skew(x),
        stats.kurtosis(x),
    ])


def _freq_features(x: np.ndarray, fs: int) -> np.ndarray:
    spectrum = np.abs(rfft(x))
    freqs = rfftfreq(len(x), d=1.0 / fs)
    if spectrum.sum() == 0:
        return np.zeros(3)
    dominant = freqs[np.argmax(spectrum)]
    energy = np.sum(spectrum**2)
    prob = spectrum / spectrum.sum()
    entropy = -np.sum(prob * np.log(prob + 1e-12))
    return np.array([dominant, energy, entropy])


def extract_features(window: np.ndarray, fs: int) -> np.ndarray:
    """
    window: (n_samples, 6) float array — [ax, ay, az, gx, gy, gz]
    returns: 1-D feature vector
    """
    feat_parts = []

    # Per-axis time + freq features
    for i in range(6):
        col = window[:, i]
        feat_parts.append(_time_features(col))
        feat_parts.append(_freq_features(col, fs))

    # Accelerometer magnitude
    acc_mag = np.linalg.norm(window[:, :3], axis=1)
    feat_parts.append(_time_features(acc_mag))

    # Gyroscope magnitude
    gyr_mag = np.linalg.norm(window[:, 3:], axis=1)
    feat_parts.append(_time_features(gyr_mag))

    # Cross-axis correlations (acc axes)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        feat_parts.append(np.array([np.corrcoef(window[:, i], window[:, j])[0, 1]]))

    return np.concatenate(feat_parts)


# ---------------------------------------------------------------------------
# Sliding window segmentation
# ---------------------------------------------------------------------------

def segment(data: np.ndarray, window_samples: int, step_samples: int) -> list[np.ndarray]:
    """Split (N, 6) array into overlapping windows."""
    windows = []
    start = 0
    while start + window_samples <= len(data):
        windows.append(data[start : start + window_samples])
        start += step_samples
    return windows


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class BehaviorClassifier:
    def __init__(self):
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            self._model = pickle.load(f)  # expects a fitted LightGBM / sklearn Pipeline

        self._fs = settings.imu_sample_rate
        self._win = int(settings.window_seconds * self._fs)
        self._step = int(self._win * (1 - settings.window_overlap))

    def predict_batch(self, data: np.ndarray) -> dict:
        """
        data: (N, 6) float array for a full fetch interval
        returns: behavior label counts + dominant behavior
        """
        windows = segment(data, self._win, self._step)
        if not windows:
            return {"dominant": "unknown", "distribution": {}}

        X = np.stack([extract_features(w, self._fs) for w in windows])
        labels = self._model.predict(X)

        unique, counts = np.unique(labels, return_counts=True)
        distribution = {str(k): int(v) for k, v in zip(unique, counts)}
        dominant = str(unique[counts.argmax()])
        return {"dominant": dominant, "distribution": distribution, "window_count": len(windows)}


# Singleton
_classifier: BehaviorClassifier | None = None


def get_classifier() -> BehaviorClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BehaviorClassifier()
    return _classifier
