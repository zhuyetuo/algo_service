import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from modules.inference.model import (
    _time_features,
    _freq_features,
    extract_features,
    segment,
    windows_to_events,
)

FS = 50
WIN = 150  # 3s * 50Hz


class TestTimeFeatures:
    def test_output_shape(self):
        x = np.random.randn(WIN)
        result = _time_features(x)
        assert result.shape == (9,)

    def test_constant_array(self):
        x = np.ones(WIN) * 5.0
        result = _time_features(x)
        # mean=5, std=0, min=5, max=5, range=0, rms=5
        assert result[0] == pytest.approx(5.0)   # mean
        assert result[1] == pytest.approx(0.0)   # std
        assert result[2] == pytest.approx(5.0)   # min
        assert result[3] == pytest.approx(5.0)   # max
        assert result[4] == pytest.approx(0.0)   # range
        assert result[5] == pytest.approx(5.0)   # rms

    def test_symmetric_skew(self):
        x = np.linspace(-1.0, 1.0, WIN)
        result = _time_features(x)
        # Symmetric array -> skewness near 0
        skewness = result[7]
        assert abs(skewness) < 0.01


class TestFreqFeatures:
    def test_output_shape(self):
        x = np.random.randn(WIN)
        result = _freq_features(x, FS)
        assert result.shape == (3,)

    def test_zero_input(self):
        x = np.zeros(WIN)
        result = _freq_features(x, FS)
        assert np.all(result == 0.0)

    def test_dominant_freq(self):
        # Pure 5 Hz sine wave
        t = np.arange(WIN) / FS
        x = np.sin(2 * np.pi * 5.0 * t)
        result = _freq_features(x, FS)
        dominant = result[0]
        assert dominant == pytest.approx(5.0, abs=0.5)


class TestExtractFeatures:
    def test_output_is_1d(self):
        window = np.random.randn(WIN, 6).astype(np.float32)
        result = extract_features(window, FS)
        assert result.ndim == 1

    def test_feature_count(self):
        window = np.random.randn(WIN, 6).astype(np.float32)
        result = extract_features(window, FS)
        assert len(result) == 93

    def test_output_length_consistent(self):
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((WIN, 6)).astype(np.float32)
        w2 = rng.standard_normal((WIN, 6)).astype(np.float32)
        r1 = extract_features(w1, FS)
        r2 = extract_features(w2, FS)
        assert len(r1) == len(r2)


class TestSegment:
    def test_exact_fit(self):
        data = np.random.randn(WIN, 6)
        windows = segment(data, window_samples=WIN, step_samples=75)
        assert len(windows) == 1

    def test_multiple_windows(self):
        data = np.random.randn(300, 6)
        windows = segment(data, window_samples=WIN, step_samples=75)
        assert len(windows) == 3

    def test_empty_when_too_short(self):
        data = np.random.randn(100, 6)
        windows = segment(data, window_samples=WIN, step_samples=75)
        assert windows == []

    def test_window_shapes(self):
        data = np.random.randn(300, 6)
        windows = segment(data, window_samples=WIN, step_samples=75)
        for w in windows:
            assert w.shape == (WIN, 6)


class TestWindowsToEvents:
    def test_empty_labels(self):
        labels = np.array([], dtype=int)
        confidences = np.array([], dtype=float)
        events = windows_to_events(labels, confidences, WIN, 75, FS, 1000)
        assert events == []

    def test_single_label(self):
        labels = np.array([1, 1, 1])
        confidences = np.array([0.9, 0.9, 0.9])
        events = windows_to_events(labels, confidences, WIN, 75, FS, 1000)
        assert len(events) == 1
        assert events[0]["start_time"] == 1000
        assert events[0]["confidence"] == pytest.approx(0.9, abs=1e-4)

    def test_two_labels(self):
        labels = np.array([1, 1, 3, 3])
        confidences = np.array([0.8, 0.8, 0.7, 0.7])
        events = windows_to_events(labels, confidences, WIN, 75, FS, 1000)
        assert len(events) == 2
        behavior_types = [e["behavior_type"] for e in events]
        assert behavior_types == [1, 3]

    def test_all_different(self):
        labels = np.array([1, 2, 3])
        confidences = np.array([0.9, 0.8, 0.7])
        events = windows_to_events(labels, confidences, WIN, 75, FS, 1000)
        assert len(events) == 3

    def test_timestamps_increase(self):
        labels = np.array([1, 2, 3, 1, 2])
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        events = windows_to_events(labels, confidences, WIN, 75, FS, 0)
        start_times = [e["start_time"] for e in events]
        for i in range(1, len(start_times)):
            assert start_times[i] > start_times[i - 1]
        for e in events:
            assert e["end_time"] > e["start_time"]
