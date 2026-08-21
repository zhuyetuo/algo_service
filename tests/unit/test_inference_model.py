import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pytest

from modules.inference import features as F
from modules.inference.gravity import gravity_align, raw_tilt, prepare_windows
from modules.inference.model import (
    _majority_smooth,
    segment,
    windows_to_events,
)

FS = 25
WIN = 50  # 2s * 25Hz —— 与 imu_train 训练参数一致


def _rand_window(n_ch=6, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((WIN, n_ch)).astype(np.float32)
    w[:, 2] += 9.8  # 让 acc_z 带上重力，更接近真实数据
    return w


# ---------------------------------------------------------------------------
# 特征提取：维度与顺序必须与 imu_train/src/ml/features.py 完全一致
# ---------------------------------------------------------------------------

class TestTimeStats:
    def test_output_length(self):
        assert len(F._time_stats_1d(np.random.randn(WIN))) == 11

    def test_constant_array(self):
        r = F._time_stats_1d(np.ones(WIN) * 5.0)
        assert r[0] == pytest.approx(5.0)   # mean
        assert r[1] == pytest.approx(0.0)   # std
        assert r[2] == pytest.approx(5.0)   # min
        assert r[3] == pytest.approx(5.0)   # max
        assert r[4] == pytest.approx(0.0)   # range
        assert r[5] == pytest.approx(5.0)   # rms
        assert r[6] == 0.0                  # skew，std≈0 时被钳为 0
        assert r[7] == 0.0                  # kurtosis 同上

    def test_symmetric_skew(self):
        r = F._time_stats_1d(np.linspace(-1.0, 1.0, WIN))
        assert abs(r[6]) < 0.01             # 对称信号偏度≈0

    def test_mean_crossing_not_zero_crossing(self):
        """mcr 统计的是穿越窗口自身均值的次数，带直流偏置时不应退化为 0。"""
        x = np.tile([1.0, -1.0], WIN // 2) + 100.0   # 整体抬高 100
        assert F._time_stats_1d(x)[8] == WIN - 1


class TestFreqStats:
    def test_output_length(self):
        assert len(F._freq_stats_1d(np.random.randn(WIN), FS)) == 8

    def test_dominant_freq(self):
        t = np.arange(WIN) / FS
        x = np.sin(2 * np.pi * 5.0 * t)
        assert F._freq_stats_1d(x, FS)[2] == pytest.approx(5.0, abs=1.5)

    def test_band_energy_is_normalized_share(self):
        """4 个频段是能量占比：各自非负、合计不超过 1。

        合计通常略小于 1 —— 最高频段是半开区间 [0.75·nyq, nyq)，正好落在
        Nyquist 频率上的那个 bin 不计入，这与 imu_train 的实现一致。
        """
        bands = F._freq_stats_1d(np.random.randn(WIN), FS)[4:8]
        assert all(b >= 0 for b in bands)
        assert 0.9 <= sum(bands) <= 1.0 + 1e-6

    def test_band_energy_follows_signal_frequency(self):
        """低频正弦的能量应集中在低频段，高频正弦集中在高频段。"""
        t = np.arange(WIN) / FS
        low  = F._freq_stats_1d(np.sin(2 * np.pi * 1.0 * t), FS)[4:8]
        high = F._freq_stats_1d(np.sin(2 * np.pi * 11.0 * t), FS)[4:8]
        assert np.argmax(low) < np.argmax(high)


class TestFeatureDim:
    @pytest.mark.parametrize("n_ch,expected", [(6, 171), (8, 193)])
    def test_dim(self, n_ch, expected):
        assert F.feature_dim(WIN, n_ch, FS) == expected
        assert len(F.extract_one(_rand_window(n_ch), FS)) == expected
        assert len(F.feature_names(n_ch)) == expected

    def test_batch_shape(self):
        X = np.stack([_rand_window(8, s) for s in range(4)])
        assert F.extract_features(X, FS).shape == (4, 193)

    def test_legacy_dim(self):
        assert len(F.extract_one_legacy(_rand_window(6), FS)) == 78

    def test_deterministic(self):
        w = _rand_window(8, seed=3)
        assert np.array_equal(F.extract_one(w, FS), F.extract_one(w, FS))


# ---------------------------------------------------------------------------
# 重力对齐 / 姿态角
# ---------------------------------------------------------------------------

class TestGravityAlign:
    def test_gravity_points_to_z(self):
        """任意朝向的静止窗口，对齐后平均加速度都应指向 +Z。"""
        w = np.zeros((WIN, 6), dtype=np.float32)
        w[:, 0] = 9.8  # 重力全在 X 轴上
        out = gravity_align(w)
        g = out[:, :3].mean(axis=0)
        assert g[2] == pytest.approx(9.8, abs=1e-3)
        assert abs(g[0]) < 1e-3 and abs(g[1]) < 1e-3

    def test_norm_preserved(self):
        """旋转是正交变换，每个采样点的模长必须不变。"""
        w = _rand_window(6, seed=5)
        out = gravity_align(w)
        assert np.allclose(np.linalg.norm(w[:, :3], axis=1),
                           np.linalg.norm(out[:, :3], axis=1), atol=1e-4)

    def test_already_aligned_is_noop(self):
        w = np.zeros((WIN, 6), dtype=np.float32)
        w[:, 2] = 9.8
        assert np.allclose(gravity_align(w), w)

    def test_zero_signal_passthrough(self):
        w = np.zeros((WIN, 6), dtype=np.float32)
        assert np.allclose(gravity_align(w), w)


class TestRawTilt:
    def test_shape(self):
        assert raw_tilt(_rand_window(6)[:, :3]).shape == (WIN, 2)

    def test_level_is_zero_tilt(self):
        acc = np.zeros((WIN, 3), dtype=np.float32)
        acc[:, 2] = 9.8
        assert np.allclose(raw_tilt(acc), 0.0, atol=1e-6)


class TestPrepareWindows:
    def test_appends_tilt_channels(self):
        X = np.stack([_rand_window(6, s) for s in range(3)])
        assert prepare_windows(X, with_tilt=True).shape == (3, WIN, 8)
        assert prepare_windows(X, with_tilt=False).shape == (3, WIN, 6)

    def test_tilt_computed_before_alignment(self):
        """姿态角必须来自原始 acc；若在对齐后计算，倾斜窗口的 tilt 会被抹成 0。"""
        w = np.zeros((WIN, 6), dtype=np.float32)
        w[:, 0] = 9.8  # 完全侧躺
        tilt = prepare_windows(w[None], with_tilt=True)[0, :, 6:8]
        assert abs(tilt[:, 0]).mean() > 1.0  # pitch 应接近 -90°，绝不是 0


# ---------------------------------------------------------------------------
# 滑动窗口 / 事件合并 / 平滑
# ---------------------------------------------------------------------------

class TestSegment:
    def test_exact_fit(self):
        assert len(segment(np.random.randn(WIN, 6), WIN, WIN // 2)) == 1

    def test_multiple_windows(self):
        assert len(segment(np.random.randn(100, 6), WIN, WIN // 2)) == 3

    def test_empty_when_too_short(self):
        assert segment(np.random.randn(WIN - 1, 6), WIN, WIN // 2) == []

    def test_window_shapes(self):
        for w in segment(np.random.randn(100, 6), WIN, WIN // 2):
            assert w.shape == (WIN, 6)


class TestMajoritySmooth:
    def test_removes_isolated_flip(self):
        labels = np.array([2, 2, 2, 1, 2, 2, 2])
        assert np.array_equal(_majority_smooth(labels, k=5), np.full(7, 2))

    def test_keeps_real_transition(self):
        labels = np.array([2] * 6 + [1] * 6)
        out = _majority_smooth(labels, k=5)
        assert out[0] == 2 and out[-1] == 1

    def test_k1_is_noop(self):
        labels = np.array([2, 1, 2, 1])
        assert np.array_equal(_majority_smooth(labels, k=1), labels)

    def test_short_input_untouched(self):
        labels = np.array([2, 1])
        assert np.array_equal(_majority_smooth(labels, k=5), labels)


class TestWindowsToEvents:
    def test_empty_labels(self):
        assert windows_to_events(np.array([], dtype=int), np.array([]),
                                 WIN, 25, FS, 1000) == []

    def test_single_label(self):
        events = windows_to_events(np.array([1, 1, 1]), np.array([0.9, 0.9, 0.9]),
                                   WIN, 25, FS, 1000)
        assert len(events) == 1
        assert events[0]["start_time"] == 1000
        assert events[0]["confidence"] == pytest.approx(0.9, abs=1e-4)

    def test_two_labels(self):
        events = windows_to_events(np.array([1, 1, 3, 3]), np.array([0.8, 0.8, 0.7, 0.7]),
                                   WIN, 25, FS, 1000)
        assert [e["behavior_type"] for e in events] == [1, 3]

    def test_all_different(self):
        assert len(windows_to_events(np.array([1, 2, 3]), np.array([0.9, 0.8, 0.7]),
                                     WIN, 25, FS, 1000)) == 3

    def test_timestamps_increase(self):
        events = windows_to_events(np.array([1, 2, 3, 1, 2]),
                                   np.array([0.9, 0.8, 0.7, 0.6, 0.5]), WIN, 25, FS, 0)
        starts = [e["start_time"] for e in events]
        assert all(starts[i] > starts[i - 1] for i in range(1, len(starts)))
        assert all(e["end_time"] > e["start_time"] for e in events)


# ---------------------------------------------------------------------------
# 量纲统一
# ---------------------------------------------------------------------------

from modules.inference import units as U


class TestResolveScales:
    def test_identity_when_units_match(self):
        assert U.resolve_scales("ms2", "dps", "ms2", "dps") == (1.0, 1.0)

    def test_rads_to_dps(self):
        _, gyro = U.resolve_scales("ms2", "rads", "ms2", "dps")
        assert gyro == pytest.approx(57.29577951, rel=1e-9)

    def test_g_to_ms2(self):
        acc, _ = U.resolve_scales("g", "dps", "ms2", "dps")
        assert acc == pytest.approx(9.80665, rel=1e-9)

    def test_ms2_to_g(self):
        acc, _ = U.resolve_scales("ms2", "dps", "g", "dps")
        assert acc == pytest.approx(1 / 9.80665, rel=1e-9)

    def test_roundtrip_is_identity(self):
        a1, g1 = U.resolve_scales("ms2", "rads", "g", "dps")
        a2, g2 = U.resolve_scales("g", "dps", "ms2", "rads")
        assert a1 * a2 == pytest.approx(1.0, rel=1e-9)
        assert g1 * g2 == pytest.approx(1.0, rel=1e-9)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="加速度"):
            U.resolve_scales("mps2", "dps", "ms2", "dps")
        with pytest.raises(ValueError, match="角速度"):
            U.resolve_scales("ms2", "degps", "ms2", "dps")

    def test_case_insensitive(self):
        assert U.resolve_scales("MS2", "DPS", "ms2", "dps") == (1.0, 1.0)


class TestApplyScales:
    def test_noop_returns_same_object(self):
        d = np.ones((10, 6), dtype=np.float32)
        assert U.apply_scales(d, 1.0, 1.0) is d

    def test_scales_acc_and_gyro_separately(self):
        d = np.ones((4, 6), dtype=np.float32)
        out = U.apply_scales(d, 2.0, 3.0)
        assert np.all(out[:, 0:3] == 2.0)
        assert np.all(out[:, 3:6] == 3.0)

    def test_does_not_mutate_input(self):
        d = np.ones((4, 6), dtype=np.float32)
        U.apply_scales(d, 2.0, 3.0)
        assert np.all(d == 1.0)


class TestDiagnose:
    def _static(self, g_value):
        d = np.zeros((100, 6), dtype=np.float32)
        d[:, 2] = g_value
        return d

    def test_detects_ms2(self):
        assert U.diagnose(self._static(9.8))["acc_unit_guess"] == "ms2"

    def test_detects_g(self):
        assert U.diagnose(self._static(1.0))["acc_unit_guess"] == "g"

    def test_unknown_when_out_of_range(self):
        r = U.diagnose(self._static(400.0))
        assert r["acc_unit_guess"] is None
        assert r["acc_confidence"] == "低"

    def test_describe_flags_mismatch(self):
        text = " ".join(U.describe(U.diagnose(self._static(1.0)), "ms2", "dps"))
        assert "对不上" in text and "IMU_DEVICE_ACC_UNIT" in text

    def test_describe_confirms_match(self):
        text = " ".join(U.describe(U.diagnose(self._static(9.8)), "ms2", "dps"))
        assert "✓" in text
