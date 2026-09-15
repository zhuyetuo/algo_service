# ============================================================================
# 这个文件是 imu_train/src/data/resample_training_match.py 的**逐字副本**。
#
# 为什么要有它：模型是 16Hz 数据训练的，而设备上报 25Hz。训练数据当年就是用
# 这个算法（滑动平均低通 + 线性插值）从原始流降到 16Hz 的，高频成分被滤掉了。
# 直接把 25Hz 喂给模型，它看到的是没见过的分布，频域特征的频率轴也不一样——
# 不报错，只是效果打折。
#
# 换成别的重采样算法也不行：跟 scipy.resample_poly 实测差 6~8%（见原文件注释）。
# 要跟训练一致，就得是同一套。
#
# **不要在这里改。** 要改去改 imu_train 那边再拷过来。
# tests/unit/test_resample_vendored.py 会在 imu_train 存在时逐字节比对。
# ============================================================================

"""
复刻 witmotion_imu/imu_camera_sync.py::resample_raw_imu() 的核心重采样算法
（滑动平均低通 + np.interp 线性插值），去掉了真实BLE流才需要的缺口检测/
视频时间戳对齐部分，只保留数值方法本身。

背景：raw_custom 那批16Hz训练数据，当年就是用这个算法从witmotion原始BLE流
生成的；而 infer_csv_scratch.py::downsample() 目前用的是 scipy.signal.
resample_poly，是完全不同的一套滤波/重采样算法。过去这个不一致基本没暴露，
因为训练/评估几乎都是直接读已经预先降采样好的16Hz CSV（device_hz==model_hz
时 downsample() 直接原样返回，不会真的走 resample_poly）。TF设备（50Hz）
第一次让"现场降采样"这条路径被真正用上，这个算法不一致才第一次有实际影响。

实测差异（src/eval/compare_resample_methods.py）：两种方法输出信号的RMSE
约为信号标准差的6~8%，量级不算离谱但确实存在，具体对分类准确率影响多大
需要真实数据验证，不能只凭理论判断。
"""
import numpy as np


def resample_training_match(data: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    """
    data: (N, C) 假定原始采样点严格按 source_hz 等间隔（真实BLE流是不规则的，
          这里简化成规则采样，跟 infer_csv_scratch.py 里其他重采样函数的输入假设一致）。
    返回: (M, C) 降采样后的数组。
    """
    n = len(data)
    if n == 0:
        return data
    t = np.arange(n) / source_hz * 1000.0  # ms
    step_ms = 1000.0 / target_hz
    avg_dt_ms = 1000.0 / source_hz
    window = max(1, int(round(step_ms / avg_dt_ms)))

    filtered = data.astype(float).copy()
    if window > 1:
        kernel = np.ones(window) / window
        for c in range(data.shape[1]):
            filtered[:, c] = np.convolve(data[:, c], kernel, mode="same")

    new_t = np.arange(t[0], t[-1], step_ms)
    out = np.stack([np.interp(new_t, t, filtered[:, c]) for c in range(data.shape[1])], axis=1)
    return out.astype(np.float32)
