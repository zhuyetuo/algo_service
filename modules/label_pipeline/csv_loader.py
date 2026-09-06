"""
读取 label_infra 那边导入的 IMU 原始 CSV（NAS 上，路径由 label_infra 传过来），
转成 modules.inference.model.BehaviorClassifier.predict() 要的 (N,6) 数组 +
base_ts_ms。列名约定跟 label_infra/smart-label/backend/app/services/imu_service.py
保持一致（两边各自独立解析，不共享代码——不同进程/不同仓库，没必要为这点逻辑
搭一个共享包）：

    timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

时间戳按数值量级猜单位（秒/毫秒/微秒/纳秒），跟 imu_service.py 的做法一致。
"""

import os

import numpy as np
import pandas as pd

from config import settings

_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
_TIMESTAMP_CANDIDATES = ["timestamp", "ts", "time", "time_stamp", "datetime"]


class ImuCsvError(Exception):
    pass


def _norm(name: object) -> str:
    s = str(name).replace("﻿", "").strip().lower()
    for ch in (" ", "-", ".", "/"):
        s = s.replace(ch, "_")
    return s


def _resolve_nas_path(relative_path: str) -> str:
    """relative_path 是相对 NAS_ROOT 的相对路径（跟 label_infra 数据库里存的一致）。
    不接受绝对路径/".."穿越，防止调用方传进来的路径跳出 NAS_ROOT。"""
    if os.path.isabs(relative_path) or ".." in relative_path.split(os.sep):
        raise ImuCsvError(f"非法路径（必须是 NAS_ROOT 下的相对路径）: {relative_path}")
    full_path = os.path.join(settings.nas_root, relative_path)
    if not os.path.exists(full_path):
        raise ImuCsvError(f"文件不存在: {full_path}")
    return full_path


def _epoch_to_ms(values: np.ndarray) -> np.ndarray:
    """按数值量级猜时间戳单位，统一换算成毫秒。"""
    sample = float(np.nanmedian(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0
    if sample > 1e17:
        return values / 1e6      # 纳秒 -> 毫秒
    if sample > 1e14:
        return values / 1e3      # 微秒 -> 毫秒
    if sample > 1e11:
        return values             # 已经是毫秒
    return values * 1000.0        # 秒 -> 毫秒


def load_imu_csv(relative_path: str) -> tuple[np.ndarray, int]:
    """返回 (data, base_ts_ms)：
      data: (N, 6) float32，列顺序 acc_x/y/z, gyro_x/y/z
      base_ts_ms: 第一行的 UTC 毫秒时间戳
    """
    full_path = _resolve_nas_path(relative_path)
    try:
        df = pd.read_csv(full_path)
    except Exception as e:
        raise ImuCsvError(f"CSV 读取失败: {e}") from e

    df.columns = [_norm(c) for c in df.columns]

    missing = [c for c in _CHANNELS if c not in df.columns]
    if missing:
        raise ImuCsvError(f"CSV 缺少必需列 {missing}，现有列: {list(df.columns)}")

    ts_col = next((c for c in _TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ImuCsvError(f"CSV 缺少时间戳列（候选名: {_TIMESTAMP_CANDIDATES}）")

    for c in _CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=_CHANNELS)
    if df.empty:
        raise ImuCsvError("CSV 里没有任何一行 6 个通道都是有效数字")

    ts_raw = pd.to_numeric(df[ts_col], errors="coerce")
    if ts_raw.isna().any():
        # 时间戳列不是纯数字（可能是日期字符串），尝试按日期时间解析
        parsed = pd.to_datetime(df[ts_col], errors="coerce")
        if parsed.isna().any():
            raise ImuCsvError(f"时间戳列 {ts_col} 既不是数字 epoch 也不是可解析的日期时间")
        ts_ms = (parsed.astype("int64") // 1_000_000).to_numpy()
    else:
        ts_ms = _epoch_to_ms(ts_raw.to_numpy()).astype("int64")

    order = np.argsort(ts_ms)
    ts_ms = ts_ms[order]
    data = df[_CHANNELS].to_numpy(dtype=np.float32)[order]

    return data, int(ts_ms[0])
