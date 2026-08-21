"""
重力轴对齐与原始姿态角 —— 与 imu_train/src/data/gravity_align.py 逐行对齐。

⚠️  raw_tilt 必须在 gravity_align 之前、用**原始未对齐**的 acc 计算。
    gravity_align 会把每个窗口的平均 acc 旋转到 +Z，这会把窗口的绝对倾角
    抹成 ~0，破坏"趴着 / 坐着 / 站着"的姿态信息；姿态角要在旋转前单独取出，
    再作为额外通道拼回去。训练侧 preprocess.py 就是这么做的。
"""

import numpy as np


def gravity_align(window: np.ndarray) -> np.ndarray:
    """
    将窗口的 acc+gyr 旋转到"重力 → +Z"的标准坐标系。

    window: (T, C)，前 3 列 acc，接着 3 列 gyr，超过 6 的通道原样透传。
    """
    acc = window[:, :3]
    gyr = window[:, 3:6] if window.shape[1] >= 6 else None

    g_est = acc.mean(axis=0)
    g_norm = np.linalg.norm(g_est)
    if g_norm < 1e-6:
        return window  # 无重力信号，跳过

    g_unit = g_est / g_norm
    ref = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(g_unit, ref), -1.0, 1.0))

    if dot > 0.9999:
        return window  # 已对齐

    if dot < -0.9999:
        # 绕 X 轴翻转 180°
        R = np.diag(np.array([1.0, -1.0, -1.0]))
    else:
        axis = np.cross(g_unit, ref)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        # 罗德里格旋转公式
        K = np.array([
            [0.0,      -axis[2],  axis[1]],
            [axis[2],   0.0,     -axis[0]],
            [-axis[1],  axis[0],  0.0    ],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    out = window.copy()
    out[:, :3] = (R @ acc.T).T
    if gyr is not None:
        out[:, 3:6] = (R @ gyr.T).T
    return out


def gravity_align_batch(X: np.ndarray) -> np.ndarray:
    """对 (N, T, C) 中每个窗口做重力对齐。"""
    if len(X) == 0:
        return X
    return np.stack([gravity_align(w) for w in X])


def raw_tilt(acc: np.ndarray) -> np.ndarray:
    """
    从**原始未旋转**的加速度计算逐采样点的 pitch / roll（弧度）。

    acc: (T, 3) → 返回 (T, 2) 的 [pitch, roll]。
    """
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    roll  = np.arctan2(ay, az)
    return np.stack([pitch, roll], axis=1).astype(np.float32)


def append_raw_tilt_batch(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C>=3) 原始（对齐前）窗口批 → 末尾追加 [pitch, roll] 两个通道。"""
    if len(X) == 0:
        n_ch = X.shape[2] if X.ndim == 3 else 6
        return np.empty((0, X.shape[1] if X.ndim == 3 else 0, n_ch + 2), dtype=np.float32)
    tilt = np.stack([raw_tilt(w[:, :3]) for w in X], axis=0)
    return np.concatenate([X, tilt], axis=2).astype(np.float32)


def prepare_windows(X_raw: np.ndarray, use_gravity_align: bool = True,
                    with_tilt: bool = True) -> np.ndarray:
    """
    把原始 6 通道窗口批处理成模型输入，顺序与训练侧 preprocess.py 完全一致：
      1. 先用**原始** acc 算 pitch/roll
      2. 再对 acc+gyr 做重力对齐
      3. 最后把姿态角拼到通道末尾

    X_raw: (N, T, 6) → (N, T, 8)（with_tilt=True）或 (N, T, 6)
    """
    if len(X_raw) == 0:
        return X_raw
    tilt = append_raw_tilt_batch(X_raw)[:, :, 6:8] if with_tilt else None
    X = gravity_align_batch(X_raw) if use_gravity_align else X_raw
    if tilt is not None:
        X = np.concatenate([X, tilt], axis=2)
    return X.astype(np.float32)
