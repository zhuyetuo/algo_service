"""推理几何：采样率、窗口长度、步长从哪来。

单独一个模块、**没有任何重依赖**，所以能不带 sklearn 直接测。
写在 BehaviorClassifier.__init__ 里的话，要验它就得先能造出一个分类器，
而那要 joblib+sklearn+一个真模型——结果就是没人测，
而这几个值算错**不会报错**，只是模型吃到错的几何、效果变差。

也不能放进 resample.py：那个文件是 imu_train 的逐字副本，
加一行都会让"副本没分家"那条检查失效。
"""

from __future__ import annotations


def needs_resample(device_hz, model_hz) -> bool:
    """要不要重采样。

    写死成 True 的话，采样率本来一致时也会白掉一个点
    （resample_training_match 没有恒等短路），不报错、也没有任何迹象。
    """
    try:
        return int(device_hz) != int(model_hz)
    except (TypeError, ValueError):
        # 取不到就当不需要——宁可不动数据，也不要按瞎猜的频率重采样
        return False


def resolve(meta: dict, device_hz: int, fallback_window_s: float,
            fallback_overlap: float) -> dict:
    """几何以**模型元数据**为准，环境变量只在元数据缺项时兜底。

    以前是反过来的：环境变量说了算，元数据只用来打个告警。结果仓库里那个
    16Hz 模型一直在吃 25Hz、50 点的窗口——而特征维度正好也是 193，
    什么都没报出来。
    """
    meta = meta or {}
    fs = int(meta.get("hz") or device_hz)
    win_s = float(meta.get("window_s") or fallback_window_s)
    stride_s = float(meta.get("stride_s")
                     or fallback_window_s * (1.0 - fallback_overlap))
    win = max(1, int(meta.get("window_size") or round(win_s * fs)))
    step = max(1, int(meta.get("stride") or round(stride_s * fs)))
    return {
        "fs": fs,
        "device_fs": int(device_hz),
        "win": win,
        "step": step,
        "need_resample": needs_resample(device_hz, fs),
    }
