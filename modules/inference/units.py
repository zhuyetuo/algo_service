"""
IMU 量纲统一：把设备上报的数值换算成模型训练时使用的单位。

为什么必须做这件事
------------------
imu_train 训练时**不做量纲统一**（见 src/data/loader_custom.py：
"单位不限（训练时不做量纲统一）"），训练 CSV 是什么单位，模型学到的就是什么量级。
而特征里绝大多数是**有量纲**的绝对量——mean/std/min/max/range/rms/iqr、
SMA、acc/gyro 模长、jerk——单位错一个量级，这些特征就整体平移，
落在模型训练分布之外。（频谱熵、能量占比、相关系数这几个是无量纲的，不受影响，
所以单位错了往往不是"全错"，而是"置信度莫名偏低、某几类永远预测不出来"，
比全错更难查。）

已知的单位约定
--------------
训练侧（imu_train/src/data/labelstudio_to_custom.py，自采数据主管线）：
  - 加速度：统一换算到 **m/s²**（`--acc_unit g` 时自动 ×9.81）
  - 角速度：**原样透传，不做任何换算** → WitMotion 传感器原始输出即 **deg/s**

设备侧（TF 固件 → TDengine）：
  - 加速度：m/s²（静止时 |acc| ≈ 9.6~9.8，与训练侧一致 ✓）
  - 角速度：rad/s（与训练侧的 deg/s 差 57.3 倍 ⚠️）

但**不要凭这段注释就照搬**——设备固件可能改过。上线前先跑：
    python backfill/diagnose_signal.py --device-sn <SN>
用真实数据确认量级，再决定 IMU_DEVICE_GYRO_UNIT 怎么填。
"""

import numpy as np

G_TO_MS2   = 9.80665
RAD_TO_DEG = 57.29577951308232

# 各单位换算到基准单位的系数（加速度基准 m/s²，角速度基准 deg/s）
ACC_TO_BASE: dict[str, float] = {
    "ms2": 1.0,          # m/s²
    "g":   G_TO_MS2,     # 重力加速度
}
GYRO_TO_BASE: dict[str, float] = {
    "dps":  1.0,         # deg/s
    "rads": RAD_TO_DEG,  # rad/s
}

ACC_UNIT_LABEL  = {"ms2": "m/s²", "g": "g"}
GYRO_UNIT_LABEL = {"dps": "deg/s", "rads": "rad/s"}


def _factor(unit: str, table: dict[str, float], kind: str) -> float:
    key = (unit or "").strip().lower()
    if key not in table:
        raise ValueError(
            f"未知的{kind}单位 {unit!r}，可选：{', '.join(table)}"
        )
    return table[key]


def resolve_scales(device_acc_unit: str, device_gyro_unit: str,
                   model_acc_unit: str, model_gyro_unit: str) -> tuple[float, float]:
    """
    返回 (acc_scale, gyro_scale)：设备数值 × scale = 模型训练单位下的数值。

    例：设备 rad/s、模型 deg/s → gyro_scale = 57.2958
        设备 m/s²、模型 m/s²   → acc_scale  = 1.0
    """
    acc_scale  = _factor(device_acc_unit,  ACC_TO_BASE,  "加速度") \
               / _factor(model_acc_unit,   ACC_TO_BASE,  "加速度")
    gyro_scale = _factor(device_gyro_unit, GYRO_TO_BASE, "角速度") \
               / _factor(model_gyro_unit,  GYRO_TO_BASE, "角速度")
    return acc_scale, gyro_scale


def apply_scales(data: np.ndarray, acc_scale: float, gyro_scale: float) -> np.ndarray:
    """
    data: (N, 6) [ax ay az gx gy gz]，返回换算后的新数组（不原地修改）。
    两个 scale 都是 1.0 时直接返回原数组，省一次拷贝。
    """
    if acc_scale == 1.0 and gyro_scale == 1.0:
        return data
    out = np.asarray(data, dtype=np.float32).copy()
    if acc_scale != 1.0:
        out[:, 0:3] *= acc_scale
    if gyro_scale != 1.0:
        out[:, 3:6] *= gyro_scale
    return out


# ---------------------------------------------------------------------------
# 诊断：用真实数据反推单位
# ---------------------------------------------------------------------------

def diagnose(data: np.ndarray) -> dict:
    """
    data: (N, 6) 原始（未换算）IMU 数据。
    返回量级统计 + 单位推断。判断依据是静止时 |acc| 必然等于重力常数：
    中位数接近 9.8 → m/s²；接近 1.0 → g。
    """
    acc, gyro = np.asarray(data)[:, 0:3], np.asarray(data)[:, 3:6]
    amag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyro, axis=1)
    amed = float(np.median(amag))

    if 7.0 <= amed <= 12.5:
        acc_unit, acc_conf = "ms2", "高"
    elif 0.7 <= amed <= 1.3:
        acc_unit, acc_conf = "g", "高"
    else:
        acc_unit, acc_conf = None, "低"

    return {
        "n": int(len(data)),
        "acc_mag": {
            "median": amed,
            "p5":  float(np.percentile(amag, 5)),
            "p95": float(np.percentile(amag, 95)),
        },
        "gyro_mag": {
            "median": float(np.median(gmag)),
            "p95":    float(np.percentile(gmag, 95)),
            "max":    float(gmag.max()) if len(gmag) else 0.0,
        },
        "acc_unit_guess": acc_unit,
        "acc_confidence": acc_conf,
    }


def describe(diag: dict, device_acc_unit: str, device_gyro_unit: str) -> list[str]:
    """把 diagnose() 的结果翻译成给人看的结论和建议。"""
    lines = []
    amed = diag["acc_mag"]["median"]
    guess = diag["acc_unit_guess"]

    if guess is None:
        lines.append(
            f"⚠️  |acc| 中位数 {amed:.3f}，既不接近 9.8（m/s²）也不接近 1.0（g），"
            f"无法判断单位。可能是数据本身有问题（量程溢出、丢包、列错位），"
            f"先确认原始数据是否正常。"
        )
    elif guess != device_acc_unit:
        lines.append(
            f"⚠️  加速度单位对不上：实测 |acc| 中位数 {amed:.3f} → 看起来是 "
            f"{ACC_UNIT_LABEL[guess]}，但配置写的是 {ACC_UNIT_LABEL.get(device_acc_unit, device_acc_unit)}。"
            f"\n     → 把 IMU_DEVICE_ACC_UNIT 改成 {guess}"
        )
    else:
        lines.append(
            f"✓  加速度单位与配置一致：|acc| 中位数 {amed:.3f} ≈ "
            f"{ACC_UNIT_LABEL[guess]} 的重力常数"
        )

    # 角速度没有"静止时等于常数"这种锚点，只能给量级参考让人自己判断
    g95 = diag["gyro_mag"]["p95"]
    lines.append(
        f"·  角速度 |gyro| p95={g95:.3f} max={diag['gyro_mag']['max']:.3f}"
        f"（当前按 {GYRO_UNIT_LABEL.get(device_gyro_unit, device_gyro_unit)} 处理）"
    )
    lines.append(
        "   角速度没有类似重力的固定锚点，无法自动判定，按经验量级判断："
        "\n     犬只日常活动 deg/s 量级通常几十~几百，剧烈甩头可到 1000+；"
        "\n     换成 rad/s 则是零点几~几，剧烈动作十几。"
        f"\n     若上面的 p95={g95:.3f} 只有个位数而设备实际在动，多半是 rad/s，"
        "应设 IMU_DEVICE_GYRO_UNIT=rads。"
    )
    return lines
