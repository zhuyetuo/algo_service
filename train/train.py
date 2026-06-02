"""
行为分类模型训练脚本。

使用合成 IMU 数据训练 LightGBM，生成 weights/behavior_lgbm.pkl。
特征提取与线上推理共用同一套逻辑（modules/inference/model.py）。

用法：
    cd algo_service
    python train/train.py

首次运行约 20–25 秒，生成合成数据并训练。
再次运行直接读缓存 CSV，约 5 秒完成。
删除 train/data/ 目录可强制重新生成数据。

训练场景（合成数据）：
    S1_Normal      运动 55% / 睡眠 40% / 抓挠  5%
    S2_Active      运动 72% / 睡眠 25% / 抓挠  3%
    S3_Calm        运动 20% / 睡眠 77% / 抓挠  3%
"""

import sys
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# 确保能 import 项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.inference.model import extract_features, BehaviorLabel

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

TRAIN_DIR   = Path(__file__).parent
DATA_DIR    = TRAIN_DIR / "data"
WEIGHTS_DIR = TRAIN_DIR.parent / "weights"
MODEL_PATH  = WEIGHTS_DIR / "behavior_lgbm.pkl"

DATA_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FS          = 50          # 采样率 Hz
WIN_SAMPLES = 3 * FS      # 窗口长度 150 个采样点
N_DAYS      = 180         # 训练天数
POOL_SIZE   = 2000        # 每类合成窗口数
RNG         = np.random.default_rng(42)

CLASS_NAMES = {
    int(BehaviorLabel.MOVEMENT): "movement",
    int(BehaviorLabel.SLEEP):    "sleep",
    int(BehaviorLabel.SCRATCH):  "scratch",
}

# 三个训练场景的每日窗口分布
SCENARIOS = {
    "S1_Normal": {BehaviorLabel.MOVEMENT: 480, BehaviorLabel.SLEEP: 350, BehaviorLabel.SCRATCH: 25},
    "S2_Active": {BehaviorLabel.MOVEMENT: 620, BehaviorLabel.SLEEP: 220, BehaviorLabel.SCRATCH: 15},
    "S3_Calm":   {BehaviorLabel.MOVEMENT: 175, BehaviorLabel.SLEEP: 670, BehaviorLabel.SCRATCH: 15},
}

# ---------------------------------------------------------------------------
# IMU 信号合成
# ---------------------------------------------------------------------------

def _movement_window() -> np.ndarray:
    t    = np.arange(WIN_SAMPLES) / FS
    freq = RNG.uniform(1.5, 2.5)
    amp  = RNG.uniform(0.4, 0.8)
    phi  = RNG.uniform(0, 2 * np.pi)
    ax = amp * np.sin(2 * np.pi * freq * t + phi)           + RNG.normal(0, 0.05, WIN_SAMPLES)
    ay = amp * 0.6 * np.sin(2 * np.pi * freq * t + phi+0.5) + RNG.normal(0, 0.04, WIN_SAMPLES)
    az = 9.8 + amp * 0.5 * np.sin(2 * np.pi * freq * t)    + RNG.normal(0, 0.06, WIN_SAMPLES)
    gx = amp * 0.5 * np.sin(2 * np.pi * freq * t)           + RNG.normal(0, 0.03, WIN_SAMPLES)
    gy = amp * 0.4 * np.cos(2 * np.pi * freq * t)           + RNG.normal(0, 0.03, WIN_SAMPLES)
    gz = RNG.normal(0, 0.02, WIN_SAMPLES)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def _sleep_window() -> np.ndarray:
    t     = np.arange(WIN_SAMPLES) / FS
    bfreq = RNG.uniform(0.2, 0.4)
    ax = 0.02 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.008, WIN_SAMPLES)
    ay = RNG.normal(0, 0.008, WIN_SAMPLES)
    az = 9.8 + 0.03 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.008, WIN_SAMPLES)
    gx = RNG.normal(0, 0.004, WIN_SAMPLES)
    gy = RNG.normal(0, 0.004, WIN_SAMPLES)
    gz = RNG.normal(0, 0.004, WIN_SAMPLES)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def _scratch_window() -> np.ndarray:
    t    = np.arange(WIN_SAMPLES) / FS
    freq = RNG.uniform(4.0, 8.0)
    amp  = RNG.uniform(1.5, 3.0)
    dom  = RNG.integers(0, 3)
    amps = [0.25 * amp] * 3
    amps[dom] = amp
    ax = amps[0] * np.sin(2 * np.pi * freq * t)           + RNG.normal(0, 0.1, WIN_SAMPLES)
    ay = amps[1] * np.sin(2 * np.pi * freq * t + np.pi/4) + RNG.normal(0, 0.1, WIN_SAMPLES)
    az = 9.8 + amps[2] * np.sin(2 * np.pi * freq * t)    + RNG.normal(0, 0.1, WIN_SAMPLES)
    gx = 0.6 * amp * np.sin(2 * np.pi * freq * t)         + RNG.normal(0, 0.05, WIN_SAMPLES)
    gy = 0.5 * amp * np.cos(2 * np.pi * freq * t)         + RNG.normal(0, 0.05, WIN_SAMPLES)
    gz = 0.3 * amp * np.sin(2 * np.pi * freq * t+np.pi/3) + RNG.normal(0, 0.05, WIN_SAMPLES)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


_GENERATORS = {
    BehaviorLabel.MOVEMENT: _movement_window,
    BehaviorLabel.SLEEP:    _sleep_window,
    BehaviorLabel.SCRATCH:  _scratch_window,
}

# ---------------------------------------------------------------------------
# 特征池：生成或加载
# ---------------------------------------------------------------------------

_POOL_CSV = DATA_DIR / "feature_pool.csv"


def _build_pool() -> dict[BehaviorLabel, np.ndarray]:
    rows = []
    pool = {}
    for label, gen_fn in tqdm(_GENERATORS.items(), desc="生成特征池", ncols=72):
        feats = []
        for _ in tqdm(range(POOL_SIZE), desc=f"  {CLASS_NAMES[int(label)]}", ncols=72, leave=False):
            w = gen_fn()
            f = extract_features(w, FS)
            feats.append(f)
            rows.append([CLASS_NAMES[int(label)]] + f.tolist())
        pool[label] = np.array(feats, dtype=np.float32)

    n_feats   = next(iter(pool.values())).shape[1]
    feat_cols = [f"feat_{i:03d}" for i in range(n_feats)]
    pd.DataFrame(rows, columns=["label"] + feat_cols).to_csv(_POOL_CSV, index=False)
    print(f"  特征池已保存 → {_POOL_CSV}  ({len(rows):,} 行)")
    return pool


def _load_pool() -> dict[BehaviorLabel, np.ndarray]:
    df        = pd.read_csv(_POOL_CSV)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    pool = {}
    for label in _GENERATORS:
        name      = CLASS_NAMES[int(label)]
        pool[label] = df[df["label"] == name][feat_cols].values.astype(np.float32)
    return pool


def get_pool() -> dict[BehaviorLabel, np.ndarray]:
    if _POOL_CSV.exists():
        print(f"  加载特征池 {_POOL_CSV.name} …")
        return _load_pool()
    print("  未找到特征池，从头生成 …")
    return _build_pool()


# ---------------------------------------------------------------------------
# 训练数据组装
# ---------------------------------------------------------------------------

def build_training_data(pool: dict) -> tuple[np.ndarray, np.ndarray]:
    X_parts, y_parts = [], []
    for sc_name, daily in tqdm(SCENARIOS.items(), desc="组装训练数据", ncols=72):
        for _ in range(N_DAYS):
            for label, n_win in daily.items():
                n   = max(1, int(n_win * RNG.uniform(0.8, 1.2)))
                idx = RNG.integers(0, len(pool[label]), size=n)
                X_parts.append(pool[label][idx])
                y_parts.append(np.full(n, int(label), dtype=np.int32))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"  训练样本：{len(X):,}  "
          f"(运动={int((y==1).sum()):,}  睡眠={int((y==2).sum()):,}  抓挠={int((y==3).sum()):,})")
    return X, y


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  训练集 {len(X_tr):,}  验证集 {len(X_val):,}")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    y_pred = model.predict(X_val)
    print(f"\n验证集报告：")
    print(classification_report(
        y_val, y_pred,
        labels=sorted(CLASS_NAMES.keys()),
        target_names=[CLASS_NAMES[l] for l in sorted(CLASS_NAMES.keys())],
        digits=3,
    ))
    print(f"  验证集准确率：{accuracy_score(y_val, y_pred):.4f}")
    return model


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    print("=" * 55)
    print("行为分类模型训练")
    print("=" * 55)

    pool            = get_pool()
    X_train, y_train = build_training_data(pool)

    print("\n开始训练 LightGBM …")
    model = train(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n模型已保存 → {MODEL_PATH}")
    print(f"总耗时：{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
