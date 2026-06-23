"""
Test 1 — IMU Behavior Classification
=====================================
Generates synthetic IMU data for 5 dog scenarios over 180 days,
trains a LightGBM model on scenarios 1-3, then evaluates on all 5.

Data pipeline (CSV-backed, incremental)
----------------------------------------
  data/imu_pool.csv            raw IMU windows (label, window_id, sample_idx, ax-gz)
  data/features_pool.csv       extracted features (label, window_id, feat_000…feat_N)
  data/features_{sc}_train.csv scenario training features  (label, label_int, feat_…)
  data/features_{sc}_test.csv  scenario test features

First run generates and saves everything. Subsequent runs load from CSV.
Delete the data/ folder to force a full regeneration.

Behavior classes
----------------
  0  UNKNOWN   (not generated in training)
  1  MOVEMENT  walking / running
  2  SLEEP     resting / sleeping
  3  SCRATCH   scratching (target behavior)

5 dog scenarios
---------------
  S1  Normal      move 55 % / sleep 40 % / scratch  5 %
  S2  Active      move 72 % / sleep 25 % / scratch  3 %
  S3  Calm        move 20 % / sleep 77 % / scratch  3 %
  S4  Mild skin   move 45 % / sleep 40 % / scratch 15 %   (unseen at train)
  S5  Severe skin move 35 % / sleep 35 % / scratch 30 %   (unseen at train)

Usage
-----
  cd algo_service
  python tests/test_1_inference.py
"""

import sys, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
from tqdm import tqdm

from modules.inference.model import extract_features, BehaviorLabel

# ── Paths ─────────────────────────────────────────────────────────────────────
TESTS_DIR   = Path(__file__).parent
DATA_DIR    = TESTS_DIR / "data"
WEIGHTS_DIR = TESTS_DIR.parent / "weights"

IMU_POOL_CSV      = DATA_DIR / "imu_pool.csv"
FEATURES_POOL_CSV = DATA_DIR / "features_pool.csv"
MODEL_PATH        = WEIGHTS_DIR / "behavior_lgbm.pkl"

DATA_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FS           = 50
WINDOW_SEC   = 3
WIN_SAMPLES  = WINDOW_SEC * FS    # 150
N_DAYS_TRAIN = 180
N_DAYS_TEST  = 30
_POOL_SIZE   = 2000
RNG          = np.random.default_rng(42)

DAILY_WINDOWS = {
    "S1_Normal":      {BehaviorLabel.MOVEMENT: 480, BehaviorLabel.SLEEP: 350, BehaviorLabel.SCRATCH:  25},
    "S2_Active":      {BehaviorLabel.MOVEMENT: 620, BehaviorLabel.SLEEP: 220, BehaviorLabel.SCRATCH:  15},
    "S3_Calm":        {BehaviorLabel.MOVEMENT: 175, BehaviorLabel.SLEEP: 670, BehaviorLabel.SCRATCH:  15},
    "S4_Mild_skin":   {BehaviorLabel.MOVEMENT: 390, BehaviorLabel.SLEEP: 340, BehaviorLabel.SCRATCH:  75},
    "S5_Severe_skin": {BehaviorLabel.MOVEMENT: 300, BehaviorLabel.SLEEP: 300, BehaviorLabel.SCRATCH: 150},
}

TRAIN_SCENARIOS = ["S1_Normal", "S2_Active", "S3_Calm"]
TEST_SCENARIOS  = list(DAILY_WINDOWS.keys())

CLASS_NAMES = {
    int(BehaviorLabel.MOVEMENT): "movement",
    int(BehaviorLabel.SLEEP):    "sleep",
    int(BehaviorLabel.SCRATCH):  "scratch",
}
_LABEL_STR_TO_INT = {v: k for k, v in CLASS_NAMES.items()}


# ── IMU signal generators ─────────────────────────────────────────────────────
# Calibrated against real sensor data collected at 50 Hz from a pet collar device.
#
# REST reference  (88fa25b7): AX≈+0.13 AY≈+1.16 AZ≈+8.63 m/s²  GX≈0.031 GY≈0.012 GZ≈0.002 rad/s
# ACTIVITY ref    (a06d2116): AX≈-4.0  AY≈-7.2  AZ≈-4.1  m/s²  GX/GZ vary ±0.3-0.4 rad/s
# (Different AZ bias reflects the collar shifting between lying-down and active postures.)

def _movement_window() -> np.ndarray:
    """Walking / running: mid-freq stride oscillation, device tilted on active collar."""
    n = WIN_SAMPLES
    t = np.arange(n) / FS
    # Realistic mean accel when dog is active (collar hangs at ~45° angle)
    ax_m = RNG.uniform(-4.8, -3.5)
    ay_m = RNG.uniform(-7.6, -6.8)
    az_m = RNG.uniform(-4.4, -3.6)
    freq  = RNG.uniform(1.5, 2.5)   # stride frequency
    phi   = RNG.uniform(0, 2 * np.pi)
    a_amp = RNG.uniform(0.25, 0.55)  # oscillation amplitude (m/s²)
    g_amp = RNG.uniform(0.12, 0.35)  # gyro amplitude (rad/s)
    ax = ax_m + a_amp * np.sin(2 * np.pi * freq * t + phi)             + RNG.normal(0, 0.05, n)
    ay = ay_m + a_amp * 0.7 * np.sin(2 * np.pi * freq * t + phi + 0.5) + RNG.normal(0, 0.05, n)
    az = az_m + a_amp * 0.5 * np.sin(2 * np.pi * freq * t)             + RNG.normal(0, 0.06, n)
    gx = g_amp * np.sin(2 * np.pi * freq * t)                          + RNG.normal(0, 0.03, n)
    gy = g_amp * 0.6 * np.cos(2 * np.pi * freq * t)                    + RNG.normal(0, 0.025, n)
    gz = g_amp * 0.8 * np.sin(2 * np.pi * freq * t + np.pi / 3)        + RNG.normal(0, 0.03, n)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def _sleep_window() -> np.ndarray:
    """Resting / sleeping: near-static, gravity mostly on Z axis, minimal gyro."""
    n     = WIN_SAMPLES
    t     = np.arange(n) / FS
    bfreq = RNG.uniform(0.2, 0.4)   # breathing oscillation
    # Realistic mean accel when dog lies down (collar roughly flat)
    ax_b = RNG.normal(0.125, 0.010)
    ay_b = RNG.normal(1.163, 0.015)
    az_b = RNG.normal(8.635, 0.025)
    ax = ax_b + 0.005 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.008, n)
    ay = ay_b + 0.008 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.010, n)
    az = az_b + 0.015 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.020, n)
    gx = 0.031 + RNG.normal(0, 0.001, n)
    gy = 0.012 + RNG.normal(0, 0.001, n)
    gz = 0.002 + RNG.normal(0, 0.001, n)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def _scratch_window() -> np.ndarray:
    """Scratching: rest-like orientation + high-frequency limb oscillation (4-8 Hz)."""
    n     = WIN_SAMPLES
    t     = np.arange(n) / FS
    freq  = RNG.uniform(4.0, 8.0)   # scratching motion frequency
    amp   = RNG.uniform(1.0, 2.5)   # oscillation amplitude (m/s²)
    g_amp = RNG.uniform(0.3, 0.9)   # rotational component from scratching
    # Base orientation same as rest (dog is stationary, collar mostly flat)
    ax_b  = RNG.normal(0.125, 0.015)
    ay_b  = RNG.normal(1.163, 0.020)
    az_b  = RNG.normal(8.635, 0.030)
    dom   = RNG.integers(0, 3)      # dominant scratch axis
    amps_a = [0.25 * amp, 0.25 * amp, 0.25 * amp]
    amps_a[dom] = amp
    ax = ax_b + amps_a[0] * np.sin(2 * np.pi * freq * t)           + RNG.normal(0, 0.08, n)
    ay = ay_b + amps_a[1] * np.sin(2 * np.pi * freq * t + np.pi/4) + RNG.normal(0, 0.08, n)
    az = az_b + amps_a[2] * np.sin(2 * np.pi * freq * t)           + RNG.normal(0, 0.10, n)
    gx = g_amp * np.sin(2 * np.pi * freq * t)                      + RNG.normal(0, 0.05, n)
    gy = g_amp * 0.7 * np.cos(2 * np.pi * freq * t)                + RNG.normal(0, 0.04, n)
    gz = g_amp * 0.5 * np.sin(2 * np.pi * freq * t + np.pi / 3)    + RNG.normal(0, 0.04, n)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


_GENERATORS = {
    BehaviorLabel.MOVEMENT: _movement_window,
    BehaviorLabel.SLEEP:    _sleep_window,
    BehaviorLabel.SCRATCH:  _scratch_window,
}


# ── Pool: build or load ───────────────────────────────────────────────────────

def _build_pool() -> dict:
    """Generate IMU windows, save imu_pool.csv + features_pool.csv, return feature arrays."""
    imu_rows  = []
    feat_rows = []
    pool      = {}

    imu_cols  = ["label", "window_id", "sample_idx", "ax", "ay", "az", "gx", "gy", "gz"]

    for label, gen_fn in tqdm(_GENERATORS.items(), desc="Building pool", unit="class", ncols=72):
        label_name = CLASS_NAMES[int(label)]
        feats = []
        for wid in tqdm(range(_POOL_SIZE), desc=f"  {label_name}", unit="win",
                        ncols=72, leave=False):
            window = gen_fn()                           # (150, 6)
            for t_idx, sample in enumerate(window):
                imu_rows.append([label_name, wid, t_idx,
                                 sample[0], sample[1], sample[2],
                                 sample[3], sample[4], sample[5]])
            feat = extract_features(window, FS)
            feat_rows.append([label_name, wid] + feat.tolist())
            feats.append(feat)
        pool[label] = np.array(feats, dtype=np.float32)

    # Save raw IMU
    pd.DataFrame(imu_rows, columns=imu_cols).to_csv(IMU_POOL_CSV, index=False)
    tqdm.write(f"  IMU pool  → {IMU_POOL_CSV}  ({len(imu_rows):,} rows)")

    # Save features
    n_feats   = pool[BehaviorLabel.MOVEMENT].shape[1]
    feat_cols = ["label", "window_id"] + [f"feat_{i:03d}" for i in range(n_feats)]
    pd.DataFrame(feat_rows, columns=feat_cols).to_csv(FEATURES_POOL_CSV, index=False)
    tqdm.write(f"  Features  → {FEATURES_POOL_CSV}  ({len(feat_rows):,} rows)")

    return pool


def _load_pool() -> dict:
    """Load feature pool from features_pool.csv."""
    df = pd.read_csv(FEATURES_POOL_CSV)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    pool = {}
    for label in _GENERATORS:
        name = CLASS_NAMES[int(label)]
        sub  = df[df["label"] == name][feat_cols].values.astype(np.float32)
        pool[label] = sub
    return pool


_FEATURE_POOL: dict | None = None

def _get_pool() -> dict:
    global _FEATURE_POOL
    if _FEATURE_POOL is None:
        if FEATURES_POOL_CSV.exists() and IMU_POOL_CSV.exists():
            print(f"  Loading pool from {FEATURES_POOL_CSV.name} …")
            _FEATURE_POOL = _load_pool()
        else:
            print("  Pool CSV not found — generating from scratch …")
            _FEATURE_POOL = _build_pool()
    return _FEATURE_POOL


# ── Scenario features: build or load ─────────────────────────────────────────

def _feat_cols_from_pool() -> list[str]:
    pool = _get_pool()
    n = next(iter(pool.values())).shape[1]
    return [f"feat_{i:03d}" for i in range(n)]


def _scenario_csv(scenario_name: str, split: str) -> Path:
    return DATA_DIR / f"features_{scenario_name}_{split}.csv"


def generate_scenario_features(
    scenario_name: str, n_days: int, split: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, y) for the given scenario.
    Loads from CSV if it exists; otherwise samples from the feature pool and saves.
    split: 'train' | 'test'
    """
    csv_path = _scenario_csv(scenario_name, split)

    if csv_path.exists():
        df        = pd.read_csv(csv_path)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df[feat_cols].values.astype(np.float32)
        y = df["label_int"].values.astype(np.int32)
        return X, y

    pool  = _get_pool()
    daily = DAILY_WINDOWS[scenario_name]
    X_parts, y_parts = [], []

    for _ in tqdm(range(n_days), desc=f"  {scenario_name}", unit="day", ncols=72, leave=False):
        for label, n_win in daily.items():
            n   = max(1, int(n_win * RNG.uniform(0.8, 1.2)))
            idx = RNG.integers(0, len(pool[label]), size=n)
            X_parts.append(pool[label][idx])
            y_parts.append(np.full(n, int(label), dtype=np.int32))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Persist to CSV
    feat_cols = [f"feat_{i:03d}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df.insert(0, "label",     [CLASS_NAMES[yi] for yi in y])
    df.insert(1, "label_int", y)
    df.to_csv(csv_path, index=False)
    tqdm.write(f"  Saved → {csv_path.name}  ({len(X):,} rows)")

    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def build_training_data() -> tuple[np.ndarray, np.ndarray]:
    print(f"\n{'='*60}")
    print("Step 1 — Training data  (S1/S2/S3 × 180 days)")
    print(f"{'='*60}")
    X_list, y_list = [], []
    for sc in tqdm(TRAIN_SCENARIOS, desc="Scenarios", unit="sc", ncols=72):
        t0   = time.time()
        X, y = generate_scenario_features(sc, N_DAYS_TRAIN, split="train")
        tqdm.write(f"  {sc:20s} → {len(X):>7,} windows  ({time.time()-t0:.1f}s)")
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def train_model(X: np.ndarray, y: np.ndarray) -> lgb.LGBMClassifier:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nStep 2 — Train LightGBM  (train={len(X_tr):,}  val={len(X_val):,})")
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"  Validation accuracy: {val_acc:.4f}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model: lgb.LGBMClassifier, scenario_name: str) -> dict:
    """Evaluate model on one scenario. Returns structured metrics dict."""
    X, y   = generate_scenario_features(scenario_name, N_DAYS_TEST, split="test")
    y_pred = model.predict(X)

    acc          = accuracy_score(y, y_pred)
    labels       = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[l] for l in labels]

    print(f"   Samples  : {len(y):,}  (30-day held-out)")
    print(f"   Accuracy : {acc:.4f}")
    print(classification_report(y, y_pred, labels=labels,
                                target_names=target_names, digits=3))

    cm = confusion_matrix(y, y_pred, labels=labels)
    print("   Confusion matrix (rows=true, cols=pred):")
    print("            " + "  ".join(f"{n:>8}" for n in target_names))
    for i, name in enumerate(target_names):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        print(f"  {name:>10}: {row}")

    # Per-class detailed metrics
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1s, sup = precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0
    )
    per_class = {}
    for i, lbl in enumerate(labels):
        per_class[CLASS_NAMES[lbl]] = {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1s[i]),
            "support":   int(sup[i]),
        }

    sc_idx    = labels.index(int(BehaviorLabel.SCRATCH))
    scratch   = per_class["scratch"]
    print(f"\n   ★ Scratch  precision={scratch['precision']:.3f}  "
          f"recall={scratch['recall']:.3f}  F1={scratch['f1']:.3f}")

    return {
        "accuracy":         acc,
        "n_samples":        len(y),
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
        "labels":           labels,
    }


def show_top_features(model: lgb.LGBMClassifier, top_n: int = 15) -> None:
    importance = model.feature_importances_
    idx        = np.argsort(importance)[::-1][:top_n]
    print(f"\nTop {top_n} features:")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:2d}. feat_{i:03d}  importance={importance[i]:.1f}")


# ── Full evaluation pipeline (importable) ────────────────────────────────────

SCENARIO_TYPES = {
    "S1_Normal":      "训练内",
    "S2_Active":      "训练内",
    "S3_Calm":        "训练内",
    "S4_Mild_skin":   "训练外",
    "S5_Severe_skin": "训练外",
}


def run_full_evaluation(force_fresh: bool = False) -> dict:
    """
    Train model on S1/S2/S3 and evaluate on all 5 scenarios.
    Returns structured metrics for writing to evaluation CSVs.
    If force_fresh=True, deletes cached data before running.
    """
    if force_fresh and DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(exist_ok=True)
        print(f"  [fresh] 已清除缓存目录 {DATA_DIR}")

    # Training
    X_train, y_train = build_training_data()
    model = train_model(X_train, y_train)

    # Save production model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {MODEL_PATH}")

    # Evaluate all scenarios
    print(f"\n{'='*60}")
    print("Step 3 — Per-scenario evaluation (30-day held-out)")
    print(f"{'='*60}")
    scenario_results = {}
    for sc in tqdm(TEST_SCENARIOS, desc="Evaluating", unit="sc", ncols=72):
        tag = "(UNSEEN)" if sc not in TRAIN_SCENARIOS else "(seen)  "
        tqdm.write(f"\n── {sc} {tag} ──")
        metrics = evaluate_model(model, sc)
        scenario_results[sc] = metrics

    # Feature importance
    show_top_features(model)
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    feature_importance = [
        (f"feat_{i:03d}", float(importance[i])) for i in sorted_idx
    ]

    return {
        "scenarios":          scenario_results,
        "feature_importance": feature_importance,
        "model":              model,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    results = run_full_evaluation()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for sc, m in results["scenarios"].items():
        tag = "(UNSEEN)" if sc not in TRAIN_SCENARIOS else "(seen)  "
        print(f"  {tag} {sc:25s} accuracy={m['accuracy']:.4f}")
    print(f"\n  Data dir : {DATA_DIR}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
