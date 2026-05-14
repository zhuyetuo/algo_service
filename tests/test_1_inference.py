"""
Test 1 — IMU Behavior Classification
=====================================
Generates synthetic IMU data for 5 dog scenarios over 180 days,
trains a LightGBM model on scenarios 1-3, then evaluates on all 5.

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

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
from tqdm import tqdm

from modules.inference.model import extract_features, segment, BehaviorLabel

# ── Constants ────────────────────────────────────────────────────────────────
FS            = 50          # Hz
WINDOW_SEC    = 3
OVERLAP       = 0.5
WIN_SAMPLES   = WINDOW_SEC * FS                    # 150 samples
STEP_SAMPLES  = int(WIN_SAMPLES * (1 - OVERLAP))   # 75 samples

N_DAYS        = 180   # 6 months
RNG           = np.random.default_rng(42)

# Daily windows per class (represents realistic 6-month data volumes)
DAILY_WINDOWS = {
    "S1_Normal":      {BehaviorLabel.MOVEMENT: 480, BehaviorLabel.SLEEP: 350, BehaviorLabel.SCRATCH:  25},
    "S2_Active":      {BehaviorLabel.MOVEMENT: 620, BehaviorLabel.SLEEP: 220, BehaviorLabel.SCRATCH:  15},
    "S3_Calm":        {BehaviorLabel.MOVEMENT: 175, BehaviorLabel.SLEEP: 670, BehaviorLabel.SCRATCH:  15},
    "S4_Mild_skin":   {BehaviorLabel.MOVEMENT: 390, BehaviorLabel.SLEEP: 340, BehaviorLabel.SCRATCH:  75},
    "S5_Severe_skin": {BehaviorLabel.MOVEMENT: 300, BehaviorLabel.SLEEP: 300, BehaviorLabel.SCRATCH: 150},
}

TRAIN_SCENARIOS = ["S1_Normal", "S2_Active", "S3_Calm"]
TEST_SCENARIOS  = list(DAILY_WINDOWS.keys())


# ── IMU signal generators ────────────────────────────────────────────────────

def _movement_window() -> np.ndarray:
    n = WIN_SAMPLES
    t = np.arange(n) / FS
    freq  = RNG.uniform(1.5, 2.5)
    phi   = RNG.uniform(0, 2 * np.pi)
    a_amp = RNG.uniform(0.4, 0.8)
    g_amp = RNG.uniform(0.2, 0.5)

    ax = a_amp * np.sin(2 * np.pi * freq * t + phi)       + RNG.normal(0, 0.05, n)
    ay = a_amp * 0.6 * np.sin(2 * np.pi * freq * t + phi + 0.5) + RNG.normal(0, 0.04, n)
    az = 9.8 + a_amp * 0.5 * np.sin(2 * np.pi * freq * t) + RNG.normal(0, 0.06, n)
    gx = g_amp * np.sin(2 * np.pi * freq * t)             + RNG.normal(0, 0.03, n)
    gy = g_amp * 0.8 * np.cos(2 * np.pi * freq * t)       + RNG.normal(0, 0.03, n)
    gz = RNG.normal(0, 0.02, n)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def _sleep_window() -> np.ndarray:
    n    = WIN_SAMPLES
    t    = np.arange(n) / FS
    bfreq = RNG.uniform(0.2, 0.4)   # breathing

    ax = 0.02 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.008, n)
    ay = RNG.normal(0, 0.008, n)
    az = 9.8 + 0.03 * np.sin(2 * np.pi * bfreq * t) + RNG.normal(0, 0.008, n)
    gx = RNG.normal(0, 0.004, n)
    gy = RNG.normal(0, 0.004, n)
    gz = RNG.normal(0, 0.004, n)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


def _scratch_window() -> np.ndarray:
    n     = WIN_SAMPLES
    t     = np.arange(n) / FS
    freq  = RNG.uniform(4.0, 8.0)
    amp   = RNG.uniform(1.5, 3.0)
    dom   = RNG.integers(0, 3)   # dominant accelerometer axis

    amps_a = [0.25 * amp, 0.25 * amp, 0.25 * amp]
    amps_a[dom] = amp

    ax = amps_a[0] * np.sin(2 * np.pi * freq * t)          + RNG.normal(0, 0.1, n)
    ay = amps_a[1] * np.sin(2 * np.pi * freq * t + np.pi/4) + RNG.normal(0, 0.1, n)
    az = 9.8 + amps_a[2] * np.sin(2 * np.pi * freq * t)    + RNG.normal(0, 0.1, n)
    gx = 0.6 * amp * np.sin(2 * np.pi * freq * t)           + RNG.normal(0, 0.05, n)
    gy = 0.5 * amp * np.cos(2 * np.pi * freq * t)           + RNG.normal(0, 0.05, n)
    gz = 0.3 * amp * np.sin(2 * np.pi * freq * t + np.pi/3) + RNG.normal(0, 0.05, n)
    return np.column_stack([ax, ay, az, gx, gy, gz]).astype(np.float32)


_GENERATORS = {
    BehaviorLabel.MOVEMENT: _movement_window,
    BehaviorLabel.SLEEP:    _sleep_window,
    BehaviorLabel.SCRATCH:  _scratch_window,
}


# ── Data generation ──────────────────────────────────────────────────────────

def generate_scenario_features(scenario_name: str, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y) where X is (N, n_features) and y is (N,) integer labels.
    Simulates n_days of daily behavior data for the given scenario.
    """
    daily = DAILY_WINDOWS[scenario_name]
    X_parts, y_parts = [], []

    for day in tqdm(range(n_days), desc=f"  {scenario_name}", unit="day", ncols=70, leave=False):
        for label, n_windows_per_day in daily.items():
            # Add daily variation (±20 %)
            n = max(1, int(n_windows_per_day * RNG.uniform(0.8, 1.2)))
            for _ in range(n):
                window = _GENERATORS[label]()
                feat   = extract_features(window, FS)
                X_parts.append(feat)
                y_parts.append(int(label))

    return np.array(X_parts, dtype=np.float32), np.array(y_parts, dtype=np.int32)


# ── Training ─────────────────────────────────────────────────────────────────

def build_training_data():
    print(f"\n{'='*60}")
    print("Generating training data (scenarios: S1 / S2 / S3 × 180 days)")
    print(f"{'='*60}")
    X_list, y_list = [], []
    for sc in tqdm(TRAIN_SCENARIOS, desc="Scenarios", unit="scenario", ncols=70):
        t0 = time.time()
        X, y = generate_scenario_features(sc, N_DAYS)
        tqdm.write(f"  {sc:20s} → {len(X):>7,} windows  ({time.time()-t0:.1f}s)")
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def train_model(X: np.ndarray, y: np.ndarray):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTraining LightGBM  (train={len(X_tr):,}  val={len(X_val):,})")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
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

    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"Validation accuracy: {val_acc:.4f}")
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

CLASS_NAMES = {
    int(BehaviorLabel.MOVEMENT): "movement",
    int(BehaviorLabel.SLEEP):    "sleep",
    int(BehaviorLabel.SCRATCH):  "scratch",
}


def evaluate_model(model, scenario_name: str):
    X, y = generate_scenario_features(scenario_name, n_days=30)   # 1 month test
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    print(f"   Samples  : {len(y):,}  (30-day held-out)")
    print(f"   Accuracy : {acc:.4f}")

    labels = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[l] for l in labels]
    print(classification_report(y, y_pred, labels=labels, target_names=target_names, digits=3))

    cm = confusion_matrix(y, y_pred, labels=labels)
    print("   Confusion matrix (rows=true, cols=pred):")
    header = "            " + "  ".join(f"{n:>8}" for n in target_names)
    print(header)
    for i, name in enumerate(target_names):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(labels)))
        print(f"  {name:>10}: {row}")

    # Scratch-specific metrics (most important)
    sc_idx = labels.index(int(BehaviorLabel.SCRATCH))
    tp = cm[sc_idx, sc_idx]
    fn = cm[sc_idx].sum() - tp
    fp = cm[:, sc_idx].sum() - tp
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"\n   ★ Scratch  precision={precision:.3f}  recall={recall:.3f}  F1={f1:.3f}")
    return acc


# ── Feature importance ───────────────────────────────────────────────────────

def show_top_features(model, top_n: int = 15):
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:top_n]
    print(f"\nTop {top_n} feature importances:")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:2d}. feature_{i:03d}  importance={importance[i]:.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # 1. Generate training data
    X_train, y_train = build_training_data()
    print(f"\nTotal training windows: {len(X_train):,}  "
          f"(move={int((y_train==1).sum()):,}  "
          f"sleep={int((y_train==2).sum()):,}  "
          f"scratch={int((y_train==3).sum()):,})")

    # 2. Train
    model = train_model(X_train, y_train)

    # 3. Save model
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, "behavior_lgbm.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {model_path}")

    # 4. Evaluate on all 5 scenarios
    print(f"\n{'='*60}")
    print("Per-scenario evaluation (30-day held-out per scenario)")
    print(f"{'='*60}")
    accs = {}
    for sc in tqdm(TEST_SCENARIOS, desc="Evaluating", unit="scenario", ncols=70):
        tqdm.write(f"\n── {sc} {'(UNSEEN)' if sc not in TRAIN_SCENARIOS else '(seen)'} ──")
        accs[sc] = evaluate_model(model, sc)

    # 5. Feature importance
    show_top_features(model)

    # 6. Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for sc, acc in accs.items():
        tag = "(UNSEEN)" if sc not in TRAIN_SCENARIOS else "(seen)  "
        print(f"  {tag} {sc:25s} accuracy={acc:.4f}")
    print(f"\nTotal time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
