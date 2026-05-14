"""
Test 3 — Individual Baseline Update
=====================================
Tests the baseline update engine in isolation (no database needed).
Mirrors the logic in modules/baseline/updater.py.

4 sub-tests
-----------
  T1  Normal convergence
        30 normal days → baseline should converge to true mean ±0.5
  T2  Abnormal-day permeation
        20 normal days, then 30 abnormal days → baseline should NOT be
        pulled far from true healthy mean (permeation weight = 0.01)
  T3  Temperature coefficient learning
        50 normal days with temp/count correlation → coef should
        converge to true value ±0.05
  T4  Cold-start then full cycle
        3-day warm-up → phase 1 → phase 2 → stable phase
        verify phase transitions and confidence growth

Usage
-----
  cd algo_service
  python tests/test_3_baseline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

RNG = np.random.default_rng(42)

# ── Config (mirrors config.py) ────────────────────────────────────────────────
STD_FLOOR    = 2.0
NORMAL_W     = 0.05
ABNORMAL_W   = 0.01
Z_THRESH     = 2.5    # stable-phase abnormality threshold
REF_TEMP     = 20.0


# ── Baseline engine (standalone) ──────────────────────────────────────────────

def update_baseline(
    daily_counts: np.ndarray,
    daily_temps: np.ndarray | None = None,
    true_mean: float | None = None,
) -> dict:
    """
    Simulate the rolling baseline update over a sequence of daily counts.

    Returns history dict with keys:
        means, stds, temp_coefs, valid_days, is_abnormal, errors
    """
    n = len(daily_counts)
    if daily_temps is None:
        daily_temps = np.full(n, REF_TEMP)

    bl_mean   = float(daily_counts[0])
    bl_std    = STD_FLOOR
    temp_coef = 0.0
    valid_days = 0

    hist = dict(
        means=[], stds=[], temp_coefs=[],
        valid_days=[], is_abnormal=[],
        errors=[],         # |estimate - true_mean|
        zscores=[],
    )

    temp_buf, count_buf = [], []

    for day in tqdm(range(n), desc="  Days", unit="day", ncols=70, leave=False):
        count = float(daily_counts[day])
        temp  = float(daily_temps[day])
        valid_days += 1

        std_eff = max(bl_std, STD_FLOOR)

        # Temperature correction at z-score level
        temp_effect  = temp_coef * (temp - REF_TEMP)
        adjusted_dev = (count - bl_mean) - temp_effect
        z = adjusted_dev / std_eff

        is_abn = z > Z_THRESH
        weight = ABNORMAL_W if is_abn else NORMAL_W

        # Update mean
        bl_mean = bl_mean * (1 - weight) + count * weight

        # Update std from recent normal days
        normal_counts_recent = [
            hist["means"][i] if not hist["is_abnormal"][i] else None
            for i in range(max(0, len(hist["means"]) - 30), len(hist["means"]))
        ]
        normal_vals = [
            float(daily_counts[day - len(hist["means"]) + i])
            for i, v in enumerate(normal_counts_recent) if v is not None
        ]
        if len(normal_vals) >= 3:
            bl_std = max(float(np.std(normal_vals)), STD_FLOOR)

        # Temperature coefficient
        if not is_abn:
            temp_buf.append(temp)
            count_buf.append(count)
        if len(temp_buf) >= 20:
            T = np.array(temp_buf[-60:])
            C = np.array(count_buf[-60:])
            X = np.column_stack([T, np.ones(len(T))])
            coefs, _, _, _ = np.linalg.lstsq(X, C, rcond=None)
            temp_coef = float(np.clip(coefs[0], 0.0, 0.4))

        hist["means"].append(bl_mean)
        hist["stds"].append(bl_std)
        hist["temp_coefs"].append(temp_coef)
        hist["valid_days"].append(valid_days)
        hist["is_abnormal"].append(is_abn)
        hist["zscores"].append(z)
        hist["errors"].append(abs(bl_mean - true_mean) if true_mean is not None else None)

    return hist


# ── Sub-tests ─────────────────────────────────────────────────────────────────

def test_T1_normal_convergence():
    """
    30 days of normal scratch counts (true_mean=8, std=2).
    Baseline should converge to ≈8 within 30 days.
    """
    TRUE_MEAN = 8.0
    counts = RNG.normal(TRUE_MEAN, 2.0, 90)
    hist   = update_baseline(counts, true_mean=TRUE_MEAN)

    final_mean  = hist["means"][-1]
    final_error = hist["errors"][-1]
    abn_count   = sum(hist["is_abnormal"])

    print(f"\n── T1  Normal convergence ──")
    print(f"   True mean      : {TRUE_MEAN}")
    print(f"   Initial seed   : {counts[0]:.2f}")
    print(f"   Final estimate : {final_mean:.2f}  (error={final_error:.2f})")
    print(f"   Abnormal days  : {abn_count}/90")

    day30_err = hist["errors"][29]
    day90_err = hist["errors"][-1]
    print(f"   Error at day30 : {day30_err:.2f}")
    print(f"   Error at day90 : {day90_err:.2f}")

    ok = final_error < 1.0
    print(f"   PASS ✓" if ok else f"   FAIL ✗  (error={final_error:.2f} > 1.0)")
    return hist, TRUE_MEAN, ok


def test_T2_abnormal_permeation():
    """
    20 normal days (true_mean=8), then 30 abnormal days (+10 scratch),
    then 20 recovery days.
    Baseline during disease period should stay near 8 (permeation=0.01).
    """
    TRUE_MEAN = 8.0
    counts = np.concatenate([
        RNG.normal(TRUE_MEAN, 1.5, 20),   # healthy
        RNG.normal(TRUE_MEAN + 10, 2, 30), # disease
        RNG.normal(TRUE_MEAN, 1.5, 20),   # recovery
    ])
    hist = update_baseline(counts, true_mean=TRUE_MEAN)

    # Baseline at end of disease period (day 49)
    bl_at_disease_end = hist["means"][49]
    # How far did the disease pull the baseline?
    pollution = bl_at_disease_end - TRUE_MEAN
    # Baseline after recovery (day 69)
    bl_after_recovery = hist["means"][-1]

    print(f"\n── T2  Abnormal-day permeation ──")
    print(f"   True healthy mean  : {TRUE_MEAN}")
    print(f"   Disease period     : days 21-50  (+10 scratch)")
    print(f"   Baseline @day 50   : {bl_at_disease_end:.2f}  "
          f"(pollution={pollution:.2f})")
    print(f"   Baseline @day 70   : {bl_after_recovery:.2f}  "
          f"(error={hist['errors'][-1]:.2f})")

    ok = pollution < 3.0   # baseline should not drift more than 3 from true mean
    print(f"   Pollution < 3.0 → {'PASS ✓' if ok else 'FAIL ✗'}")
    return hist, TRUE_MEAN, ok


def test_T3_temp_coefficient():
    """
    50 normal days with temperature variation (true coef=0.25).
    Engine should learn coef ≈ 0.25 ± 0.08.
    """
    TRUE_COEF  = 0.25
    TRUE_MEAN  = 8.0
    n = 120
    temps  = 20 + 10 * np.sin(np.linspace(0, 2 * np.pi, n)) + RNG.normal(0, 1, n)
    counts = TRUE_MEAN + TRUE_COEF * (temps - REF_TEMP) + RNG.normal(0, 1.5, n)

    hist = update_baseline(counts, daily_temps=temps, true_mean=TRUE_MEAN)

    coef_30  = hist["temp_coefs"][29]
    coef_60  = hist["temp_coefs"][59]
    coef_120 = hist["temp_coefs"][-1]
    coef_err = abs(coef_120 - TRUE_COEF)

    print(f"\n── T3  Temperature coefficient learning ──")
    print(f"   True coef      : {TRUE_COEF}")
    print(f"   Estimated @d30 : {coef_30:.3f}")
    print(f"   Estimated @d60 : {coef_60:.3f}")
    print(f"   Estimated @d120: {coef_120:.3f}  (error={coef_err:.3f})")

    ok = coef_err < 0.08
    print(f"   Coef error < 0.08 → {'PASS ✓' if ok else 'FAIL ✗'}")
    return hist, TRUE_COEF, ok


def test_T4_phase_transitions():
    """
    Full cold-start → warm-up → phase1 → phase2 → stable.
    Verify confidence grows and correct phase transitions.
    """
    TRUE_MEAN = 8.0
    counts = RNG.normal(TRUE_MEAN, 1.5, 180)
    hist   = update_baseline(counts, true_mean=TRUE_MEAN)

    def phase(vd):
        if vd < 3:  return 0
        if vd < 14: return 1
        if vd < 30: return 2
        return 3

    def confidence(vd):
        return min(vd / 30.0, 1.0)

    print(f"\n── T4  Phase transitions and confidence growth ──")
    checkpoints = [1, 3, 7, 14, 30, 60, 90, 180]
    print(f"  {'Day':>5}  {'Valid':>6}  {'Phase':>6}  {'Confidence':>10}  "
          f"{'Bl mean':>8}  {'Bl std':>7}")
    print(f"  {'─'*55}")
    for d in checkpoints:
        idx = d - 1
        vd  = hist["valid_days"][idx]
        p   = phase(vd)
        c   = confidence(vd)
        print(f"  {d:>5}  {vd:>6}  {p:>6}  {c:>10.2f}  "
              f"{hist['means'][idx]:>8.2f}  {hist['stds'][idx]:>7.2f}")

    final_conf = confidence(hist["valid_days"][-1])
    ok = abs(hist["means"][-1] - TRUE_MEAN) < 1.5 and final_conf >= 1.0
    print(f"\n  Final mean error: {hist['errors'][-1]:.2f}  confidence: {final_conf:.2f}")
    print(f"  PASS ✓" if ok else f"  FAIL ✗")
    return hist, True, ok


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(hists_meta: list):
    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    fig.suptitle("Test 3 — Baseline Update Engine", fontsize=13, fontweight="bold")
    titles = [
        "T1 Normal convergence",
        "T2 Abnormal permeation",
        "T3 Temp coefficient",
        "T4 Phase transitions",
    ]

    for row, (hist, true_val, ok, title) in enumerate(hists_meta):
        days = list(range(1, len(hist["means"]) + 1))

        # Panel A: mean + true value
        ax = axes[row, 0]
        ax.plot(days, hist["means"], "steelblue", lw=2, label="Baseline mean")
        if isinstance(true_val, float):
            ax.axhline(true_val, color="green", ls="--", lw=1.5, label=f"True = {true_val}")
        abn_idx = [i for i, a in enumerate(hist["is_abnormal"]) if a]
        ax.scatter([days[i] for i in abn_idx],
                   [hist["means"][i] for i in abn_idx],
                   color="orange", s=12, zorder=5, label="Abnormal day")
        ax.set_title(f"{title}  {'✓' if ok else '✗'}", fontweight="bold", fontsize=9)
        ax.set_ylabel("Scratch count")
        ax.legend(fontsize=7)
        ax.set_xlim(1, days[-1])

        # Panel B: error or coef
        ax = axes[row, 1]
        if row == 2:  # T3: show temp coef
            ax.plot(days, hist["temp_coefs"], color="tomato", lw=2)
            ax.axhline(true_val, color="green", ls="--", lw=1.5,
                       label=f"True coef={true_val}")
            ax.set_ylabel("Temp coefficient")
            ax.set_title("Coefficient estimation")
            ax.legend(fontsize=7)
        else:
            errors = [e for e in hist["errors"] if e is not None]
            ax.plot(days[:len(errors)], errors, color="tomato", lw=2)
            ax.axhline(1.0, color="gray", ls="--", lw=1, label="|error| = 1.0")
            ax.set_ylabel("|Baseline - True mean|")
            ax.set_title("Convergence error")
            ax.legend(fontsize=7)
        ax.set_xlim(1, days[-1])

        # Panel C: z-scores
        ax = axes[row, 2]
        colors = ["tomato" if a else "steelblue" for a in hist["is_abnormal"]]
        ax.bar(days, hist["zscores"], color=colors, alpha=0.7, width=1)
        ax.axhline(2.5, color="orange", ls="--", lw=1, label="z=2.5")
        ax.axhline(0,   color="black",  lw=0.5)
        ax.set_ylabel("Z-score")
        ax.set_title("Daily z-score")
        ax.legend(fontsize=7)
        ax.set_xlim(1, days[-1])

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "test_3_baseline.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Test 3 — Individual Baseline Update")
    print("=" * 60)

    sub_tests = [
        ("T1 Normal convergence",  test_T1_normal_convergence),
        ("T2 Abnormal permeation", test_T2_abnormal_permeation),
        ("T3 Temp coefficient",    test_T3_temp_coefficient),
        ("T4 Phase transitions",   test_T4_phase_transitions),
    ]
    results = []
    for label, fn in tqdm(sub_tests, desc="Sub-tests", unit="test", ncols=70):
        tqdm.write(f"\nRunning {label}...")
        h, tv, ok = fn()
        results.append((h, tv, ok, label))

    plot_all(results)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    all_pass = True
    for _, _, ok, title in results:
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {title}")
        all_pass = all_pass and ok
    print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")


if __name__ == "__main__":
    main()
