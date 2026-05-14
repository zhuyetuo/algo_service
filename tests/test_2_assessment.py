"""
Test 2 — Skin Health Assessment (5 scenarios × 180 days)
=========================================================
Standalone implementation of the assessment engine — no database needed.
Mirrors the logic in modules/assessment/evaluator.py exactly.

5 scenarios
-----------
  A  Normal          healthy dog, no alerts expected
  B  Skin disease    episode on days 45-75, recovers → 1 alert expected
  C  Seasonal        summer temperature increase → 0 alerts (temp-corrected)
  D  Slow increase   gradual drift over 180 days → baseline tracks, ≤1 alert
  E  Data gaps       device off/lost for several periods → no false alerts

Assessment design (from conversation)
--------------------------------------
- Warm-up:   days  1-3   (no assessment)
- Phase 1:   days  4-14  z>4.0, consec≥5, avg_z>5.0
- Phase 2:   days 15-30  z>3.5, consec≥4, avg_z>4.0
- Phase 3:   days 31+    z>2.5, consec≥3, avg_z>3.5
- Temperature correction at z-score level (raw counts unchanged)
- Abnormal-day baseline update weight: 0.01 (normal: 0.05)

Usage
-----
  cd algo_service
  python tests/test_2_assessment.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from tqdm import tqdm

RNG   = np.random.default_rng(42)
N_DAYS = 180

# ── Thresholds ────────────────────────────────────────────────────────────────

@dataclass
class Thresh:
    z: float
    consec: int
    avg_z: float

PHASE_THRESH = {
    1: Thresh(4.0, 5, 5.0),
    2: Thresh(3.5, 4, 4.0),
    3: Thresh(2.5, 3, 3.5),
}
STD_FLOOR      = 2.0
NORMAL_W       = 0.05
ABNORMAL_W     = 0.01


def _phase(valid_days: int) -> int:
    if valid_days < 3:  return 0
    if valid_days < 14: return 1
    if valid_days < 30: return 2
    return 3


def _thresh(valid_days: int) -> Thresh | None:
    p = _phase(valid_days)
    return PHASE_THRESH.get(p)


# ── Assessment engine (standalone) ───────────────────────────────────────────

@dataclass
class DayRecord:
    day: int
    scratch_count: int
    temperature: float
    data_quality: int   # 0=ok, 1=no data
    # computed
    baseline_mean: float = 0.0
    baseline_std:  float = 0.0
    temp_coef:     float = 0.0
    temp_effect:   float = 0.0
    zscore:        float | None = None
    avg_zscore:    float | None = None
    consec_abnormal: int = 0
    is_abnormal:   bool = False
    alert:         bool = False
    phase:         int  = 0
    valid_days:    int  = 0


def run_assessment(
    counts: np.ndarray,
    temperatures: np.ndarray,
    data_quality: np.ndarray,   # 0=valid, 1=gap
) -> list[DayRecord]:
    """
    Run 180-day assessment engine on daily scratch counts.

    counts, temperatures, data_quality: arrays of length N_DAYS
    Returns list of DayRecord (one per day).
    """
    records: list[DayRecord] = []

    bl_mean    = float(counts[0])
    bl_std     = STD_FLOOR
    temp_coef  = 0.0
    valid_days = 0
    consec     = 0
    REF_TEMP   = 20.0

    # Buffers for temp-coef estimation
    temp_buf  = []
    count_buf = []

    for day in tqdm(range(N_DAYS), desc="  Simulating", unit="day", ncols=70, leave=False):
        count   = int(counts[day])
        temp    = float(temperatures[day])
        quality = int(data_quality[day])

        rec = DayRecord(day=day+1, scratch_count=count,
                        temperature=temp, data_quality=quality)

        if quality != 0:
            # Gap day — freeze baseline, reset consec
            consec = 0
            rec.baseline_mean = bl_mean
            rec.baseline_std  = bl_std
            rec.temp_coef     = temp_coef
            records.append(rec)
            continue

        valid_days += 1
        phase  = _phase(valid_days)
        thresh = _thresh(valid_days)

        rec.valid_days    = valid_days
        rec.phase         = phase
        rec.baseline_mean = bl_mean
        rec.baseline_std  = max(bl_std, STD_FLOOR)
        rec.temp_coef     = temp_coef

        if thresh is None:
            # Warm-up: update baseline, no assessment
            bl_mean = bl_mean * (1 - NORMAL_W) + count * NORMAL_W
            records.append(rec)
            continue

        # Temperature effect (subtracted from deviation, NOT from raw count)
        temp_effect = temp_coef * (temp - REF_TEMP)
        rec.temp_effect = round(temp_effect, 2)

        # Z-score
        std_eff = max(bl_std, STD_FLOOR)
        adjusted_dev = (count - bl_mean) - temp_effect
        z = adjusted_dev / std_eff
        rec.zscore = round(z, 3)

        # Recent avg_z (last consec-1 days + today)
        recent_z = [r.zscore for r in records[-thresh.consec+1:]
                    if r.zscore is not None and r.data_quality == 0]
        recent_z.append(z)
        avg_z = float(np.mean(recent_z))
        rec.avg_zscore = round(avg_z, 3)

        # Abnormality check
        is_abn = z > thresh.z
        rec.is_abnormal = is_abn

        consec = consec + 1 if is_abn else 0
        rec.consec_abnormal = consec

        # Alert: consecutive days + amplitude filter
        if is_abn and consec >= thresh.consec and avg_z >= thresh.avg_z:
            rec.alert = True

        # Baseline update
        weight = ABNORMAL_W if is_abn else NORMAL_W
        bl_mean = bl_mean * (1 - weight) + count * weight

        # Std: recompute from last 30 normal days
        normal_counts = [r.scratch_count for r in records[-30:]
                         if not r.is_abnormal and r.data_quality == 0]
        if len(normal_counts) >= 3:
            bl_std = max(float(np.std(normal_counts)), STD_FLOOR)

        # Temp coefficient: least-squares once we have 20 normal points
        if quality == 0 and not is_abn:
            temp_buf.append(temp)
            count_buf.append(count)
        if len(temp_buf) >= 20:
            T = np.array(temp_buf[-60:])
            C = np.array(count_buf[-60:])
            X = np.column_stack([T, np.ones(len(T))])
            coefs, _, _, _ = np.linalg.lstsq(X, C, rcond=None)
            temp_coef = float(np.clip(coefs[0], 0.0, 0.4))

        records.append(rec)

    return records


# ── Scenario generators ───────────────────────────────────────────────────────

def _seasonal_temp(n: int) -> np.ndarray:
    """Sinusoidal temperature: 15°C in winter, 32°C in summer."""
    t = np.linspace(0, 2 * np.pi, n)
    return 23.5 + 8.5 * np.sin(t - np.pi / 2)


def scenario_A_normal() -> tuple:
    """Healthy dog — no skin events."""
    counts = RNG.integers(5, 13, N_DAYS).astype(float)
    counts += RNG.normal(0, 0.5, N_DAYS)
    temps   = _seasonal_temp(N_DAYS) + RNG.normal(0, 1, N_DAYS)
    quality = np.zeros(N_DAYS, dtype=int)
    return counts.clip(0), temps, quality, "A — Normal (no disease)"


def scenario_B_skin_disease() -> tuple:
    """Skin disease episode days 45-75, true baseline=8."""
    counts  = RNG.normal(8, 1.5, N_DAYS)
    # Disease period: scratch doubles
    counts[44:75] += RNG.normal(10, 2, 31)
    temps   = _seasonal_temp(N_DAYS) + RNG.normal(0, 1, N_DAYS)
    quality = np.zeros(N_DAYS, dtype=int)
    return counts.clip(0), temps, quality, "B — Skin disease (days 45-75)"


def scenario_C_seasonal() -> tuple:
    """Summer heat causes more scratching (real temp correlation)."""
    temps   = _seasonal_temp(N_DAYS) + RNG.normal(0, 1, N_DAYS)
    # True relationship: +0.25 scratches per °C above 20°C
    base    = RNG.normal(8, 1.5, N_DAYS)
    counts  = base + 0.25 * (temps - 20.0)
    quality = np.zeros(N_DAYS, dtype=int)
    return counts.clip(0), temps, quality, "C — Seasonal temperature increase"


def scenario_D_slow_increase() -> tuple:
    """Gradual drift: scratch increases from 8→18 over 180 days."""
    trend   = np.linspace(0, 10, N_DAYS)
    counts  = RNG.normal(8, 1.5, N_DAYS) + trend
    temps   = _seasonal_temp(N_DAYS) + RNG.normal(0, 1, N_DAYS)
    quality = np.zeros(N_DAYS, dtype=int)
    return counts.clip(0), temps, quality, "D — Slow gradual increase"


def scenario_E_gaps() -> tuple:
    """
    Data gaps:
      - days 30-34   (forgot to wear)
      - days 60-69   (battery dead)
      - days 100-107 (signal lost)
    Skin disease days 120-145 to verify detection still works after gaps.
    """
    counts  = RNG.normal(8, 1.5, N_DAYS)
    counts[119:145] += RNG.normal(9, 2, 26)          # disease after gaps
    temps   = _seasonal_temp(N_DAYS) + RNG.normal(0, 1, N_DAYS)
    quality = np.zeros(N_DAYS, dtype=int)
    quality[29:34]   = 1   # forgot to wear
    quality[59:69]   = 1   # battery
    quality[99:107]  = 1   # signal
    return counts.clip(0), temps, quality, "E — Data gaps + late disease"


SCENARIOS = [
    scenario_A_normal,
    scenario_B_skin_disease,
    scenario_C_seasonal,
    scenario_D_slow_increase,
    scenario_E_gaps,
]


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(records: list[DayRecord], title: str):
    alerts     = [r for r in records if r.alert]
    abn_days   = [r for r in records if r.is_abnormal]
    gap_days   = [r for r in records if r.data_quality != 0]
    final_bl   = records[-1].baseline_mean
    final_std  = records[-1].baseline_std
    final_coef = records[-1].temp_coef

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    print(f"  Days total    : {len(records)}")
    print(f"  Gap days      : {len(gap_days)}")
    print(f"  Abnormal days : {len(abn_days)}")
    print(f"  Alerts sent   : {len(alerts)}")
    if alerts:
        for a in alerts:
            print(f"    → Day {a.day:3d}  z={a.zscore:.2f}  "
                  f"avg_z={a.avg_zscore:.2f}  consec={a.consec_abnormal}")
    print(f"  Final baseline: mean={final_bl:.2f}  std={final_std:.2f}  "
          f"temp_coef={final_coef:.3f}")
    return len(alerts), len(abn_days)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scenario(records: list[DayRecord], title: str, ax_arr):
    days    = [r.day for r in records]
    counts  = [r.scratch_count for r in records]
    bl_mean = [r.baseline_mean for r in records]
    bl_std  = [r.baseline_std for r in records]
    zscores = [r.zscore if r.zscore is not None else 0 for r in records]
    temps   = [r.temperature for r in records]

    alert_days = [r.day for r in records if r.alert]
    abn_days   = [r.day for r in records if r.is_abnormal]
    gap_days   = [r.day for r in records if r.data_quality != 0]

    bl_upper = [m + 2.5 * s for m, s in zip(bl_mean, bl_std)]
    bl_lower = [max(0, m - 2.5 * s) for m, s in zip(bl_mean, bl_std)]

    ax1, ax2, ax3 = ax_arr

    # ── Panel 1: counts + baseline ──
    ax1.fill_between(days, bl_lower, bl_upper, alpha=0.15, color="green", label="±2.5σ normal")
    ax1.plot(days, bl_mean, "g--", lw=1.5, label="Baseline mean")
    ax1.plot(days, counts,  "steelblue", lw=1, alpha=0.7, label="Scratch count")
    for d in abn_days:
        ax1.axvline(d, color="orange", alpha=0.25, lw=1)
    for d in alert_days:
        idx = d - 1
        ax1.axvline(d, color="red", lw=1.5, alpha=0.7)
        ax1.scatter([d], [counts[idx]], color="red", zorder=5, s=60)
    for d in gap_days:
        ax1.axvspan(d - 0.5, d + 0.5, color="lightgray", alpha=0.6)
    ax1.set_title(title, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Scratch count / day")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_xlim(1, N_DAYS)

    # ── Panel 2: z-score ──
    ax2.axhline(0, color="black", lw=0.5)
    ax2.axhline(2.5, color="orange", lw=1, ls="--", alpha=0.7)
    ax2.axhline(4.0, color="red",    lw=1, ls="--", alpha=0.7)
    colors = ["red" if z > 2.5 else "steelblue" for z in zscores]
    ax2.bar(days, zscores, color=colors, alpha=0.7, width=1)
    for d in gap_days:
        ax2.axvspan(d - 0.5, d + 0.5, color="lightgray", alpha=0.6)
    ax2.set_ylabel("Z-score")
    ax2.set_xlim(1, N_DAYS)
    ax2.set_ylim(-4, max(8, max(zscores) + 1))

    # ── Panel 3: temperature + phase ──
    ax3.plot(days, temps, color="tomato", lw=1, label="Temperature °C")
    ax3.set_ylabel("Temperature (°C)", color="tomato")
    ax3.tick_params(axis="y", labelcolor="tomato")
    phase_colors = {0: "lightyellow", 1: "mistyrose", 2: "lightyellow", 3: "honeydew"}
    phase_vals   = [r.phase for r in records]
    for i in range(len(days) - 1):
        ax3.axvspan(days[i] - 0.5, days[i+1] - 0.5,
                    alpha=0.3, color=phase_colors[phase_vals[i]])
    ax3.legend(fontsize=7, loc="upper right")
    ax3.set_xlabel("Day")
    ax3.set_xlim(1, N_DAYS)

    # Status strip
    status_colors = []
    for r in records:
        if r.data_quality != 0:
            status_colors.append([0.85, 0.85, 0.85])   # gray = gap
        elif r.phase == 0:
            status_colors.append([1.0,  1.0,  0.8])    # yellow = warm-up
        elif r.alert:
            status_colors.append([0.9,  0.2,  0.2])    # red = alert
        elif r.is_abnormal:
            status_colors.append([1.0,  0.6,  0.2])    # orange = abnormal
        else:
            status_colors.append([0.4,  0.8,  0.4])    # green = normal
    status_img = np.array(status_colors).reshape(1, N_DAYS, 3)
    ax3.imshow(status_img, aspect="auto", extent=[1, N_DAYS, ax3.get_ylim()[0], ax3.get_ylim()[0] + 2],
               interpolation="nearest")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Test 2 — Skin Health Assessment (5 scenarios × 180 days)")
    print("=" * 60)

    fig, axes = plt.subplots(5, 3, figsize=(18, 22))
    fig.suptitle("Skin Health Assessment — 5 Scenarios × 180 Days", fontsize=13, fontweight="bold")

    summary = []

    sc_bar = tqdm(enumerate(SCENARIOS), total=len(SCENARIOS),
                  desc="Scenarios", unit="scenario", ncols=70)
    for i, sc_fn in sc_bar:
        counts, temps, quality, title = sc_fn()
        sc_bar.set_postfix_str(title.split()[0])
        records = run_assessment(counts, temps, quality)
        n_alerts, n_abn = report(records, title)
        summary.append((title, n_alerts, n_abn))
        plot_scenario(records, title, axes[i])

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "test_2_assessment.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  {'Scenario':<40} {'Alerts':>6}  {'Abn days':>8}")
    print(f"  {'-'*57}")
    for title, n_alerts, n_abn in summary:
        print(f"  {title:<40} {n_alerts:>6}  {n_abn:>8}")


if __name__ == "__main__":
    main()
