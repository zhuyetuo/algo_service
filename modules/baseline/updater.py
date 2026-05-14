"""
Individual scratch baseline updater.

Update strategy (matches demo design):
- Normal days  → weight = NORMAL_W  (0.05)
- Abnormal days → weight = ABNORMAL_W (0.01)  ← slow permeation, not full exclusion
- Runs every 7 days via scheduler
- Uses past 30 normal days to recompute std and temp_coef
"""

import logging
import time

import numpy as np
from sqlalchemy import text

from config import settings
from db.client import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def _update_one(device_sn: str) -> None:
    async with AsyncSessionLocal() as db:
        # ── Fetch past 30 valid days ─────────────────────────────────────────
        rows_sql = text("""
            SELECT scratch_count, avg_temperature, is_abnormal
            FROM   pet_skin_health_daily
            WHERE  device_sn   = :sn
              AND  data_quality = 0
            ORDER BY stat_date_ts DESC
            LIMIT 30
        """)
        rows = (await db.execute(rows_sql, {"sn": device_sn})).fetchall()

        if not rows:
            return

        # ── Current baseline ─────────────────────────────────────────────────
        bl_sql = text("""
            SELECT baseline_mean, baseline_std, temp_coef, valid_days
            FROM   pet_skin_baseline
            WHERE  device_sn = :sn
        """)
        bl = (await db.execute(bl_sql, {"sn": device_sn})).fetchone()

        if bl:
            cur_mean     = float(bl.baseline_mean)
            cur_std      = float(bl.baseline_std)
            cur_coef     = float(bl.temp_coef)
            valid_days   = int(bl.valid_days)
        else:
            # Bootstrap: use today's scratch count as seed
            cur_mean     = float(rows[0].scratch_count)
            cur_std      = settings.baseline_std_floor
            cur_coef     = 0.0
            valid_days   = 0

        # ── Rolling exponential update ───────────────────────────────────────
        # Process oldest → newest (rows are DESC, so reverse)
        for row in reversed(rows):
            count    = float(row.scratch_count)
            is_abn   = bool(row.is_abnormal)
            weight   = settings.abnormal_day_weight if is_abn else settings.normal_day_weight
            cur_mean = cur_mean * (1 - weight) + count * weight
            valid_days += 0 if is_abn else 1

        # ── Recompute std from normal days (last 30) ─────────────────────────
        normal_counts = [
            float(r.scratch_count) for r in rows if not r.is_abnormal
        ]
        if len(normal_counts) >= 3:
            cur_std = max(float(np.std(normal_counts)), settings.baseline_std_floor)

        # ── Temperature coefficient via least squares ────────────────────────
        temp_pairs = [
            (float(r.avg_temperature), float(r.scratch_count))
            for r in rows
            if r.avg_temperature is not None and not r.is_abnormal
        ]
        if len(temp_pairs) >= 20:
            temps   = np.array([p[0] for p in temp_pairs])
            counts  = np.array([p[1] for p in temp_pairs])
            X       = np.column_stack([temps, np.ones(len(temps))])
            coefs, _, _, _ = np.linalg.lstsq(X, counts, rcond=None)
            cur_coef = float(np.clip(coefs[0], 0.0, 0.4))

        # ── Confidence 0-1, caps at 30 days ─────────────────────────────────
        confidence = min(valid_days / 30.0, 1.0)
        phase = (
            0 if valid_days < 3 else
            1 if valid_days < 14 else
            2 if valid_days < 30 else 3
        )

        now_ms = int(time.time() * 1000)

        upsert_sql = text("""
            INSERT INTO pet_skin_baseline
                (device_sn, baseline_mean, baseline_std, temp_coef,
                 valid_days, eval_phase, confidence,
                 last_updated_ts, created_at)
            VALUES
                (:sn, :mean, :std, :coef,
                 :valid_days, :phase, :confidence,
                 :now_ms, :now_ms)
            ON CONFLICT (device_sn) DO UPDATE SET
                baseline_mean    = EXCLUDED.baseline_mean,
                baseline_std     = EXCLUDED.baseline_std,
                temp_coef        = EXCLUDED.temp_coef,
                valid_days       = EXCLUDED.valid_days,
                eval_phase       = EXCLUDED.eval_phase,
                confidence       = EXCLUDED.confidence,
                last_updated_ts  = EXCLUDED.last_updated_ts
        """)
        await db.execute(upsert_sql, {
            "sn":         device_sn,
            "mean":       round(cur_mean, 2),
            "std":        round(cur_std, 2),
            "coef":       round(cur_coef, 3),
            "valid_days": min(valid_days, 30),
            "phase":      phase,
            "confidence": round(confidence, 2),
            "now_ms":     now_ms,
        })
        await db.commit()

        logger.info(
            "Baseline updated device=%s mean=%.2f std=%.2f coef=%.3f valid_days=%d",
            device_sn, cur_mean, cur_std, cur_coef, valid_days,
        )


async def run_baseline_update() -> None:
    """Update baselines for all devices that have daily assessment data."""
    async with AsyncSessionLocal() as db:
        devices_sql = text(
            "SELECT DISTINCT device_sn FROM pet_skin_health_daily"
        )
        devices = (await db.execute(devices_sql)).fetchall()

    for row in devices:
        try:
            await _update_one(row.device_sn)
        except Exception:
            logger.exception("Baseline update failed for device=%s", row.device_sn)
