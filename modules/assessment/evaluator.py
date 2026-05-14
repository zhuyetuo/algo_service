"""
Daily skin-health assessment.

Logic (matches demo_all.py design):
1. Aggregate today's scratch events from pet_behavior_record
2. Read individual baseline from pet_skin_baseline
3. Compute z-score with temperature correction at the z-score level
   (raw counts are NEVER modified)
4. Apply dynamic thresholds based on eval_phase
5. Track consecutive abnormal days
6. Write result to pet_skin_health_daily
"""

import logging
import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.client import AsyncSessionLocal, get_session
from modules.inference.model import BehaviorLabel

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Phase thresholds
# ---------------------------------------------------------------------------

@dataclass
class PhaseThreshold:
    z: float
    consec: int
    avg_z: float


def get_threshold(valid_days: int) -> PhaseThreshold | None:
    """Return None during warm-up (no assessment)."""
    if valid_days < 3:
        return None
    if valid_days < 14:
        return PhaseThreshold(
            settings.phase1_threshold_z,
            settings.phase1_threshold_consec,
            settings.phase1_threshold_avgz,
        )
    if valid_days < 30:
        return PhaseThreshold(
            settings.phase2_threshold_z,
            settings.phase2_threshold_consec,
            settings.phase2_threshold_avgz,
        )
    return PhaseThreshold(
        settings.phase3_threshold_z,
        settings.phase3_threshold_consec,
        settings.phase3_threshold_avgz,
    )


def eval_phase(valid_days: int) -> int:
    if valid_days < 3:
        return 0
    if valid_days < 14:
        return 1
    if valid_days < 30:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Daily assessment for a single device
# ---------------------------------------------------------------------------

async def assess_device(db: AsyncSession, device_sn: str, stat_date_ts: int) -> None:
    """
    Compute and upsert the daily skin-health row for one device.

    stat_date_ts : UTC ms timestamp of midnight (local day start) for this device,
                   provided by the caller (backend converts timezone).
    """
    day_end_ts = stat_date_ts + 86_400_000  # +24h in ms

    # ── 1. Aggregate today's scratch events ─────────────────────────────────
    scratch_sql = text("""
        SELECT start_time, end_time
        FROM   pet_behavior_record
        WHERE  device_sn      = :sn
          AND  behavior_type  = :btype
          AND  start_time    >= :day_start
          AND  start_time     < :day_end
        ORDER BY start_time
    """)
    rows = (await db.execute(scratch_sql, {
        "sn": device_sn,
        "btype": int(BehaviorLabel.SCRATCH),
        "day_start": stat_date_ts,
        "day_end":   day_end_ts,
    })).fetchall()

    scratch_count = len(rows)
    durations_ms  = [r.end_time - r.start_time for r in rows]
    scratch_dur   = sum(durations_ms)
    scratch_avg_dur = int(scratch_dur / scratch_count) if scratch_count else 0
    scratch_max_dur = max(durations_ms, default=0)

    # Night scratch: hours [22, 06) in local time
    # We receive stat_date_ts as local midnight, so:
    #   night_start = stat_date_ts - 2h  (previous evening 22:00)
    #   night_end   = stat_date_ts + 6h
    night_start_ts = stat_date_ts - 2 * 3_600_000
    night_end_ts   = stat_date_ts + 6 * 3_600_000
    night_count = sum(
        1 for r in rows
        if night_start_ts <= r.start_time < night_end_ts
    )

    # ── 2. Wear minutes (approximate from any behavior record) ───────────────
    wear_sql = text("""
        SELECT COALESCE(SUM(end_time - start_time), 0) AS wore_ms
        FROM   pet_behavior_record
        WHERE  device_sn   = :sn
          AND  start_time >= :day_start
          AND  start_time  < :day_end
    """)
    wore_ms = (await db.execute(wear_sql, {
        "sn": device_sn, "day_start": stat_date_ts, "day_end": day_end_ts,
    })).scalar() or 0
    wear_minutes = int(wore_ms / 60_000)

    # Mark as not-worn if below threshold
    data_quality = 0
    if wear_minutes < settings.min_wear_minutes:
        data_quality = 1  # treated as no-data day

    # ── 3. Read baseline ─────────────────────────────────────────────────────
    bl_sql = text("""
        SELECT baseline_mean, baseline_std, temp_coef, valid_days
        FROM   pet_skin_baseline
        WHERE  device_sn = :sn
    """)
    bl = (await db.execute(bl_sql, {"sn": device_sn})).fetchone()

    if bl is None:
        # No baseline yet — create initial row with today's data as seed
        valid_days   = 1
        bl_mean      = float(scratch_count)
        bl_std       = settings.baseline_std_floor
        temp_coef    = 0.0
    else:
        valid_days = bl.valid_days
        bl_mean    = float(bl.baseline_mean)
        bl_std     = max(float(bl.baseline_std), settings.baseline_std_floor)
        temp_coef  = float(bl.temp_coef)

    phase    = eval_phase(valid_days)
    thresh   = get_threshold(valid_days)

    # ── 4. Z-score with temperature correction at computation level ──────────
    # avg_temperature comes from external source (backend provides it).
    # For now we read it from the daily row if it was pre-filled, else None.
    avg_temp_sql = text("""
        SELECT avg_temperature FROM pet_skin_health_daily
        WHERE device_sn = :sn AND stat_date_ts = :ts
    """)
    existing = (await db.execute(avg_temp_sql, {
        "sn": device_sn, "ts": stat_date_ts,
    })).fetchone()
    avg_temperature = float(existing.avg_temperature) if existing and existing.avg_temperature else None

    # Temperature effect: remove from the deviation, not from the raw count
    ref_temp    = 20.0
    temp_effect = round(temp_coef * (avg_temperature - ref_temp), 2) if avg_temperature else 0.0

    zscore = None
    avg_zscore = None
    is_abnormal = False
    consec_abnormal = 0
    alert_triggered = False
    alert_reason = None

    if thresh is not None and data_quality == 0:
        # z-score: deviation minus temperature effect, divided by std
        raw_deviation = scratch_count - bl_mean
        adjusted_dev  = raw_deviation - temp_effect
        zscore = round(adjusted_dev / bl_std, 3)

        # Read recent N days to compute avg_zscore
        recent_sql = text("""
            SELECT zscore FROM pet_skin_health_daily
            WHERE  device_sn    = :sn
              AND  stat_date_ts  < :ts
              AND  data_quality  = 0
              AND  zscore IS NOT NULL
            ORDER BY stat_date_ts DESC
            LIMIT  :n
        """)
        recent_rows = (await db.execute(recent_sql, {
            "sn": device_sn, "ts": stat_date_ts,
            "n": thresh.consec - 1,
        })).fetchall()

        recent_zscores = [float(r.zscore) for r in recent_rows] + [zscore]
        avg_zscore = round(sum(recent_zscores) / len(recent_zscores), 3)

        is_abnormal = zscore > thresh.z

        # Consecutive abnormal days
        prev_sql = text("""
            SELECT consec_abnormal FROM pet_skin_health_daily
            WHERE  device_sn    = :sn
              AND  stat_date_ts  < :ts
              AND  data_quality  = 0
            ORDER BY stat_date_ts DESC
            LIMIT 1
        """)
        prev_row = (await db.execute(prev_sql, {
            "sn": device_sn, "ts": stat_date_ts,
        })).fetchone()
        prev_consec = int(prev_row.consec_abnormal) if prev_row else 0

        consec_abnormal = (prev_consec + 1) if is_abnormal else 0

        if (
            is_abnormal
            and consec_abnormal >= thresh.consec
            and avg_zscore >= thresh.avg_z
        ):
            alert_triggered = True
            alert_reason = (
                f"连续{consec_abnormal}天z-score>{thresh.z}，"
                f"均值z={avg_zscore}"
            )

    # ── 5. Upsert result ─────────────────────────────────────────────────────
    now_ms = int(time.time() * 1000)
    upsert_sql = text("""
        INSERT INTO pet_skin_health_daily (
            device_sn, stat_date_ts,
            scratch_count, scratch_duration, scratch_avg_dur, scratch_max_dur,
            night_scratch_count,
            avg_temperature,
            baseline_mean, baseline_std, temp_coef,
            temp_effect, zscore, avg_zscore,
            consec_abnormal,
            eval_phase, threshold_z, threshold_consec, threshold_avgz,
            valid_days,
            is_abnormal, alert_triggered, alert_reason,
            data_quality, wear_minutes,
            created_at, updated_at
        ) VALUES (
            :sn, :stat_date_ts,
            :scratch_count, :scratch_dur, :scratch_avg_dur, :scratch_max_dur,
            :night_count,
            :avg_temp,
            :bl_mean, :bl_std, :temp_coef,
            :temp_effect, :zscore, :avg_zscore,
            :consec_abnormal,
            :phase, :thresh_z, :thresh_consec, :thresh_avgz,
            :valid_days,
            :is_abnormal, :alert_triggered, :alert_reason,
            :data_quality, :wear_minutes,
            :now_ms, :now_ms
        )
        ON CONFLICT (device_sn, stat_date_ts) DO UPDATE SET
            scratch_count      = EXCLUDED.scratch_count,
            scratch_duration   = EXCLUDED.scratch_duration,
            scratch_avg_dur    = EXCLUDED.scratch_avg_dur,
            scratch_max_dur    = EXCLUDED.scratch_max_dur,
            night_scratch_count= EXCLUDED.night_scratch_count,
            temp_effect        = EXCLUDED.temp_effect,
            zscore             = EXCLUDED.zscore,
            avg_zscore         = EXCLUDED.avg_zscore,
            consec_abnormal    = EXCLUDED.consec_abnormal,
            is_abnormal        = EXCLUDED.is_abnormal,
            alert_triggered    = EXCLUDED.alert_triggered,
            alert_reason       = EXCLUDED.alert_reason,
            data_quality       = EXCLUDED.data_quality,
            wear_minutes       = EXCLUDED.wear_minutes,
            updated_at         = EXCLUDED.updated_at
    """)
    await db.execute(upsert_sql, {
        "sn": device_sn,
        "stat_date_ts": stat_date_ts,
        "scratch_count": scratch_count,
        "scratch_dur": scratch_dur,
        "scratch_avg_dur": scratch_avg_dur,
        "scratch_max_dur": scratch_max_dur,
        "night_count": night_count,
        "avg_temp": avg_temperature,
        "bl_mean": bl_mean,
        "bl_std": bl_std,
        "temp_coef": temp_coef,
        "temp_effect": temp_effect,
        "zscore": zscore,
        "avg_zscore": avg_zscore,
        "consec_abnormal": consec_abnormal,
        "phase": phase,
        "thresh_z": thresh.z if thresh else None,
        "thresh_consec": thresh.consec if thresh else None,
        "thresh_avgz": thresh.avg_z if thresh else None,
        "valid_days": valid_days,
        "is_abnormal": int(is_abnormal),
        "alert_triggered": int(alert_triggered),
        "alert_reason": alert_reason,
        "data_quality": data_quality,
        "wear_minutes": wear_minutes,
        "now_ms": now_ms,
    })
    await db.commit()

    if alert_triggered:
        logger.warning(
            "ALERT device=%s stat_date_ts=%d reason=%s",
            device_sn, stat_date_ts, alert_reason,
        )
    else:
        logger.info(
            "assessed device=%s phase=%d zscore=%s consec=%d alert=%s",
            device_sn, phase, zscore, consec_abnormal, alert_triggered,
        )


# ---------------------------------------------------------------------------
# Batch job — called by scheduler
# ---------------------------------------------------------------------------

async def run_batch_assessment(stat_date_ts: int | None = None) -> None:
    """
    Run daily assessment for all active devices.

    stat_date_ts : UTC ms of local midnight for today.
                   If None, uses current UTC midnight (suitable for single-timezone deploys).
    """
    if stat_date_ts is None:
        # Fallback: current UTC day midnight
        now = int(time.time())
        stat_date_ts = (now // 86400) * 86400 * 1000

    async with AsyncSessionLocal() as db:
        # Fetch all distinct devices that have data today
        devices_sql = text("""
            SELECT DISTINCT device_sn
            FROM pet_behavior_record
            WHERE start_time >= :day_start AND start_time < :day_end
        """)
        devices = (await db.execute(devices_sql, {
            "day_start": stat_date_ts,
            "day_end":   stat_date_ts + 86_400_000,
        })).fetchall()

    for row in devices:
        async with AsyncSessionLocal() as db:
            try:
                await assess_device(db, row.device_sn, stat_date_ts)
            except Exception:
                logger.exception("Assessment failed for device=%s", row.device_sn)


# ---------------------------------------------------------------------------
# REST endpoint — latest assessment report for one device
# ---------------------------------------------------------------------------

class AssessmentReport(BaseModel):
    device_sn: str
    stat_date_ts: int
    scratch_count: int
    zscore: float | None
    is_abnormal: bool
    alert_triggered: bool
    alert_reason: str | None
    eval_phase: int
    valid_days: int
    confidence: float   # baseline confidence 0-1


@router.get("/report/{device_sn}", response_model=AssessmentReport)
async def get_report(
    device_sn: str,
    db: AsyncSession = Depends(get_session),
):
    sql = text("""
        SELECT d.device_sn, d.stat_date_ts,
               d.scratch_count, d.zscore,
               d.is_abnormal, d.alert_triggered, d.alert_reason,
               d.eval_phase, d.valid_days,
               b.confidence
        FROM   pet_skin_health_daily d
        LEFT JOIN pet_skin_baseline b USING (device_sn)
        WHERE  d.device_sn = :sn
        ORDER BY d.stat_date_ts DESC
        LIMIT 1
    """)
    row = (await db.execute(sql, {"sn": device_sn})).fetchone()
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No assessment data yet")

    return AssessmentReport(
        device_sn=row.device_sn,
        stat_date_ts=row.stat_date_ts,
        scratch_count=row.scratch_count,
        zscore=row.zscore,
        is_abnormal=bool(row.is_abnormal),
        alert_triggered=bool(row.alert_triggered),
        alert_reason=row.alert_reason,
        eval_phase=row.eval_phase,
        valid_days=row.valid_days,
        confidence=float(row.confidence or 0),
    )
