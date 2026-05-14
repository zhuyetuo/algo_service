"""
Scheduler wiring.

Three jobs:
  inference_cycle  — every FETCH_INTERVAL_MIN minutes
                     fetch unprocessed IMU data → behavior events → pet_behavior_record
  batch_assessment — daily at 03:00 UTC
                     aggregate scratch stats → z-score → pet_skin_health_daily
  baseline_update  — every 7 days at 02:00 UTC
                     recompute individual baselines → pet_skin_baseline
"""

import asyncio
import logging
import time

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import text

from config import settings
from db.client import AsyncSessionLocal
from modules.baseline.updater import run_baseline_update
from modules.assessment.evaluator import run_batch_assessment
from modules.inference.model import BehaviorLabel, get_classifier

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler()


def _run(coro_fn, *args, **kwargs):
    """Run an async function from a sync APScheduler thread."""
    def wrapper():
        asyncio.run(coro_fn(*args, **kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def start_scheduler():
    _scheduler.add_job(
        _run(run_inference_cycle),
        IntervalTrigger(minutes=settings.fetch_interval_min),
        id="inference_cycle",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run(run_batch_assessment),
        CronTrigger.from_crontab(settings.assessment_cron),
        id="batch_assessment",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run(run_baseline_update),
        CronTrigger.from_crontab(settings.baseline_update_cron),
        id="baseline_update",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info(
        "Scheduler started — inference every %d min, assessment %s, baseline %s",
        settings.fetch_interval_min,
        settings.assessment_cron,
        settings.baseline_update_cron,
    )


def stop_scheduler():
    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# Inference cycle
# ---------------------------------------------------------------------------

async def run_inference_cycle() -> None:
    """
    1. Fetch unprocessed IMU rows from algo_input (or equivalent source)
    2. For each device, run behavior classification
    3. Persist behavior events to pet_behavior_record
    4. Mark IMU rows as processed
    """
    logger.info("Inference cycle started (fetch_interval=%d min)", settings.fetch_interval_min)

    try:
        clf = get_classifier()
    except FileNotFoundError:
        logger.warning("Model file not found — skipping inference cycle")
        return

    async with AsyncSessionLocal() as db:
        # Fetch unprocessed IMU data grouped by device
        # Adjust the table/column names to match your actual input schema
        fetch_sql = text("""
            SELECT id, device_sn, data_ts, imu_data
            FROM   imu_raw_data
            WHERE  algo_processed = 0
            ORDER  BY device_sn, data_ts
            LIMIT  10000
        """)
        try:
            rows = (await db.execute(fetch_sql)).fetchall()
        except Exception:
            logger.warning("imu_raw_data table not accessible — using placeholder data")
            rows = []

    if not rows:
        logger.debug("No unprocessed IMU data found")
        return

    # Group by device
    from collections import defaultdict
    device_rows: dict[str, list] = defaultdict(list)
    for row in rows:
        device_rows[row.device_sn].append(row)

    processed_ids = []

    for device_sn, device_data in device_rows.items():
        try:
            # Build numpy array — imu_data is JSON: {"ax":…,"ay":…,"az":…,"gx":…,"gy":…,"gz":…}
            samples = sorted(device_data, key=lambda r: r.data_ts)
            base_ts_ms = samples[0].data_ts

            data = np.array(
                [[
                    r.imu_data["ax"], r.imu_data["ay"], r.imu_data["az"],
                    r.imu_data["gx"], r.imu_data["gy"], r.imu_data["gz"],
                ] for r in samples],
                dtype=np.float32,
            )

            events = clf.predict(data, base_ts_ms)

            async with AsyncSessionLocal() as db:
                for ev in events:
                    insert_sql = text("""
                        INSERT INTO pet_behavior_record
                            (device_sn, behavior_type, behavior_detail,
                             start_time, end_time, confidence)
                        VALUES
                            (:sn, :btype, NULL, :start, :end, :conf)
                        ON CONFLICT (device_sn, start_time) DO NOTHING
                    """)
                    await db.execute(insert_sql, {
                        "sn":    device_sn,
                        "btype": ev["behavior_type"],
                        "start": ev["start_time"],
                        "end":   ev["end_time"],
                        "conf":  ev["confidence"],
                    })
                await db.commit()

            processed_ids.extend([r.id for r in samples])
            logger.info("device=%s events=%d", device_sn, len(events))

        except Exception:
            logger.exception("Inference failed for device=%s", device_sn)

    # Mark as processed in batch
    if processed_ids:
        now_ms = int(time.time() * 1000)
        async with AsyncSessionLocal() as db:
            mark_sql = text("""
                UPDATE imu_raw_data
                SET    algo_processed = 1, algo_processed_at = :now
                WHERE  id = ANY(:ids)
            """)
            try:
                await db.execute(mark_sql, {"now": now_ms, "ids": processed_ids})
                await db.commit()
            except Exception:
                logger.warning("Could not mark IMU rows as processed")
