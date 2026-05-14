import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config import settings
from modules.baseline.updater import run_baseline_update
from modules.assessment.evaluator import run_batch_assessment

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler()


def _run_async(coro_fn):
    """Wrap an async function so APScheduler (sync) can call it."""
    def wrapper():
        asyncio.run(coro_fn())
    return wrapper


def start_scheduler():
    # Fetch + infer every N minutes — driven by config, not hardcoded
    _scheduler.add_job(
        _run_async(run_inference_cycle),
        IntervalTrigger(minutes=settings.fetch_interval_min),
        id="inference_cycle",
        replace_existing=True,
    )

    _scheduler.add_job(
        _run_async(run_baseline_update),
        CronTrigger.from_crontab(settings.baseline_update_cron),
        id="baseline_update",
        replace_existing=True,
    )

    _scheduler.add_job(
        _run_async(run_batch_assessment),
        CronTrigger.from_crontab(settings.assessment_cron),
        id="batch_assessment",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        "Scheduler started — inference every %d min", settings.fetch_interval_min
    )


def stop_scheduler():
    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


# ---------------------------------------------------------------------------
# Inference cycle (fetch → infer → store)
# ---------------------------------------------------------------------------

async def run_inference_cycle():
    """
    1. Fetch IMU data for the last FETCH_INTERVAL_MIN minutes from the source
    2. Run behavior classification
    3. Persist results to DB
    """
    from db.client import AsyncSessionLocal
    from modules.inference.model import get_classifier
    import numpy as np

    logger.info("Inference cycle started (window=%d min)", settings.fetch_interval_min)

    # TODO: replace with real data fetch (device API / DB query)
    # data shape: (N, 6) — [ax, ay, az, gx, gy, gz]
    data: np.ndarray = await _fetch_imu_data()

    try:
        clf = get_classifier()
    except FileNotFoundError:
        logger.warning("Model not found, skipping inference cycle")
        return

    result = clf.predict_batch(data)
    logger.info("Inference result: %s", result)

    async with AsyncSessionLocal() as db:
        # TODO: persist result to behavior_log table
        pass


async def _fetch_imu_data():
    """Placeholder: return IMU data for the last fetch window."""
    import numpy as np
    # Replace with actual device/DB fetch
    n_samples = settings.fetch_interval_min * 60 * settings.imu_sample_rate
    return np.zeros((n_samples, 6), dtype=np.float32)
