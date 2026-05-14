import asyncio
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import settings
from modules.baseline.updater import run_baseline_update
from modules.assessment.evaluator import run_batch_assessment

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler()


def _run_async(coro):
    asyncio.run(coro)


def start_scheduler():
    _scheduler.add_job(
        _run_async,
        CronTrigger.from_crontab(settings.baseline_update_cron),
        args=[run_baseline_update()],
        id="baseline_update",
        replace_existing=True,
    )
    _scheduler.add_job(
        _run_async,
        CronTrigger.from_crontab(settings.assessment_cron),
        args=[run_batch_assessment()],
        id="batch_assessment",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler():
    _scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")
