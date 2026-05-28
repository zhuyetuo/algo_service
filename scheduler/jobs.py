"""
调度器任务配置。

三个任务：
  inference_cycle  — 每隔 FETCH_INTERVAL_MIN 分钟执行
                     拉取未处理的 IMU 数据 → 行为事件 → pet_behavior_record
  batch_assessment — 每天 03:00 UTC 执行
                     汇总抓挠统计 → z-score → pet_skin_health_daily
  baseline_update  — 每天 02:00 UTC 执行
                     重新计算个体基线 → pet_skin_baseline
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
    """从同步的 APScheduler 线程中运行异步函数。"""
    def wrapper():
        asyncio.run(coro_fn(*args, **kwargs))
    return wrapper


# ---------------------------------------------------------------------------
# 调度器启动与停止
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
        "调度器已启动 — 推理每 %d 分钟执行，评估 %s，基线更新 %s",
        settings.fetch_interval_min,
        settings.assessment_cron,
        settings.baseline_update_cron,
    )


def stop_scheduler():
    _scheduler.shutdown(wait=False)
    logger.info("调度器已停止")


# ---------------------------------------------------------------------------
# 推理周期
# ---------------------------------------------------------------------------

async def run_inference_cycle() -> None:
    """
    1. 从 algo_input（或等效数据源）拉取未处理的 IMU 行
    2. 对每个设备运行行为分类
    3. 将行为事件持久化到 pet_behavior_record
    4. 将 IMU 行标记为已处理
    """
    logger.info("推理周期开始（fetch_interval=%d 分钟）", settings.fetch_interval_min)

    try:
        clf = get_classifier()
    except FileNotFoundError:
        logger.warning("未找到模型文件 — 跳过本次推理周期")
        return

    async with AsyncSessionLocal() as db:
        # 按设备分组拉取未处理的 IMU 数据，表名和列名需与实际输入 schema 匹配
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
            logger.warning("imu_raw_data 表不可访问 — 使用占位数据")
            rows = []

    if not rows:
        logger.debug("未找到未处理的 IMU 数据")
        return

    # 按设备分组
    from collections import defaultdict
    device_rows: dict[str, list] = defaultdict(list)
    for row in rows:
        device_rows[row.device_sn].append(row)

    processed_ids = []

    for device_sn, device_data in device_rows.items():
        try:
            # 构建 numpy 数组，imu_data 为 JSON 格式：{"ax":…,"ay":…,"az":…,"gx":…,"gy":…,"gz":…}
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
            logger.info("设备=%s 事件数=%d", device_sn, len(events))

        except Exception:
            logger.exception("设备 %s 推理失败", device_sn)

    # 批量标记为已处理
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
                logger.warning("无法将 IMU 行标记为已处理")
