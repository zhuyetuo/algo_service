"""
调度器任务配置。

三个任务：
  inference_cycle  — 每隔 FETCH_INTERVAL_MIN 分钟执行
                     从 TDengine 拉取 IMU 数据 → 行为事件 → behavior.{device_sn}
  batch_assessment — 每天执行
                     汇总抓挠统计 → z-score → skin_assessment.{device_sn}
  baseline_update  — 每天执行
                     重新计算个体基线 → pet_skin_baseline
"""

import asyncio
import time

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import text

from loguru import logger

from config import settings
from db.client import AsyncSessionLocal
from db.tdengine import td_get_devices, td_fetch
from modules.baseline.updater import run_baseline_update
from modules.assessment.evaluator import run_batch_assessment, assess_device
from modules.inference.model import BehaviorLabel, get_classifier

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
    推理主循环：
    1. 从 TDengine 获取所有设备列表，同步到 device_sync_state
    2. 对每个设备，读取 last_processed_ts，拉取新 IMU 数据
    3. 无新数据则跳过；有数据则按 UTC 日期分组
    4. 每组数据跑 LightGBM，写入 behavior.{device_sn} 表
    5. 历史完整天（非今天）立即补跑 assess_device
    6. 更新 device_sync_state.last_processed_ts
    """
    logger.info("推理周期开始（fetch_interval=%d 分钟）", settings.fetch_interval_min)

    try:
        clf = get_classifier()
    except FileNotFoundError:
        logger.warning("未找到模型文件 — 跳过本次推理周期")
        return

    # 从 TDengine 拉取全量设备列表
    try:
        devices = await asyncio.to_thread(td_get_devices)
    except Exception:
        logger.exception("无法从 TDengine 获取设备列表")
        return

    if not devices:
        logger.debug("TDengine 中暂无设备数据")
        return

    now_ms = int(time.time() * 1000)

    # 将新设备同步到 device_sync_state（已存在的设备不更新）
    async with AsyncSessionLocal() as db:
        for sn in devices:
            await db.execute(text("""
                INSERT INTO device_sync_state (device_sn, last_processed_ts, last_sync_at, updated_at)
                VALUES (:sn, 0, :now, :now)
                ON CONFLICT (device_sn) DO NOTHING
            """), {"sn": sn, "now": now_ms})
        await db.commit()

    # 今天 UTC 零点毫秒时间戳，用于区分历史完整天与当天
    today_ts = (int(time.time() * 1000) // 86_400_000) * 86_400_000

    for device_sn in devices:
        try:
            # 读取该设备上次处理到的时间戳
            async with AsyncSessionLocal() as db:
                row = (await db.execute(text("""
                    SELECT last_processed_ts FROM device_sync_state WHERE device_sn = :sn
                """), {"sn": device_sn})).fetchone()
            last_ts = int(row.last_processed_ts) if row else 0

            # 从 TDengine 拉取新 IMU 数据
            imu_rows = await asyncio.to_thread(td_fetch, device_sn, last_ts)

            if not imu_rows:
                logger.debug("设备 %s 暂无新 IMU 数据", device_sn)
                continue

            # 按 UTC 日期零点分组（ts_ms // 86_400_000 * 86_400_000）
            day_groups: dict[int, list[dict]] = {}
            for r in imu_rows:
                day_key = (r["ts_ms"] // 86_400_000) * 86_400_000
                day_groups.setdefault(day_key, []).append(r)

            max_ts = max(r["ts_ms"] for r in imu_rows)

            for day_ts, day_rows in sorted(day_groups.items()):
                try:
                    # 按时间戳排序后构建 numpy 数组
                    day_rows_sorted = sorted(day_rows, key=lambda r: r["ts_ms"])
                    base_ts_ms = day_rows_sorted[0]["ts_ms"]

                    data = np.array(
                        [[r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"]]
                         for r in day_rows_sorted],
                        dtype=np.float32,
                    )

                    # 运行行为分类推理
                    events = clf.predict(data, base_ts_ms)

                    # 写入 behavior.{device_sn} 表（先建表）
                    async with AsyncSessionLocal() as db:
                        # 建表 SQL，每次写入前确保表存在
                        await db.execute(text(f"""
                            CREATE TABLE IF NOT EXISTS behavior.{device_sn} (
                                id           bigserial PRIMARY KEY,
                                ts_start     bigint        NOT NULL,
                                ts_end       bigint        NOT NULL,
                                behavior     smallint      NOT NULL,
                                duration_sec decimal(10,2) NOT NULL,
                                confidence   decimal(5,3)  NOT NULL
                            )
                        """))

                        for ev in events:
                            # behavior 字段映射：MOVEMENT=1, SLEEP=2, SCRATCH=3
                            btype = int(ev["behavior_type"])
                            dur_sec = round((ev["end_time"] - ev["start_time"]) / 1000.0, 2)
                            await db.execute(text(f"""
                                INSERT INTO behavior.{device_sn}
                                    (ts_start, ts_end, behavior, duration_sec, confidence)
                                VALUES
                                    (:ts_start, :ts_end, :behavior, :duration_sec, :confidence)
                                ON CONFLICT DO NOTHING
                            """), {
                                "ts_start":    ev["start_time"],
                                "ts_end":      ev["end_time"],
                                "behavior":    btype,
                                "duration_sec": dur_sec,
                                "confidence":  ev["confidence"],
                            })
                        await db.commit()

                    logger.info("设备=%s 日期=%d 事件数=%d", device_sn, day_ts, len(events))

                    # 历史完整天（非今天）立即触发皮肤评估
                    if day_ts < today_ts:
                        async with AsyncSessionLocal() as db:
                            await assess_device(db, device_sn, day_ts)

                except Exception:
                    logger.exception("设备 %s 日期 %d 推理失败", device_sn, day_ts)

            # 更新 device_sync_state.last_processed_ts
            now_ms = int(time.time() * 1000)
            async with AsyncSessionLocal() as db:
                await db.execute(text("""
                    UPDATE device_sync_state
                    SET last_processed_ts = :ts, last_sync_at = :now, updated_at = :now
                    WHERE device_sn = :sn
                """), {"ts": max_ts, "now": now_ms, "sn": device_sn})
                await db.commit()

        except Exception:
            logger.exception("设备 %s 推理周期失败", device_sn)
