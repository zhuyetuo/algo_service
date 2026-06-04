"""
调度器任务配置。

三个任务：
  inference_cycle  — 每隔 FETCH_INTERVAL_MIN 分钟执行
                     从 TDengine 拉取 IMU 数据 → 行为事件 → pet_dog_behavior.{device_sn}
  batch_assessment — 每天执行
                     汇总抓挠统计 → z-score → pet_dog_skin_assessment.{device_sn}
  baseline_update  — 每天执行
                     重新计算个体基线 → pet_dog_scratch_baseline.pet_skin_baseline
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
from db.tdengine import td_get_devices, td_fetch, td_fetch_env, td_fetch_neck_temp
from modules.baseline.updater import run_baseline_update
from modules.assessment.evaluator import run_batch_assessment, assess_device
from modules.inference.model import BehaviorLabel, get_classifier

MAX_RETRIES = 3

_scheduler = BackgroundScheduler()
_main_loop: asyncio.AbstractEventLoop | None = None


def _run(coro_fn, *args, **kwargs):
    """从同步的 APScheduler 线程中运行异步函数。

    必须在主事件循环上执行（asyncpg 连接池绑定到该循环），
    不能用 asyncio.run() 新建循环。
    """
    def wrapper():
        if _main_loop is None:
            logger.error("主事件循环未初始化，跳过任务 {}", coro_fn.__name__)
            return
        future = asyncio.run_coroutine_threadsafe(coro_fn(*args, **kwargs), _main_loop)
        try:
            future.result()
        except Exception:
            logger.exception("调度任务 {} 抛出异常", coro_fn.__name__)
    return wrapper


# ---------------------------------------------------------------------------
# 调度器启动与停止
# ---------------------------------------------------------------------------

def start_scheduler():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
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
        "调度器已启动 — 推理每 {} 分钟执行，评估 {}，基线更新 {}",
        settings.fetch_interval_min,
        settings.assessment_cron,
        settings.baseline_update_cron,
    )


def stop_scheduler():
    _scheduler.shutdown(wait=False)
    logger.info("调度器已停止")


# ---------------------------------------------------------------------------
# 失败记录辅助函数
# ---------------------------------------------------------------------------

async def _record_failure(device_sn: str, day_ts: int, error: Exception) -> None:
    """写入或更新 processing_errors，超过 MAX_RETRIES 次标记为 abandoned。"""
    now_ms = int(time.time() * 1000)
    error_msg = f"{type(error).__name__}: {error}"
    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            INSERT INTO processing_errors
                (device_sn, day_ts, error_msg, retry_count, status, created_at, updated_at)
            VALUES
                (:sn, :day_ts, :msg, 1, 'pending', :now, :now)
            ON CONFLICT (device_sn, day_ts) DO UPDATE SET
                error_msg   = EXCLUDED.error_msg,
                retry_count = processing_errors.retry_count + 1,
                status      = CASE
                                WHEN processing_errors.retry_count + 1 >= :max_retries
                                THEN 'abandoned'
                                ELSE 'pending'
                              END,
                updated_at  = EXCLUDED.updated_at
        """), {"sn": device_sn, "day_ts": day_ts, "msg": error_msg,
               "now": now_ms, "max_retries": MAX_RETRIES})
        await db.commit()


async def _mark_success(device_sn: str, day_ts: int) -> None:
    """任务成功时清除对应的失败记录（如果存在）。"""
    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            DELETE FROM processing_errors
            WHERE device_sn = :sn AND day_ts = :day_ts
        """), {"sn": device_sn, "day_ts": day_ts})
        await db.commit()


async def _retry_pending(clf, today_ts: int) -> None:
    """重试所有 status='pending' 的失败记录。"""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT device_sn, day_ts FROM processing_errors
            WHERE status = 'pending'
            ORDER BY day_ts
        """))).fetchall()

    if not rows:
        return

    logger.info("发现 {} 条待重试记录", len(rows))
    for row in rows:
        device_sn, day_ts = row.device_sn, row.day_ts
        try:
            last_ts = day_ts - 1
            imu_rows = await asyncio.to_thread(td_fetch, device_sn, last_ts)
            day_rows = [r for r in imu_rows
                        if (r["ts_ms"] // 86_400_000) * 86_400_000 == day_ts]
            if not day_rows:
                logger.warning("重试 设备={} 日期={} 无数据，跳过", device_sn, day_ts)
                continue

            await _process_day(clf, device_sn, day_ts, day_rows, today_ts)
            await _mark_success(device_sn, day_ts)
            logger.info("重试成功 设备={} 日期={}", device_sn, day_ts)
        except Exception as e:
            logger.exception("重试仍失败 设备={} 日期={}", device_sn, day_ts)
            await _record_failure(device_sn, day_ts, e)


async def _write_behavior(clf, device_sn: str, day_ts: int,
                          day_rows: list[dict]) -> int:
    """对一批 IMU 数据执行推理并将行为事件写入行为表，返回写入的事件数。

    同一天可能跨多个批次调用，每次追加写入，不做评估。
    ts_start 上的唯一约束保证重叠窗口不会产生重复行。
    """
    day_rows_sorted = sorted(day_rows, key=lambda r: r["ts_ms"])
    base_ts_ms = day_rows_sorted[0]["ts_ms"]

    data = np.array(
        [[r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"]]
         for r in day_rows_sorted],
        dtype=np.float32,
    )
    events = clf.predict(data, base_ts_ms)

    async with AsyncSessionLocal() as db:
        await db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {settings.pg_schema_behavior}.{device_sn} (
                id           bigserial PRIMARY KEY,
                ts_start     bigint        NOT NULL UNIQUE,
                ts_end       bigint        NOT NULL,
                behavior     smallint      NOT NULL,
                duration_sec decimal(10,2) NOT NULL,
                confidence   decimal(5,3)  NOT NULL
            )
        """))
        await db.commit()
        for ev in events:
            btype = int(ev["behavior_type"])
            dur_sec = round((ev["end_time"] - ev["start_time"]) / 1000.0, 2)
            await db.execute(text(f"""
                INSERT INTO {settings.pg_schema_behavior}.{device_sn}
                    (ts_start, ts_end, behavior, duration_sec, confidence)
                VALUES
                    (:ts_start, :ts_end, :behavior, :duration_sec, :confidence)
                ON CONFLICT (ts_start) DO NOTHING
            """), {
                "ts_start":     ev["start_time"],
                "ts_end":       ev["end_time"],
                "behavior":     btype,
                "duration_sec": dur_sec,
                "confidence":   ev["confidence"],
            })
        await db.commit()

    return len(events)


async def _process_day(clf, device_sn: str, day_ts: int,
                       day_rows: list[dict], today_ts: int) -> None:
    """写入行为事件并立即评估，专供重试逻辑使用（该天数据已完整）。"""
    event_count = await _write_behavior(clf, device_sn, day_ts, day_rows)
    logger.info("设备={} 日期={} 事件数={}", device_sn, day_ts, event_count)
    if day_ts < today_ts:
        async with AsyncSessionLocal() as db:
            await assess_device(db, device_sn, day_ts)


async def _sync_env_for_device(device_sn: str) -> None:
    """
    从 TDengine 拉取环境（env_temp/env_humi）和颈部体温（neck_temp）数据，
    按天聚合后写入 pet_dog_environment.{device_sn}。
    """
    # 读取两个数据源各自的断点
    async with AsyncSessionLocal() as db:
        row = (await db.execute(text("""
            SELECT last_env_ts, last_neck_temp_ts
            FROM device_sync_state WHERE device_sn = :sn
        """), {"sn": device_sn})).fetchone()
    last_env_ts       = int(row.last_env_ts)       if row else 0
    last_neck_temp_ts = int(row.last_neck_temp_ts) if row else 0

    # 按 UTC 日期零点聚合：每天取均值
    def _group_by_day(rows, value_keys):
        groups: dict[int, dict[str, list]] = {}
        for r in rows:
            day = (r["ts_ms"] // 86_400_000) * 86_400_000
            if day not in groups:
                groups[day] = {k: [] for k in value_keys}
            for k in value_keys:
                if r.get(k) is not None:
                    groups[day][k].append(r[k])
        return groups

    # 确保环境表存在（仅建一次）
    async with AsyncSessionLocal() as db:
        await db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {settings.pg_schema_environment}.{device_sn} (
                ts              bigint PRIMARY KEY,
                env_temp        decimal(5,2),
                env_humidity    decimal(5,1),
                neck_temp       decimal(5,2),
                created_at      bigint NOT NULL,
                updated_at      bigint NOT NULL
            )
        """))
        await db.commit()

    total_env = total_neck = total_days = 0

    # ── 循环拉取 env_raw，直到追上历史 ───────────────────────────────────
    while True:
        env_rows = await asyncio.to_thread(td_fetch_env, device_sn, last_env_ts)
        if not env_rows:
            break
        env_by_day = _group_by_day(env_rows, ["env_temp", "env_humi"])
        now_ms = int(time.time() * 1000)
        async with AsyncSessionLocal() as db:
            for day_ts, env_d in env_by_day.items():
                avg_env_temp = round(sum(env_d["env_temp"]) / len(env_d["env_temp"]), 2) \
                               if env_d.get("env_temp") else None
                avg_env_humi = round(sum(env_d["env_humi"]) / len(env_d["env_humi"]), 1) \
                               if env_d.get("env_humi") else None
                await db.execute(text(f"""
                    INSERT INTO {settings.pg_schema_environment}.{device_sn}
                        (ts, env_temp, env_humidity, neck_temp, created_at, updated_at)
                    VALUES (:ts, :env_temp, :env_humi, NULL, :now, :now)
                    ON CONFLICT (ts) DO UPDATE SET
                        env_temp     = COALESCE(EXCLUDED.env_temp,    {settings.pg_schema_environment}.{device_sn}.env_temp),
                        env_humidity = COALESCE(EXCLUDED.env_humidity, {settings.pg_schema_environment}.{device_sn}.env_humidity),
                        updated_at   = EXCLUDED.updated_at
                """), {"ts": day_ts, "env_temp": avg_env_temp, "env_humi": avg_env_humi, "now": now_ms})
            await db.commit()
        new_env_ts = max(r["ts_ms"] for r in env_rows)
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                UPDATE device_sync_state SET last_env_ts = :val, updated_at = :now
                WHERE device_sn = :sn
            """), {"val": new_env_ts, "now": now_ms, "sn": device_sn})
            await db.commit()
        total_env += len(env_rows)
        total_days += len(env_by_day)
        last_env_ts = new_env_ts

    # ── 循环拉取 neck_temp_raw，直到追上历史 ─────────────────────────────
    while True:
        neck_rows = await asyncio.to_thread(td_fetch_neck_temp, device_sn, last_neck_temp_ts)
        if not neck_rows:
            break
        neck_by_day = _group_by_day(neck_rows, ["neck_temp"])
        now_ms = int(time.time() * 1000)
        async with AsyncSessionLocal() as db:
            for day_ts, neck_d in neck_by_day.items():
                avg_neck_temp = round(sum(neck_d["neck_temp"]) / len(neck_d["neck_temp"]), 2) \
                                if neck_d.get("neck_temp") else None
                await db.execute(text(f"""
                    INSERT INTO {settings.pg_schema_environment}.{device_sn}
                        (ts, env_temp, env_humidity, neck_temp, created_at, updated_at)
                    VALUES (:ts, NULL, NULL, :neck_temp, :now, :now)
                    ON CONFLICT (ts) DO UPDATE SET
                        neck_temp  = COALESCE(EXCLUDED.neck_temp, {settings.pg_schema_environment}.{device_sn}.neck_temp),
                        updated_at = EXCLUDED.updated_at
                """), {"ts": day_ts, "neck_temp": avg_neck_temp, "now": now_ms})
            await db.commit()
        new_neck_ts = max(r["ts_ms"] for r in neck_rows)
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                UPDATE device_sync_state SET last_neck_temp_ts = :val, updated_at = :now
                WHERE device_sn = :sn
            """), {"val": new_neck_ts, "now": now_ms, "sn": device_sn})
            await db.commit()
        total_neck += len(neck_rows)
        last_neck_temp_ts = new_neck_ts

    if total_env or total_neck:
        logger.info("环境数据同步 设备={} env={} 条 neck_temp={} 条 天数={}",
                    device_sn, total_env, total_neck, total_days)
    else:
        logger.info("设备 {} 环境/体温数据无更新，跳过", device_sn)


# ---------------------------------------------------------------------------
# 推理周期
# ---------------------------------------------------------------------------

async def run_inference_cycle() -> None:
    """
    推理主循环：
    1. 重试上次失败的任务（status='pending'）
    2. 从 TDengine 获取所有设备列表，同步到 device_sync_state
    3. 对每个设备，读取 last_processed_ts，拉取新 IMU 数据
    4. 无新数据则跳过；有数据则按 UTC 日期分组
    5. 每组数据跑 LightGBM，写入 pet_dog_behavior.{device_sn} 表
    6. 历史完整天（非今天）立即补跑 assess_device
    7. 成功则推进 last_processed_ts；失败则写入 processing_errors
    """
    logger.info("推理周期开始（fetch_interval={} 分钟）", settings.fetch_interval_min)

    try:
        clf = get_classifier()
    except FileNotFoundError:
        logger.warning("未找到模型文件 — 跳过本次推理周期")
        return

    today_ts = (int(time.time() * 1000) // 86_400_000) * 86_400_000

    # ── 1. 先重试上次失败的记录 ──────────────────────────────────────────
    await _retry_pending(clf, today_ts)

    # ── 2. 拉取 TDengine 设备列表 ────────────────────────────────────────
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

    # ── 3. 逐设备同步环境 + 颈温数据 ────────────────────────────────────
    for device_sn in devices:
        try:
            await _sync_env_for_device(device_sn)
        except Exception:
            logger.exception("设备 {} 环境数据同步失败", device_sn)

    # ── 4. 逐设备处理 IMU 新数据 ─────────────────────────────────────────
    for device_sn in devices:
        try:
            async with AsyncSessionLocal() as db:
                row = (await db.execute(text("""
                    SELECT last_processed_ts FROM device_sync_state WHERE device_sn = :sn
                """), {"sn": device_sn})).fetchone()
            last_ts = int(row.last_processed_ts) if row else 0

            any_data = False
            failed = False
            # 已写入行为事件但尚未评估的日期集合（等到下一批确认该天数据已完整再评估）
            pending_assess: set[int] = set()

            while not failed:
                imu_rows = await asyncio.to_thread(td_fetch, device_sn, last_ts)

                if not imu_rows:
                    if not any_data:
                        logger.info("设备 {} 数据无更新，跳过本次推理", device_sn)
                    # 所有数据已拉完，对剩余待评估天做最终评估
                    for day_ts in sorted(pending_assess):
                        if day_ts < today_ts:
                            try:
                                async with AsyncSessionLocal() as db:
                                    await assess_device(db, device_sn, day_ts)
                                await _mark_success(device_sn, day_ts)
                            except Exception as e:
                                logger.exception("设备 {} 日期 {} 评估失败", device_sn, day_ts)
                                await _record_failure(device_sn, day_ts, e)
                    break

                any_data = True

                # 本批次最早时间戳所在的日：此日之前的所有天数据已确认完整，可评估
                batch_min_day = (min(r["ts_ms"] for r in imu_rows) // 86_400_000) * 86_400_000
                for day_ts in sorted(list(pending_assess)):
                    if day_ts < batch_min_day:
                        if day_ts < today_ts:
                            try:
                                async with AsyncSessionLocal() as db:
                                    await assess_device(db, device_sn, day_ts)
                                await _mark_success(device_sn, day_ts)
                            except Exception as e:
                                logger.exception("设备 {} 日期 {} 评估失败", device_sn, day_ts)
                                await _record_failure(device_sn, day_ts, e)
                        pending_assess.discard(day_ts)

                # 按 UTC 日期零点分组，写入行为事件
                day_groups: dict[int, list[dict]] = {}
                for r in imu_rows:
                    day_key = (r["ts_ms"] // 86_400_000) * 86_400_000
                    day_groups.setdefault(day_key, []).append(r)

                max_ts = last_ts
                for day_ts, day_rows in sorted(day_groups.items()):
                    try:
                        event_count = await _write_behavior(clf, device_sn, day_ts, day_rows)
                        logger.info("设备={} 日期={} 事件数={}", device_sn, day_ts, event_count)
                        pending_assess.add(day_ts)
                        max_ts = max(r["ts_ms"] for r in day_rows)
                    except Exception as e:
                        logger.exception("设备 {} 日期 {} 推理失败，已记录待重试", device_sn, day_ts)
                        await _record_failure(device_sn, day_ts, e)
                        failed = True
                        break

                # 推进断点，供下一批次使用
                if max_ts > last_ts:
                    last_ts = max_ts
                    now_ms = int(time.time() * 1000)
                    async with AsyncSessionLocal() as db:
                        await db.execute(text("""
                            UPDATE device_sync_state
                            SET last_processed_ts = :ts, last_sync_at = :now, updated_at = :now
                            WHERE device_sn = :sn
                        """), {"ts": max_ts, "now": now_ms, "sn": device_sn})
                        await db.commit()
                elif not failed:
                    break

        except Exception:
            logger.exception("设备 {} 推理周期失败", device_sn)
