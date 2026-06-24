"""
调度器任务配置。

三个任务：
  inference_cycle  — 每隔 FETCH_INTERVAL_MIN 分钟执行
                     从 TDengine 拉取 IMU 数据 → 行为事件 → pet_dog_behavior.d_{device_id}
  batch_assessment — 每天执行
                     汇总抓挠统计 → z-score → pet_dog_skin_assessment.d_{device_id}
  baseline_update  — 每天执行
                     重新计算个体基线 → pet_dog_scratch_baseline.pet_skin_baseline
"""

import asyncio
import time
from datetime import datetime, timezone as dt_tz
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiomysql
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import text
from sqlalchemy.exc import OperationalError as SAOperationalError

from loguru import logger

from config import settings
from db.client import AsyncSessionLocal
from db.tdengine import td_fetch, td_fetch_env
from modules.baseline.updater import run_baseline_update
from modules.assessment.evaluator import run_batch_assessment, assess_device
from modules.inference.model import BehaviorLabel, get_classifier

MAX_RETRIES = 3

_scheduler = BackgroundScheduler()
_main_loop: asyncio.AbstractEventLoop | None = None


def _run(coro_fn, *args, **kwargs):
    """从同步的 APScheduler 线程中运行异步函数。"""
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


def _day_start_utc_ms(ts_ms: int, tz_name: str | None) -> int:
    """返回 ts_ms 所在本地日期的零点对应 UTC 毫秒时间戳。"""
    if not tz_name or tz_name == "UTC":
        return (ts_ms // 86_400_000) * 86_400_000
    try:
        tz = ZoneInfo(tz_name)
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=dt_tz.utc).astimezone(tz)
        midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(midnight.timestamp() * 1000)
    except (ZoneInfoNotFoundError, Exception):
        return (ts_ms // 86_400_000) * 86_400_000


def _ts_to_local_str(ts_ms: int, tz_name: str | None, date_only: bool = False) -> str:
    """UTC 毫秒时间戳 → 用户本地时间字符串（"%Y-%m-%d %H:%M:%S" 或 "%Y-%m-%d"）。"""
    try:
        tz = ZoneInfo(tz_name) if tz_name and tz_name != "UTC" else dt_tz.utc
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=dt_tz.utc).astimezone(tz)
    except Exception:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=dt_tz.utc)
    return dt.strftime("%Y-%m-%d") if date_only else dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 调度器启动与停止
# ---------------------------------------------------------------------------

def start_scheduler():
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    _scheduler.add_job(
        _run(run_inference_cycle),
        IntervalTrigger(seconds=settings.fetch_interval_sec),
        id="inference_cycle",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.add_job(
        _run(run_batch_assessment),
        CronTrigger.from_crontab(settings.assessment_cron),
        id="batch_assessment",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.add_job(
        _run(run_baseline_update),
        CronTrigger.from_crontab(settings.baseline_update_cron),
        id="baseline_update",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info(
        "调度器已启动 — 推理每 {} 秒执行，评估 {}，基线更新 {}",
        settings.fetch_interval_sec,
        settings.assessment_cron,
        settings.baseline_update_cron,
    )


def stop_scheduler():
    _scheduler.shutdown(wait=False)
    logger.info("调度器已停止")


# ---------------------------------------------------------------------------
# 失败记录辅助函数
# ---------------------------------------------------------------------------

async def _record_failure(device_id: int, day_ts: int, error: Exception) -> None:
    now_ms = int(time.time() * 1000)
    error_msg = f"{type(error).__name__}: {error}"
    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            INSERT INTO processing_errors
                (device_id, day_ts, error_msg, retry_count, status, created_at, updated_at)
            VALUES
                (:did, :day_ts, :msg, 1, 'pending', :now, :now)
            ON DUPLICATE KEY UPDATE
                error_msg   = VALUES(error_msg),
                retry_count = processing_errors.retry_count + 1,
                status      = CASE
                                WHEN processing_errors.retry_count + 1 >= :max_retries
                                THEN 'abandoned'
                                ELSE 'pending'
                              END,
                updated_at  = VALUES(updated_at)
        """), {"did": device_id, "day_ts": day_ts, "msg": error_msg,
               "now": now_ms, "max_retries": MAX_RETRIES})
        await db.commit()


async def _mark_success(device_id: int, day_ts: int) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(text("""
            DELETE FROM processing_errors
            WHERE device_id = :did AND day_ts = :day_ts
        """), {"did": device_id, "day_ts": day_ts})
        await db.commit()


async def _retry_pending(clf, device_tz_map: dict[int, str],
                         device_sn_map: dict[int, str]) -> None:
    """重试所有 status='pending' 的失败记录。"""
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text("""
            SELECT device_id, day_ts FROM processing_errors
            WHERE status = 'pending'
            ORDER BY day_ts
        """))).fetchall()

    if not rows:
        return

    logger.info("发现 {} 条待重试记录", len(rows))
    now_ms = int(time.time() * 1000)
    for row in rows:
        device_id, day_ts = int(row.device_id), row.day_ts
        tz = device_tz_map.get(device_id)
        sn = device_sn_map.get(device_id, "")
        today_ts = _day_start_utc_ms(now_ms, tz)
        try:
            imu_rows = await asyncio.to_thread(td_fetch, sn, day_ts - 1)
            day_rows = [r for r in imu_rows
                        if _day_start_utc_ms(r["ts_ms"], tz) == day_ts]
            if not day_rows:
                logger.warning("重试 设备={} 日期={} ({}) 无数据，跳过", device_id, day_ts,
                               _ts_to_local_str(day_ts, tz, date_only=True))
                continue
            await _process_day(clf, device_id, day_ts, day_rows, today_ts, tz)
            await _mark_success(device_id, day_ts)
            logger.info("重试成功 设备={} 日期={} ({})", device_id, day_ts,
                        _ts_to_local_str(day_ts, tz, date_only=True))
        except Exception as e:
            logger.exception("重试仍失败 设备={} 日期={} ({})", device_id, day_ts,
                             _ts_to_local_str(day_ts, tz, date_only=True))
            await _record_failure(device_id, day_ts, e)


async def _write_behavior(clf, device_id: int, day_ts: int,
                          day_rows: list[dict],
                          user_timezone: str | None = None) -> int:
    """对一批 IMU 数据执行推理并将行为事件写入行为表，返回写入的事件数。"""
    tbl = f"{settings.pg_schema_behavior}.d_{device_id}"
    day_rows_sorted = sorted(day_rows, key=lambda r: r["ts_ms"])
    base_ts_ms = day_rows_sorted[0]["ts_ms"]

    data = np.array(
        [[r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"]]
         for r in day_rows_sorted],
        dtype=np.float32,
    )
    events = clf.predict(data, base_ts_ms)

    tz = user_timezone or "UTC"
    async with AsyncSessionLocal() as db:
        await db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
                id            BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
                ts_start      BIGINT        NOT NULL,
                ts_end        BIGINT        NOT NULL,
                behavior      SMALLINT      NOT NULL,
                duration_sec  DECIMAL(10,2) NOT NULL,
                confidence    DECIMAL(5,3)  NOT NULL,
                local_start   VARCHAR(24),
                local_end     VARCHAR(24),
                user_timezone VARCHAR(32),
                UNIQUE KEY uq_ts_start (ts_start)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
        await db.commit()
        for ev in events:
            btype = int(ev["behavior_type"])
            dur_sec = round((ev["end_time"] - ev["start_time"]) / 1000.0, 2)
            await db.execute(text(f"""
                INSERT IGNORE INTO {tbl}
                    (ts_start, ts_end, behavior, duration_sec, confidence,
                     local_start, local_end, user_timezone)
                VALUES
                    (:ts_start, :ts_end, :behavior, :duration_sec, :confidence,
                     :local_start, :local_end, :tz)
            """), {
                "ts_start":    ev["start_time"],
                "ts_end":      ev["end_time"],
                "behavior":    btype,
                "duration_sec": dur_sec,
                "confidence":  ev["confidence"],
                "local_start": _ts_to_local_str(ev["start_time"], tz),
                "local_end":   _ts_to_local_str(ev["end_time"],   tz),
                "tz":          tz,
            })
        await db.commit()

    return len(events)


async def _process_day(clf, device_id: int, day_ts: int,
                       day_rows: list[dict], today_ts: int,
                       user_timezone: str | None = None) -> None:
    """写入行为事件并立即评估，专供重试逻辑使用（该天数据已完整）。"""
    event_count = await _write_behavior(clf, device_id, day_ts, day_rows, user_timezone)
    logger.info("设备={} 日期={} ({}) 事件数={}", device_id, day_ts,
                _ts_to_local_str(day_ts, user_timezone, date_only=True), event_count)
    if day_ts < today_ts:
        async with AsyncSessionLocal() as db:
            await assess_device(db, device_id, day_ts, user_timezone)


async def _sync_env_for_device(device_id: int, device_sn: str,
                               user_timezone: str | None) -> None:
    """从 TDengine env_data 拉取环境数据（温湿度 + 体温），按本地日期聚合后写入 pet_dog_environment.d_{device_id}。"""
    tbl = f"{settings.pg_schema_environment}.d_{device_id}"

    async with AsyncSessionLocal() as db:
        row = (await db.execute(text("""
            SELECT last_env_ts FROM device_sync_state WHERE device_id = :did
        """), {"did": device_id})).fetchone()
    last_env_ts = int(row.last_env_ts) if row else 0

    def _group_by_day(rows, value_keys):
        groups: dict[int, dict[str, list]] = {}
        for r in rows:
            day = _day_start_utc_ms(r["ts_ms"], user_timezone)
            if day not in groups:
                groups[day] = {k: [] for k in value_keys}
            for k in value_keys:
                if r.get(k) is not None:
                    groups[day][k].append(r[k])
        return groups

    async with AsyncSessionLocal() as db:
        await db.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
                ts              BIGINT       PRIMARY KEY,
                env_temp        DECIMAL(5,2),
                env_humidity    DECIMAL(5,1),
                neck_temp       DECIMAL(5,2),
                local_date      VARCHAR(12),
                user_timezone   VARCHAR(32),
                created_at      BIGINT       NOT NULL,
                updated_at      BIGINT       NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
        await db.commit()

    total_rows = total_days = 0

    while True:
        env_rows = await asyncio.to_thread(td_fetch_env, device_sn, last_env_ts)
        if not env_rows:
            break
        by_day = _group_by_day(env_rows, ["env_temp", "env_humi", "neck_temp"])
        now_ms = int(time.time() * 1000)
        async with AsyncSessionLocal() as db:
            for day_ts, d in by_day.items():
                avg_env_temp  = round(sum(d["env_temp"])  / len(d["env_temp"]),  2) if d.get("env_temp")  else None
                avg_env_humi  = round(sum(d["env_humi"])  / len(d["env_humi"]),  1) if d.get("env_humi")  else None
                avg_neck_temp = round(sum(d["neck_temp"]) / len(d["neck_temp"]), 2) if d.get("neck_temp") else None
                await db.execute(text(f"""
                    INSERT INTO {tbl}
                        (ts, env_temp, env_humidity, neck_temp,
                         local_date, user_timezone, created_at, updated_at)
                    VALUES
                        (:ts, :env_temp, :env_humi, :neck_temp,
                         :local_date, :tz, :now, :now)
                    ON DUPLICATE KEY UPDATE
                        env_temp      = COALESCE(VALUES(env_temp),     env_temp),
                        env_humidity  = COALESCE(VALUES(env_humidity), env_humidity),
                        neck_temp     = COALESCE(VALUES(neck_temp),    neck_temp),
                        local_date    = VALUES(local_date),
                        user_timezone = VALUES(user_timezone),
                        updated_at    = VALUES(updated_at)
                """), {
                    "ts": day_ts,
                    "env_temp":  avg_env_temp,
                    "env_humi":  avg_env_humi,
                    "neck_temp": avg_neck_temp,
                    "local_date": _ts_to_local_str(day_ts, user_timezone, date_only=True),
                    "tz": user_timezone or "UTC",
                    "now": now_ms,
                })
            await db.commit()
        new_env_ts = max(r["ts_ms"] for r in env_rows)
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                UPDATE device_sync_state SET last_env_ts = :val, updated_at = :now
                WHERE device_id = :did
            """), {"val": new_env_ts, "now": now_ms, "did": device_id})
            await db.commit()
        total_rows += len(env_rows)
        total_days += len(by_day)
        last_env_ts = new_env_ts

    if total_rows:
        logger.info("环境数据同步 设备={} 共 {} 条 天数={}",
                    device_id, total_rows, total_days)
    else:
        logger.info("设备 {} 环境/体温数据无更新，跳过", device_id)


# ---------------------------------------------------------------------------
# 单设备 IMU 处理（供并发调用）
# ---------------------------------------------------------------------------

async def _process_device_imu(clf, device_id: int, device_sn: str,
                               user_tz: str | None, now_ms: int) -> None:
    """拉取并推理单台设备的 IMU 新数据，写入行为事件并触发评估。"""
    today_ts = _day_start_utc_ms(now_ms, user_tz)
    try:
        async with AsyncSessionLocal() as db:
            row = (await db.execute(text("""
                SELECT last_processed_ts FROM device_sync_state WHERE device_id = :did
            """), {"did": device_id})).fetchone()
        last_ts = int(row.last_processed_ts) if row else 0

        any_data = False
        failed = False
        pending_assess: set[int] = set()

        while not failed:
            imu_rows = await asyncio.to_thread(td_fetch, device_sn, last_ts)

            if not imu_rows:
                if not any_data:
                    logger.info("设备 {} 数据无更新，跳过本次推理", device_id)
                for day_ts in sorted(pending_assess):
                    if day_ts < today_ts:
                        try:
                            async with AsyncSessionLocal() as db:
                                await assess_device(db, device_id, day_ts, user_tz)
                            await _mark_success(device_id, day_ts)
                        except Exception as e:
                            logger.exception("设备 {} 日期 {} ({}) 评估失败", device_id,
                                            day_ts, _ts_to_local_str(day_ts, user_tz, date_only=True))
                            await _record_failure(device_id, day_ts, e)
                break

            any_data = True

            batch_min_day = _day_start_utc_ms(min(r["ts_ms"] for r in imu_rows), user_tz)
            for day_ts in sorted(list(pending_assess)):
                if day_ts < batch_min_day:
                    if day_ts < today_ts:
                        try:
                            async with AsyncSessionLocal() as db:
                                await assess_device(db, device_id, day_ts, user_tz)
                            await _mark_success(device_id, day_ts)
                        except Exception as e:
                            logger.exception("设备 {} 日期 {} ({}) 评估失败", device_id,
                                            day_ts, _ts_to_local_str(day_ts, user_tz, date_only=True))
                            await _record_failure(device_id, day_ts, e)
                    pending_assess.discard(day_ts)

            day_groups: dict[int, list[dict]] = {}
            for r in imu_rows:
                day_key = _day_start_utc_ms(r["ts_ms"], user_tz)
                day_groups.setdefault(day_key, []).append(r)

            max_ts = last_ts
            for day_ts, day_rows in sorted(day_groups.items()):
                try:
                    event_count = await _write_behavior(clf, device_id, day_ts, day_rows, user_tz)
                    logger.info("设备={} 日期={} ({}) 事件数={}", device_id,
                                day_ts, _ts_to_local_str(day_ts, user_tz, date_only=True), event_count)
                    pending_assess.add(day_ts)
                    max_ts = max(r["ts_ms"] for r in day_rows)
                except Exception as e:
                    logger.exception("设备 {} 日期 {} ({}) 推理失败，已记录待重试", device_id,
                                    day_ts, _ts_to_local_str(day_ts, user_tz, date_only=True))
                    await _record_failure(device_id, day_ts, e)
                    failed = True
                    break

            if max_ts > last_ts:
                last_ts = max_ts
                cur_ms = int(time.time() * 1000)
                async with AsyncSessionLocal() as db:
                    await db.execute(text("""
                        UPDATE device_sync_state
                        SET last_processed_ts = :ts, last_sync_at = :now, updated_at = :now
                        WHERE device_id = :did
                    """), {"ts": max_ts, "now": cur_ms, "did": device_id})
                    await db.commit()
            elif not failed:
                break

    except ConnectionError as e:
        logger.warning("设备 {} TDengine 无法连接，本次周期跳过 — {}", device_id, e)
    except Exception:
        logger.exception("设备 {} 推理周期失败", device_id)


# ---------------------------------------------------------------------------
# 推理周期
# ---------------------------------------------------------------------------

async def run_inference_cycle() -> None:
    """
    推理主循环：
    1. 从 device_bind_history 同步活跃绑定 → device_sync_state（含用户时区）
    2. 重试上次失败的任务
    3. 逐设备同步环境 + 颈温数据
    4. 逐设备处理 IMU 数据：写行为事件，完整天触发评估
    """
    logger.info("推理周期开始（fetch_interval={} 秒）", settings.fetch_interval_sec)

    try:
        clf = get_classifier()
    except FileNotFoundError:
        logger.warning("未找到模型文件 — 跳过本次推理周期")
        return

    now_ms = int(time.time() * 1000)

    # ── 1. 同步活跃设备绑定关系和用户时区 ────────────────────────────────
    async with AsyncSessionLocal() as db:
        try:
            bindings = (await db.execute(text(f"""
                SELECT dbh.device_id, d.device_sn, dbh.user_id,
                       COALESCE(u.timezone, 'UTC') AS timezone
                FROM {settings.biz_schema}.device_bind_history dbh
                JOIN {settings.biz_schema}.device d ON dbh.device_id = d.id
                JOIN {settings.biz_schema}.`user` u ON dbh.user_id = u.id
                WHERE dbh.bind_status = 1
            """))).fetchall()
        except (SAOperationalError, aiomysql.OperationalError) as e:
            await db.rollback()
            logger.warning("MySQL 无法连接 {}:{} — 跳过本次推理周期 ({})",
                           settings.db_host, settings.db_port, e.__class__.__name__)
            return
        except Exception:
            await db.rollback()
            logger.warning("device_bind_history 不可用，从 device_sync_state 读取已知设备")
            bindings = None

    if bindings is None:
        # fallback：使用 device_sync_state 中已知的设备和时区
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(text(
                "SELECT device_id, device_sn, user_id, user_timezone AS timezone FROM device_sync_state"
            ))).fetchall()
        bindings = rows

    if not bindings:
        logger.info("暂无活跃设备绑定，跳过本次推理周期")
        return

    # device_id → timezone / device_sn 映射，供本次周期使用
    device_tz_map: dict[int, str] = {
        int(b.device_id): (b.timezone or "UTC") for b in bindings
    }
    device_sn_map: dict[int, str] = {
        int(b.device_id): (b.device_sn or "") for b in bindings
    }

    async with AsyncSessionLocal() as db:
        for b in bindings:
            await db.execute(text("""
                INSERT INTO device_sync_state
                    (device_id, user_id, user_timezone, device_sn,
                     last_processed_ts, last_sync_at, updated_at)
                VALUES (:did, :uid, :tz, :sn, 0, :now, :now)
                ON DUPLICATE KEY UPDATE
                    user_id       = VALUES(user_id),
                    user_timezone = VALUES(user_timezone),
                    device_sn     = VALUES(device_sn),
                    updated_at    = VALUES(updated_at)
            """), {"did": int(b.device_id), "uid": int(b.user_id),
                   "tz": b.timezone or "UTC", "sn": b.device_sn or "",
                   "now": now_ms})
        await db.commit()

    devices = list(device_tz_map.keys())
    logger.info("本次周期活跃设备 {} 台: {}", len(devices), ", ".join(str(d) for d in devices))

    # ── 2. 重试上次失败的记录 ────────────────────────────────────────────
    await _retry_pending(clf, device_tz_map, device_sn_map)

    sem = asyncio.Semaphore(settings.device_concurrency)

    # ── 3. 并发同步环境 + 颈温数据 ───────────────────────────────────────
    async def _env_task(device_id: int) -> None:
        async with sem:
            try:
                sn = device_sn_map.get(device_id, "")
                await _sync_env_for_device(device_id, sn, device_tz_map.get(device_id))
            except ConnectionError as e:
                logger.warning("设备 {} TDengine 无法连接，环境同步跳过 — {}", device_id, e)
            except Exception:
                logger.exception("设备 {} 环境数据同步失败", device_id)

    await asyncio.gather(*[_env_task(d) for d in devices])

    # ── 4. 并发处理 IMU 新数据 ────────────────────────────────────────────
    async def _imu_task(device_id: int) -> None:
        async with sem:
            sn = device_sn_map.get(device_id, "")
            await _process_device_imu(clf, device_id, sn, device_tz_map.get(device_id), now_ms)

    await asyncio.gather(*[_imu_task(d) for d in devices])
