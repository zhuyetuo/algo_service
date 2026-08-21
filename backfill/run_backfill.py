#!/usr/bin/env python3
"""
离线回补：把 TDengine 里已有的历史 IMU 数据跑一遍推理，结果写入 MySQL 行为表。

设备端链路还没打通时用这个先把数据流验证起来——指定一个日期（或日期区间），
脚本会取出该区间内每台设备的 IMU 原始数据，按用户本地日期分组，逐天推理，
写入 pet_dog_behavior.d_{device_id}，可选再跑一遍当日皮肤评估。

用法见 backfill/README.md，或 python backfill/run_backfill.py --help
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timedelta, timezone as dt_tz
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from loguru import logger
from sqlalchemy import text

from config import settings
from db.client import AsyncSessionLocal
from db.tdengine import td_fetch_range, td_get_devices, td_device_span
from modules.assessment.evaluator import assess_device
from modules.inference.model import get_classifier
from scheduler.jobs import _write_behavior, _day_start_utc_ms, _ts_to_local_str, _BEHAVIOR_ZH


# ---------------------------------------------------------------------------
# 时间解析
# ---------------------------------------------------------------------------

def _parse_day(s: str) -> datetime:
    """'2026-08-19' → 当天 00:00 的 naive datetime。"""
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"日期格式错误：{s}（应为 YYYY-MM-DD）")


def _day_bounds_utc_ms(day: datetime, tz_name: str) -> tuple[int, int]:
    """把"某个本地日期"换算成 UTC 毫秒区间 [start, end)。"""
    try:
        tz = ZoneInfo(tz_name or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        tz = dt_tz.utc
    start = day.replace(tzinfo=tz)
    end = start + timedelta(days=1)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


# ---------------------------------------------------------------------------
# 设备清单
# ---------------------------------------------------------------------------

async def _load_devices(device_filter: set[int] | None) -> list[dict]:
    """
    读取设备清单：优先用业务库的活跃绑定（能拿到 bind_id 和用户时区），
    业务库不可用时退回 device_sync_state。
    """
    async with AsyncSessionLocal() as db:
        try:
            rows = (await db.execute(text(f"""
                SELECT dbh.bind_id, dbh.device_id, d.device_sn,
                       COALESCE(u.timezone, 'UTC') AS timezone
                FROM {settings.biz_schema}.device_bind_history dbh
                JOIN {settings.biz_schema}.device d ON dbh.device_id = d.id
                JOIN {settings.biz_schema}.`user` u ON dbh.user_id = u.id
                WHERE dbh.bind_status = 1
            """))).fetchall()
            source = "device_bind_history"
        except Exception:
            await db.rollback()
            rows = (await db.execute(text("""
                SELECT bind_id, device_id, device_sn, user_timezone AS timezone
                FROM device_sync_state
            """))).fetchall()
            source = "device_sync_state"

    devices = [{
        "bind_id":   int(r.bind_id) if getattr(r, "bind_id", None) else None,
        "device_id": int(r.device_id),
        "device_sn": (r.device_sn or "").strip(),
        "timezone":  r.timezone or "UTC",
    } for r in rows if (r.device_sn or "").strip()]

    if device_filter:
        devices = [d for d in devices if d["device_id"] in device_filter]

    logger.info("设备清单来源={} 共 {} 台", source, len(devices))
    return devices


# ---------------------------------------------------------------------------
# 单设备回补
# ---------------------------------------------------------------------------

async def _backfill_device(clf, dev: dict, day_from: datetime, day_to: datetime,
                           dry_run: bool, do_assess: bool) -> dict:
    """回补单台设备在 [day_from, day_to] 本地日期区间内的数据，返回统计结果。"""
    did, sn, tz = dev["device_id"], dev["device_sn"], dev["timezone"]
    start_ms, _ = _day_bounds_utc_ms(day_from, tz)
    _, end_ms   = _day_bounds_utc_ms(day_to, tz)

    stat = {"device_id": did, "device_sn": sn, "timezone": tz,
            "rows": 0, "days": 0, "events": 0, "behaviors": {}, "error": None}

    try:
        rows = await asyncio.to_thread(td_fetch_range, sn, start_ms, end_ms)
    except Exception as e:
        stat["error"] = f"TDengine 拉取失败: {e}"
        logger.error("设备 {} ({}) {}", did, sn, stat["error"])
        return stat

    stat["rows"] = len(rows)
    if not rows:
        logger.warning("设备 {} ({}) 区间内无 IMU 数据", did, sn)
        return stat

    # 按用户本地日期分组，与线上推理周期的分组口径保持一致
    day_groups: dict[int, list[dict]] = {}
    for r in rows:
        day_groups.setdefault(_day_start_utc_ms(r["ts_ms"], tz), []).append(r)
    stat["days"] = len(day_groups)

    for day_ts, day_rows in sorted(day_groups.items()):
        day_label = _ts_to_local_str(day_ts, tz, date_only=True)
        try:
            if dry_run:
                events = _predict_only(clf, day_rows, did)
                n = len(events)
                for ev in events:
                    name = _BEHAVIOR_ZH.get(int(ev["behavior_type"]), "未知")
                    stat["behaviors"][name] = stat["behaviors"].get(name, 0) + 1
            else:
                n = await _write_behavior(clf, did, day_ts, day_rows, tz, dev["bind_id"])
                for name, c in (await _count_behaviors(did, day_ts, tz)).items():
                    stat["behaviors"][name] = stat["behaviors"].get(name, 0) + c
            stat["events"] += n
            logger.info("设备={} 日期={} 采样点={} 事件数={}{}",
                        did, day_label, len(day_rows), n, "  [dry-run 未写库]" if dry_run else "")

            if do_assess and not dry_run:
                async with AsyncSessionLocal() as db:
                    await assess_device(db, did, day_ts, tz, dev["bind_id"])
                logger.info("设备={} 日期={} 已完成皮肤评估", did, day_label)
        except Exception as e:
            stat["error"] = f"{day_label} 处理失败: {e}"
            logger.exception("设备 {} 日期 {} 回补失败", did, day_label)

    return stat


def _predict_only(clf, day_rows: list[dict], device_id: int) -> list[dict]:
    """dry-run 用：只推理不写库。"""
    rows_sorted = sorted(day_rows, key=lambda r: r["ts_ms"])
    data = np.array([[r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"]]
                     for r in rows_sorted], dtype=np.float32)
    return clf.predict(data, rows_sorted[0]["ts_ms"], device_id=device_id)


async def _count_behaviors(device_id: int, day_ts: int, tz: str) -> dict[str, int]:
    """统计某天写入的各类行为事件数，用于汇总展示。"""
    tbl = f"{settings.pg_schema_behavior}.d_{device_id}"
    day_end = day_ts + 86_400_000
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text(f"""
            SELECT behavior_label, COUNT(*) AS c FROM {tbl}
            WHERE ts_start >= :a AND ts_start < :b GROUP BY behavior_label
        """), {"a": day_ts, "b": day_end})).fetchall()
    return {(r.behavior_label or "未知"): int(r.c) for r in rows}


# ---------------------------------------------------------------------------
# 子命令：列出 TDengine 中的设备
# ---------------------------------------------------------------------------

async def _cmd_list_devices() -> None:
    sns = await asyncio.to_thread(td_get_devices)
    if not sns:
        print("TDengine 超级表中没有任何设备数据")
        return
    print(f"\nTDengine {settings.td_database}.{settings.td_supertable} 共 {len(sns)} 台设备：\n")
    print(f"  {'device_sn':<24} {'采样点数':>10}  数据时间范围（UTC）")
    print(f"  {'-' * 70}")
    for sn in sns:
        span = await asyncio.to_thread(td_device_span, sn)
        if not span:
            print(f"  {sn:<24} {'0':>10}  —")
            continue
        f = datetime.fromtimestamp(span["first_ts"] / 1000, tz=dt_tz.utc)
        l = datetime.fromtimestamp(span["last_ts"]  / 1000, tz=dt_tz.utc)
        print(f"  {sn:<24} {span['count']:>10}  {f:%Y-%m-%d %H:%M} → {l:%Y-%m-%d %H:%M}")
    print()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def _run(args) -> int:
    if args.list_devices:
        await _cmd_list_devices()
        return 0

    if not args.date and not args.start:
        raise SystemExit("请指定 --date 或 --start/--end，或用 --list-devices 查看可用数据")

    day_from = _parse_day(args.date or args.start)
    day_to   = _parse_day(args.end) if args.end else day_from
    if day_to < day_from:
        raise SystemExit("--end 早于 --start")

    clf = get_classifier()

    device_filter = None
    if args.devices:
        device_filter = {int(x) for x in args.devices.split(",") if x.strip()}
    devices = await _load_devices(device_filter)
    if not devices:
        logger.warning("没有匹配的设备，退出")
        return 1

    logger.info("回补区间：{} → {}（各设备按自己的用户时区解释该日期）",
                day_from.date(), day_to.date())
    if args.dry_run:
        logger.warning("dry-run 模式：只推理、不写库")

    t0 = time.time()
    stats = []
    for dev in devices:
        stats.append(await _backfill_device(clf, dev, day_from, day_to,
                                            args.dry_run, args.assess))

    _print_summary(stats, time.time() - t0, args.dry_run)
    return 0 if all(s["error"] is None for s in stats) else 1


def _print_summary(stats: list[dict], elapsed: float, dry_run: bool) -> None:
    print("\n" + "=" * 78)
    print(f"回补汇总{'（dry-run，未写库）' if dry_run else ''}    耗时 {elapsed:.1f}s")
    print("=" * 78)
    print(f"  {'设备ID':>6}  {'device_sn':<20} {'采样点':>9} {'天数':>5} {'事件数':>7}  行为分布")
    print(f"  {'-' * 74}")
    for s in stats:
        dist = "  ".join(f"{k}={v}" for k, v in sorted(s["behaviors"].items())) or "—"
        print(f"  {s['device_id']:>6}  {s['device_sn']:<20} {s['rows']:>9} "
              f"{s['days']:>5} {s['events']:>7}  {dist}")
        if s["error"]:
            print(f"          ⚠️  {s['error']}")
    total_rows = sum(s["rows"] for s in stats)
    total_ev   = sum(s["events"] for s in stats)
    print(f"  {'-' * 74}")
    print(f"  {'合计':>6}  {'':<20} {total_rows:>9} {'':>5} {total_ev:>7}")
    print("=" * 78 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="离线回补：把 TDengine 历史 IMU 数据跑推理并写入 MySQL 行为表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 先看看 TDengine 里有哪些设备、数据覆盖哪段时间
  python backfill/run_backfill.py --list-devices

  # 回补 8月19日 这一天的全部设备
  python backfill/run_backfill.py --date 2026-08-19

  # 回补一段区间，并顺带跑当天皮肤评估
  python backfill/run_backfill.py --start 2026-08-19 --end 2026-08-21 --assess

  # 只回补设备 70 和 72，先 dry-run 看看结果分布再决定要不要写库
  python backfill/run_backfill.py --date 2026-08-19 --devices 70,72 --dry-run
        """)
    p.add_argument("--date", help="回补单个日期（YYYY-MM-DD，按各设备用户时区解释）")
    p.add_argument("--start", help="回补区间起始日期（YYYY-MM-DD）")
    p.add_argument("--end", help="回补区间结束日期（含当天，默认与 --start 相同）")
    p.add_argument("--devices", help="只处理指定 device_id，逗号分隔，如 70,72")
    p.add_argument("--assess", action="store_true", help="写完行为事件后再跑一遍当天皮肤评估")
    p.add_argument("--dry-run", action="store_true", help="只推理并打印结果分布，不写数据库")
    p.add_argument("--list-devices", action="store_true", help="列出 TDengine 中的设备及数据时间范围后退出")
    sys.exit(asyncio.run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
