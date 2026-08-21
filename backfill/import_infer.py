#!/usr/bin/env python3
"""
把 imu_train 的离线推理结果导入 MySQL 行为表。

支持两种输入，按扩展名/表头自动识别：

1. imu_train 原生产物 `*_infer.json`（推荐，零额外工作量，默认扫描的唯一目标）
   run_review_bins_all_days.sh 跑完后就在 RESULT_ROOT/{day}/ 下，
   里面的 windows 数组已经是逐窗口的 ts / label / conf。

2. CSV，两种表头之一（需要显式加 --include-csv 才会扫描）：
   窗口级：device_sn,ts,label,conf
   事件级：device_sn,start_ts,end_ts,label,conf

   默认不扫描 *.csv 是因为 run_review_bins_all_days.sh 的输出目录里还混着
   复核用的原始片段 CSV（by_conf_max/clips_*/ 下那些，表头是 acc_x/.../timestamp，
   根本不是推理结果）、Nginx 媒体目录同步产物等，不加区分地扫全部 *.csv
   会把这些也当成推理结果。

窗口级输入会用与线上推理一致的规则合并成行为事件（连续同标签合并，
跨数据空洞不合并），事件级输入原样导入。单个文件解析失败只跳过该文件，
不会中断整个批次。

用法见 backfill/README.md。
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone as dt_tz
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from sqlalchemy import text

from config import settings
from db.client import AsyncSessionLocal
from modules.inference.model import BehaviorLabel
from timezones import resolve as resolve_tz, canonical_name as canonical_tz

# imu_train 中文类别 → BehaviorLabel
_ZH_TO_LABEL: dict[str, int] = {
    "抓挠": int(BehaviorLabel.SCRATCH),
    "活动": int(BehaviorLabel.MOVEMENT),
    "睡觉": int(BehaviorLabel.SLEEP),
}
_LABEL_ZH: dict[int, str] = {v: k for k, v in _ZH_TO_LABEL.items()}
_LABEL_ZH[int(BehaviorLabel.UNKNOWN)] = "未知"

_TS_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
               "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# 时间解析
# ---------------------------------------------------------------------------

def parse_ts(s: str, tz: ZoneInfo) -> int:
    """时间字符串 → UTC 毫秒。带时区偏移的按其自身解释，否则按 tz 解释。"""
    s = str(s).strip()
    if not s:
        raise ValueError("空时间戳")
    if s.isdigit():                       # 已经是 epoch 毫秒
        return int(s)
    try:                                  # ISO8601（可能带 +08:00）
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = None
        for fmt in _TS_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise ValueError(f"无法解析时间戳：{s!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return int(dt.timestamp() * 1000)


def _tz(name: str):
    """时区名 → tzinfo。支持 IANA 名、CST 等缩写、+08:00 这类偏移写法。"""
    tz = resolve_tz(name, default="")
    if tz is None or (getattr(tz, "key", None) is None and str(tz) == "UTC" and
                      name.strip().upper() not in ("UTC", "GMT", "Z")):
        raise SystemExit(f"未知时区：{name}（用 IANA 名，如 Asia/Shanghai）")
    return tz


def fmt_local(ts_ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=tz).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# 设备映射
# ---------------------------------------------------------------------------

class DeviceMap:
    """
    把来源文件名映射到具体设备。映射表 CSV 表头：
        match,device_sn,device_id,bind_id,timezone
    match 是对文件名做的子串匹配（大小写不敏感），如 IMU1 / task496_imu1。
    device_sn 和 device_id 至少填一个：只给 device_sn 时从业务库反查 device_id。
    """

    def __init__(self, rules: list[dict]):
        self.rules = rules

    @classmethod
    def load(cls, path: str) -> "DeviceMap":
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise SystemExit(f"设备映射表为空：{path}")
        rules = []
        for i, r in enumerate(rows, 2):
            r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            if not r.get("match"):
                raise SystemExit(f"{path} 第 {i} 行缺少 match 列")
            if not r.get("device_sn") and not r.get("device_id"):
                raise SystemExit(f"{path} 第 {i} 行 device_sn 和 device_id 至少填一个")
            rules.append(r)
        return cls(rules)

    def resolve(self, source_name: str) -> dict | None:
        name = source_name.lower()
        # 取匹配串最长的那条，避免 IMU1 抢走本该给 IMU10 的行
        best = None
        for r in self.rules:
            m = r["match"].lower()
            if m in name and (best is None or len(m) > len(best["match"])):
                best = r
        return best


async def enrich_from_db(rule: dict) -> dict:
    """用业务库补全 device_id / bind_id / timezone。"""
    out = dict(rule)
    async with AsyncSessionLocal() as db:
        try:
            if out.get("device_sn"):
                row = (await db.execute(text(f"""
                    SELECT dbh.bind_id, dbh.device_id, COALESCE(u.timezone,'UTC') AS timezone
                    FROM {settings.biz_schema}.device d
                    JOIN {settings.biz_schema}.device_bind_history dbh ON dbh.device_id = d.id
                    JOIN {settings.biz_schema}.`user` u ON dbh.user_id = u.id
                    WHERE d.device_sn = :sn AND dbh.bind_status = 1
                    LIMIT 1
                """), {"sn": out["device_sn"]})).fetchone()
            else:
                row = (await db.execute(text(f"""
                    SELECT dbh.bind_id, dbh.device_id, COALESCE(u.timezone,'UTC') AS timezone
                    FROM {settings.biz_schema}.device_bind_history dbh
                    JOIN {settings.biz_schema}.`user` u ON dbh.user_id = u.id
                    WHERE dbh.device_id = :did AND dbh.bind_status = 1
                    LIMIT 1
                """), {"did": int(out["device_id"])})).fetchone()
        except Exception as e:
            await db.rollback()
            logger.warning("业务库查询失败（{}），仅用映射表里填的值", e.__class__.__name__)
            row = None

    if row:
        out.setdefault("device_id", str(row.device_id))
        out["device_id"] = out.get("device_id") or str(row.device_id)
        if not out.get("bind_id") and row.bind_id:
            out["bind_id"] = str(row.bind_id)
        if not out.get("timezone"):
            out["timezone"] = row.timezone
    if not out.get("device_id"):
        raise SystemExit(
            f"设备 {out.get('device_sn')} 在业务库里查不到 device_id，"
            f"请在映射表里直接填 device_id 列"
        )
    return out


# ---------------------------------------------------------------------------
# 读取输入
# ---------------------------------------------------------------------------

def read_infer_json(path: Path, tz: ZoneInfo) -> list[dict]:
    """imu_train *_infer.json → 逐窗口记录 [{ts_ms, label, conf}]。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for w in data.get("windows", []):
        lbl = _ZH_TO_LABEL.get(w.get("label"))
        if lbl is None:
            continue  # 未知类别（比如 remap 之外的标签）直接跳过
        out.append({"ts_ms": parse_ts(w["ts"], tz),
                    "label": lbl,
                    "conf": float(w.get("conf", 0.0))})
    return out


def read_csv_rows(path: Path, tz: ZoneInfo) -> tuple[str, list[dict]]:
    """CSV → ('window'|'event', 记录列表)。按表头自动判断。"""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "window", []
    cols = {c.strip() for c in rows[0]}

    if {"start_ts", "end_ts"} <= cols:
        out = []
        for r in rows:
            lbl = _ZH_TO_LABEL.get((r.get("label") or "").strip())
            if lbl is None:
                continue
            out.append({
                "device_sn": (r.get("device_sn") or "").strip(),
                "start_ms": parse_ts(r["start_ts"], tz),
                "end_ms":   parse_ts(r["end_ts"], tz),
                "label": lbl,
                "conf": float(r.get("conf") or 0.0),
            })
        return "event", out

    if "ts" not in cols:
        raise ValueError(
            f"表头无法识别，需要 ts（窗口级）或 start_ts/end_ts（事件级），"
            f"当前表头：{sorted(cols)}"
        )
    out = []
    for r in rows:
        lbl = _ZH_TO_LABEL.get((r.get("label") or "").strip())
        if lbl is None:
            continue
        out.append({
            "device_sn": (r.get("device_sn") or "").strip(),
            "ts_ms": parse_ts(r["ts"], tz),
            "label": lbl,
            "conf": float(r.get("conf") or 0.0),
        })
    return "window", out


# ---------------------------------------------------------------------------
# 窗口 → 事件
# ---------------------------------------------------------------------------

def merge_windows(wins: list[dict], window_sec: float,
                  max_gap_sec: float | None) -> list[dict]:
    """
    连续同标签窗口合并成事件，规则与线上 windows_to_events 一致。

    多一条：相邻窗口时间间隔超过 max_gap_sec 时不合并——录制中断、设备掉线
    产生的空洞，两边不该被算成一段连续行为。max_gap_sec 为 None 时按
    2.5 倍步长自动推断（步长取相邻窗口时间差的中位数）。
    """
    if not wins:
        return []
    wins = sorted(wins, key=lambda w: w["ts_ms"])
    win_ms = int(window_sec * 1000)

    if max_gap_sec is None:
        diffs = sorted(b["ts_ms"] - a["ts_ms"] for a, b in zip(wins, wins[1:]))
        stride_ms = diffs[len(diffs) // 2] if diffs else win_ms
        gap_ms = max(int(stride_ms * 2.5), win_ms)
    else:
        gap_ms = int(max_gap_sec * 1000)

    events, cur = [], None
    for w in wins:
        if (cur and w["label"] == cur["label"]
                and w["ts_ms"] - cur["last_ts"] <= gap_ms):
            cur["last_ts"] = w["ts_ms"]
            cur["conf_sum"] += w["conf"]
            cur["n"] += 1
            continue
        if cur:
            events.append(_close(cur, win_ms))
        cur = {"label": w["label"], "start_ts": w["ts_ms"], "last_ts": w["ts_ms"],
               "conf_sum": w["conf"], "n": 1}
    if cur:
        events.append(_close(cur, win_ms))
    return events


def _close(cur: dict, win_ms: int) -> dict:
    return {
        "label": cur["label"],
        "start_ms": cur["start_ts"],
        "end_ms": cur["last_ts"] + win_ms,
        "conf": round(cur["conf_sum"] / cur["n"], 4),
        "n_windows": cur["n"],
    }


# ---------------------------------------------------------------------------
# 写库
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS {tbl} (
    id            BIGINT        NOT NULL AUTO_INCREMENT PRIMARY KEY,
    bind_id       BIGINT,
    ts_start      BIGINT        NOT NULL,
    ts_end        BIGINT        NOT NULL,
    behavior      SMALLINT      NOT NULL,
    behavior_label VARCHAR(8),
    duration_sec  DECIMAL(10,2) NOT NULL,
    confidence    DECIMAL(5,3)  NOT NULL,
    local_start   VARCHAR(24),
    local_end     VARCHAR(24),
    user_timezone VARCHAR(32),
    UNIQUE KEY uq_ts_start (ts_start)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


async def write_events(device_id: int, bind_id: int | None, tz_name: str,
                       events: list[dict], suffix: str) -> int:
    tbl = f"{settings.pg_schema_behavior}.d_{device_id}{suffix}"
    tz = _tz(tz_name)
    async with AsyncSessionLocal() as db:
        await db.execute(text(DDL.format(tbl=tbl)))
        await db.commit()
        for ev in events:
            await db.execute(text(f"""
                INSERT IGNORE INTO {tbl}
                    (bind_id, ts_start, ts_end, behavior, behavior_label,
                     duration_sec, confidence, local_start, local_end, user_timezone)
                VALUES
                    (:bind_id, :ts_start, :ts_end, :behavior, :behavior_label,
                     :duration_sec, :confidence, :local_start, :local_end, :tz)
            """), {
                "bind_id":        bind_id,
                "ts_start":       ev["start_ms"],
                "ts_end":         ev["end_ms"],
                "behavior":       ev["label"],
                "behavior_label": _LABEL_ZH.get(ev["label"], "未知"),
                "duration_sec":   round((ev["end_ms"] - ev["start_ms"]) / 1000.0, 2),
                "confidence":     ev["conf"],
                "local_start":    fmt_local(ev["start_ms"], tz),
                "local_end":      fmt_local(ev["end_ms"], tz),
                "tz":             canonical_tz(tz_name, default="UTC"),
            })
        await db.commit()
    return len(events)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def collect_inputs(root: str, include_csv: bool = False) -> list[Path]:
    """
    默认只找 *_infer.json（imu_train 推理产物）。

    run_review_bins_all_days.sh 的输出目录里还混着复核用的原始片段 CSV
    （by_conf_max/clips_*/ 下那些，表头是 acc_x/.../timestamp，根本不是推理
    结果）、Label Studio 任务文件等其它产物，不能不加区分地扫全部 *.csv——
    传 include_csv=True 才会额外找 *.csv，用于手工整理好的窗口级/事件级表。
    """
    p = Path(root)
    if p.is_file():
        return [p]
    files = list(p.rglob("*_infer.json"))
    if include_csv:
        files += list(p.rglob("*.csv"))
    # 排除 imu_train 顺带产出的统计表，那不是推理结果
    return sorted(f for f in files if "daily_scratch_stats" not in f.name)


async def _run(args) -> int:
    tz_default = _tz(args.tz)
    dmap = DeviceMap.load(args.device_map)
    files = collect_inputs(args.input, include_csv=args.include_csv)
    if not files:
        hint = "" if args.include_csv else "（如果你要导入手工整理的 CSV，加 --include-csv）"
        raise SystemExit(f"{args.input} 下没找到 *_infer.json{'/*.csv' if args.include_csv else ''}{hint}")

    logger.info("输入文件 {} 个，时区默认 {}", len(files), args.tz)
    if args.dry_run:
        logger.warning("dry-run：只解析和合并，不写库")

    # device_id → 累积的事件
    buckets: dict[str, dict] = {}
    unmatched: list[str] = []
    skipped: list[str] = []

    for f in files:
        try:
            if f.suffix == ".json":
                wins = read_infer_json(f, tz_default)
                kind, rows = "window", wins
            else:
                kind, rows = read_csv_rows(f, tz_default)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            skipped.append(f"{f.name}：{e}")
            continue
        src_name = f.name
        if not rows:
            logger.warning("{} 没有可用记录，跳过", f.name)
            continue

        # CSV 里带 device_sn 就用它，否则按文件名匹配映射表
        sn_in_file = rows[0].get("device_sn") if f.suffix != ".json" else ""
        rule = dmap.resolve(sn_in_file) if sn_in_file else None
        if rule is None:
            rule = dmap.resolve(src_name)
        if rule is None:
            unmatched.append(src_name)
            continue

        key = rule["match"]
        b = buckets.setdefault(key, {"rule": rule, "windows": [], "events": [],
                                     "files": 0, "kind": kind})
        b["files"] += 1
        if kind == "window":
            b["windows"].extend(rows)
        else:
            b["events"].extend({"label": r["label"], "start_ms": r["start_ms"],
                                "end_ms": r["end_ms"], "conf": r["conf"],
                                "n_windows": 0} for r in rows)

    if skipped:
        logger.warning("以下文件解析失败，已跳过（不影响其它文件）：\n  {}",
                       "\n  ".join(skipped[:20]))
        if len(skipped) > 20:
            logger.warning("  ...还有 {} 个未列出", len(skipped) - 20)

    if unmatched:
        logger.warning("以下文件在映射表里找不到对应设备，已跳过：\n  {}",
                       "\n  ".join(unmatched[:20]))

    if not buckets:
        logger.error("没有任何文件匹配到设备，检查 --device-map 的 match 列")
        return 1

    stats = []
    for key, b in buckets.items():
        rule = await enrich_from_db(b["rule"])
        device_id = int(rule["device_id"])
        bind_id = int(rule["bind_id"]) if rule.get("bind_id") else None
        tz_name = rule.get("timezone") or args.tz

        events = b["events"]
        if b["windows"]:
            events = events + merge_windows(b["windows"], args.window_sec, args.max_gap_sec)
        events.sort(key=lambda e: e["start_ms"])

        dist: dict[str, int] = {}
        secs: dict[str, float] = {}
        for ev in events:
            name = _LABEL_ZH.get(ev["label"], "未知")
            dist[name] = dist.get(name, 0) + 1
            secs[name] = secs.get(name, 0.0) + (ev["end_ms"] - ev["start_ms"]) / 1000.0

        written = 0
        if not args.dry_run:
            written = await write_events(device_id, bind_id, tz_name, events, args.table_suffix)

        stats.append({"match": key, "device_id": device_id, "device_sn": rule.get("device_sn", ""),
                      "tz": tz_name, "files": b["files"], "windows": len(b["windows"]),
                      "events": len(events), "written": written, "dist": dist, "secs": secs})

    _print_summary(stats, args)
    return 0


def _print_summary(stats: list[dict], args) -> None:
    print("\n" + "=" * 86)
    print(f"导入汇总{'（dry-run，未写库）' if args.dry_run else ''}"
          f"    目标表 {settings.pg_schema_behavior}.d_<device_id>{args.table_suffix}")
    print("=" * 86)
    for s in stats:
        print(f"\n  [{s['match']}] → device_id={s['device_id']}"
              f"{'  ' + s['device_sn'] if s['device_sn'] else ''}  时区={s['tz']}")
        print(f"      文件={s['files']}  窗口={s['windows']}  事件={s['events']}"
              f"{'  已写入=' + str(s['written']) if not args.dry_run else ''}")
        for name in sorted(s["dist"]):
            mins = s["secs"][name] / 60.0
            print(f"      {name:<4} 事件={s['dist'][name]:>5}   累计时长={mins:>8.1f} 分钟")
    print("\n" + "=" * 86 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="把 imu_train 离线推理结果导入 MySQL 行为表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 直接吃 imu_train 的原生产物
  python backfill/import_infer.py \\
      --input infer_result_majority/2026_8_19 \\
      --device-map backfill/device_map.csv --dry-run

  # 确认无误后写库
  python backfill/import_infer.py \\
      --input infer_result_majority/2026_8_19 \\
      --device-map backfill/device_map.csv

  # 另一个模型变体导到带后缀的表，方便并排比较，不污染正式表
  python backfill/import_infer.py \\
      --input infer_result_majority_syn/2026_8_19 \\
      --device-map backfill/device_map.csv --table-suffix _syn
        """)
    p.add_argument("--input", required=True,
                   help="推理结果目录（默认只递归找 *_infer.json）或单个文件")
    p.add_argument("--include-csv", action="store_true",
                   help="额外扫描目录下的 *.csv（默认关闭：run_review_bins_all_days.sh "
                        "的输出目录里混着复核用的原始片段 CSV，格式跟推理结果不同，"
                        "不加这个开关不会被误当成推理结果导入）")
    p.add_argument("--device-map", required=True,
                   help="设备映射表 CSV：match,device_sn,device_id,bind_id,timezone")
    p.add_argument("--tz", default="Asia/Shanghai",
                   help="时间戳不带时区时按此时区解释（默认 Asia/Shanghai）")
    p.add_argument("--window-sec", type=float, default=2.0,
                   help="推理窗口长度（秒），用于算事件结束时间（默认 2.0）")
    p.add_argument("--max-gap-sec", type=float, default=None,
                   help="相邻窗口间隔超过该值就断开，不合并成一段"
                        "（默认按 2.5 倍步长自动推断）")
    p.add_argument("--table-suffix", default="",
                   help="目标表名后缀，如 _syn 会写入 d_70_syn，用于并排比较模型变体")
    p.add_argument("--dry-run", action="store_true", help="只解析统计，不写库")
    sys.exit(asyncio.run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
