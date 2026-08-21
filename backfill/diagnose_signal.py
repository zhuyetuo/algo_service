#!/usr/bin/env python3
"""
信号量级诊断：用真实 TDengine 数据核对 IMU 单位，别靠猜。

模型训练时不做量纲统一（imu_train 是什么单位进就是什么单位学），而特征里
mean/std/rms/range/SMA/模长/jerk 全是有量纲的绝对量——设备上报单位和训练单位
差一个量级，这些特征就整体平移出训练分布。表现往往不是"全错"，而是置信度
莫名偏低、某几类死活预测不出来，比全错更难查。

判断依据：静止时 |acc| 必然等于重力常数。中位数 ≈9.8 就是 m/s²，≈1.0 就是 g。

用法：
  # 抽最近的数据看量级（不指定设备则逐台都看）
  python backfill/diagnose_signal.py

  # 只看某台设备
  python backfill/diagnose_signal.py --device-sn EA:CB:3E:CF:00:11

  # 看指定某天的数据
  python backfill/diagnose_signal.py --device-sn EA:CB:3E:CF:00:11 --date 2026-08-19
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone as dt_tz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from config import settings
from db.tdengine import td_fetch_range, td_get_devices, td_device_span
from modules.inference import units as U


def _rows_to_array(rows: list[dict]) -> np.ndarray:
    return np.array([[r["ax"], r["ay"], r["az"], r["gx"], r["gy"], r["gz"]]
                     for r in sorted(rows, key=lambda r: r["ts_ms"])], dtype=np.float32)


async def _sample(device_sn: str, date_str: str | None, max_rows: int) -> np.ndarray | None:
    """取一段数据用于诊断：指定日期就取那天，否则取最近的一段。"""
    if date_str:
        day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt_tz.utc)
        start_ms = int(day.timestamp() * 1000)
        end_ms = start_ms + 86_400_000
    else:
        span = await asyncio.to_thread(td_device_span, device_sn)
        if not span:
            return None
        # 往前推 1 小时，够算量级了，不用把整天拉下来
        end_ms = span["last_ts"] + 1
        start_ms = max(span["first_ts"], end_ms - 3_600_000)

    rows = await asyncio.to_thread(td_fetch_range, device_sn, start_ms, end_ms)
    if not rows:
        return None
    return _rows_to_array(rows[:max_rows] if max_rows else rows)


async def _run(args) -> int:
    sns = [args.device_sn] if args.device_sn else await asyncio.to_thread(td_get_devices)
    if not sns:
        print("TDengine 中没有任何设备数据")
        return 1

    print(f"\n配置中的单位约定："
          f"设备加速度={U.ACC_UNIT_LABEL.get(settings.imu_device_acc_unit, '?')}  "
          f"设备角速度={U.GYRO_UNIT_LABEL.get(settings.imu_device_gyro_unit, '?')}  →  "
          f"模型加速度={U.ACC_UNIT_LABEL.get(settings.imu_model_acc_unit, '?')}  "
          f"模型角速度={U.GYRO_UNIT_LABEL.get(settings.imu_model_gyro_unit, '?')}")

    any_ok = False
    for sn in sns:
        print("\n" + "=" * 74)
        print(f"设备 {sn}")
        print("=" * 74)
        try:
            data = await _sample(sn, args.date, args.max_rows)
        except Exception as e:
            print(f"  拉取失败：{e}")
            continue
        if data is None or len(data) == 0:
            print("  该区间内没有数据")
            continue

        any_ok = True
        diag = U.diagnose(data)
        a, g = diag["acc_mag"], diag["gyro_mag"]
        print(f"  采样点数 : {diag['n']}")
        print(f"  |acc|    : 中位数={a['median']:.4f}  p5={a['p5']:.4f}  p95={a['p95']:.4f}")
        print(f"  |gyro|   : 中位数={g['median']:.4f}  p95={g['p95']:.4f}  max={g['max']:.4f}")
        print()
        for line in U.describe(diag, settings.imu_device_acc_unit,
                               settings.imu_device_gyro_unit):
            print("  " + line)

    if any_ok:
        print("\n" + "-" * 74)
        print("改单位的方法：在 docker-compose.yml 或 .env 里设置，例如角速度实际是 rad/s：")
        print("    IMU_DEVICE_GYRO_UNIT=rads")
        print("改完执行 docker compose down && docker compose up -d 生效。")
        print("启动日志的「量纲统一」那行会打印最终生效的换算系数，可核对。")
        print("-" * 74 + "\n")
    return 0 if any_ok else 1


def main() -> None:
    p = argparse.ArgumentParser(
        description="用真实 TDengine 数据诊断 IMU 单位是否与模型训练单位一致",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device-sn", help="只诊断指定设备，不填则逐台都看")
    p.add_argument("--date", help="诊断指定日期的数据（YYYY-MM-DD，UTC），不填取最近 1 小时")
    p.add_argument("--max-rows", type=int, default=200_000,
                   help="单台设备最多取多少行参与统计（默认 20 万，0=不限）")
    sys.exit(asyncio.run(_run(p.parse_args())))


if __name__ == "__main__":
    main()
