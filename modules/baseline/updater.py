"""
个体抓挠基线更新模块。

更新策略（与 demo 设计一致）：
- 软冻结权重基于 Z-score 三档：
    Z < 1  → weight = 0.10（正常偏差，正常学习）
    1 ≤ Z < 2 → weight = 0.03（轻度异常，缓慢渗透）
    Z ≥ 2  → weight = 0.00（明显异常，完全冻结）
- 每天执行（由调度器 cron 驱动）
- 使用过去30天的有效天数重新计算标准差和温度系数
- 同步更新 wpeb_mean 和 wpeb_std 的 EWMA
- 数据来源：pet_dog_skin_assessment.{device_sn}（每设备独立表），基线写入 pet_dog_scratch_baseline.pet_skin_baseline
- 设备列表来源：device_sync_state（统一表）
"""

import time

import numpy as np
from sqlalchemy import text

from loguru import logger

from config import settings
from db.client import AsyncSessionLocal


async def _update_one(device_sn: str) -> None:
    async with AsyncSessionLocal() as db:
        # 从 pet_dog_skin_assessment.{device_sn} 拉取过去30天有效天的数据（data_quality=0）
        rows_sql = text(f"""
            SELECT scratch_count, avg_temperature, zscore, wpeb_score
            FROM   {settings.pg_schema_assessment}.{device_sn}
            WHERE  data_quality = 0
            ORDER BY stat_date_ts DESC
            LIMIT 30
        """)
        rows = (await db.execute(rows_sql)).fetchall()

        if not rows:
            return

        # 读取当前基线（pet_dog_scratch_baseline.pet_skin_baseline 按 device_sn 主键的统一表）
        bl_sql = text(f"""
            SELECT baseline_mean, baseline_std, temp_coef, valid_days,
                   wpeb_mean, wpeb_std
            FROM   {settings.pg_schema_baseline}.pet_skin_baseline
            WHERE  device_sn = :sn
        """)
        try:
            bl = (await db.execute(bl_sql, {"sn": device_sn})).fetchone()
        except Exception:
            bl = None

        if bl:
            cur_mean   = float(bl.baseline_mean)
            cur_std    = float(bl.baseline_std)
            cur_coef   = float(bl.temp_coef)
            valid_days = int(bl.valid_days)
            try:
                cur_wpeb_mean = float(bl.wpeb_mean) if bl.wpeb_mean is not None else float(rows[0].wpeb_score or 0)
                cur_wpeb_std  = float(bl.wpeb_std) if bl.wpeb_std is not None else settings.baseline_std_floor_wpeb
            except Exception:
                # wpeb_mean/wpeb_std 字段可能尚未存在
                cur_wpeb_mean = float(rows[0].wpeb_score or 0) if rows else 0.0
                cur_wpeb_std  = settings.baseline_std_floor_wpeb
        else:
            # 首次初始化：以最新一天数据作为种子
            cur_mean      = float(rows[0].scratch_count)
            cur_std       = settings.baseline_std_floor
            cur_coef      = 0.0
            valid_days    = 0
            cur_wpeb_mean = float(rows[0].wpeb_score or 0) if rows else 0.0
            cur_wpeb_std  = settings.baseline_std_floor_wpeb

        # 滚动指数加权更新（从最旧到最新，rows 为 DESC 故先 reverse）
        for row in reversed(rows):
            count  = float(row.scratch_count)
            zscore = float(row.zscore) if row.zscore is not None else 0.0

            # Z-score 三档软冻结权重
            if zscore < 1.0:
                weight = 0.10    # 正常范围，正常学习速率
            elif zscore < 2.0:
                weight = 0.03    # 轻度异常，缓慢渗透
            else:
                weight = 0.00    # 明显异常，完全冻结基线

            # 仅在 weight > 0 时更新均值
            if weight > 0:
                cur_mean = cur_mean * (1 - weight) + count * weight
                valid_days += 1

            # wpeb_score EWMA 更新（同样采用 Z-score 三档权重）
            wpeb = float(row.wpeb_score) if row.wpeb_score is not None else 0.0
            if weight > 0:
                cur_wpeb_mean = cur_wpeb_mean * (1 - weight) + wpeb * weight

        # 用过去30天正常天（Z<1）重新计算标准差
        normal_counts = [
            float(r.scratch_count)
            for r in rows
            if r.zscore is None or float(r.zscore) < 1.0
        ]
        if len(normal_counts) >= 3:
            cur_std = max(float(np.std(normal_counts)), settings.baseline_std_floor)

        # 用过去30天正常天的 wpeb_score 重新计算 wpeb_std
        normal_wpeb = [
            float(r.wpeb_score)
            for r in rows
            if r.wpeb_score is not None and (r.zscore is None or float(r.zscore) < 1.0)
        ]
        if len(normal_wpeb) >= 3:
            cur_wpeb_std = max(float(np.std(normal_wpeb)), settings.baseline_std_floor_wpeb)

        # 温度系数：最小二乘回归（仅使用正常天，至少需要20个点）
        temp_pairs = [
            (float(r.avg_temperature), float(r.scratch_count))
            for r in rows
            if r.avg_temperature is not None and (r.zscore is None or float(r.zscore) < 1.0)
        ]
        if len(temp_pairs) >= 20:
            temps  = np.array([p[0] for p in temp_pairs])
            counts = np.array([p[1] for p in temp_pairs])
            X      = np.column_stack([temps, np.ones(len(temps))])
            coefs, _, _, _ = np.linalg.lstsq(X, counts, rcond=None)
            cur_coef = float(np.clip(coefs[0], 0.0, 0.4))

        # 基线置信度 0-1，上限30天
        confidence = min(valid_days / 30.0, 1.0)
        phase = (
            0 if valid_days < 3 else
            1 if valid_days < 14 else
            2 if valid_days < 30 else 3
        )

        now_ms = int(time.time() * 1000)

        # Upsert 写入基线，包含 wpeb_mean 和 wpeb_std
        upsert_sql = text(f"""
            INSERT INTO {settings.pg_schema_baseline}.pet_skin_baseline
                (device_sn, baseline_mean, baseline_std, temp_coef,
                 valid_days, eval_phase, confidence,
                 wpeb_mean, wpeb_std,
                 last_updated_ts, created_at)
            VALUES
                (:sn, :mean, :std, :coef,
                 :valid_days, :phase, :confidence,
                 :wpeb_mean, :wpeb_std,
                 :now_ms, :now_ms)
            ON CONFLICT (device_sn) DO UPDATE SET
                baseline_mean    = EXCLUDED.baseline_mean,
                baseline_std     = EXCLUDED.baseline_std,
                temp_coef        = EXCLUDED.temp_coef,
                valid_days       = EXCLUDED.valid_days,
                eval_phase       = EXCLUDED.eval_phase,
                confidence       = EXCLUDED.confidence,
                wpeb_mean        = EXCLUDED.wpeb_mean,
                wpeb_std         = EXCLUDED.wpeb_std,
                last_updated_ts  = EXCLUDED.last_updated_ts
        """)
        await db.execute(upsert_sql, {
            "sn":         device_sn,
            "mean":       round(cur_mean, 2),
            "std":        round(cur_std, 2),
            "coef":       round(cur_coef, 3),
            "valid_days": min(valid_days, 30),
            "phase":      phase,
            "confidence": round(confidence, 2),
            "wpeb_mean":  round(cur_wpeb_mean, 4),
            "wpeb_std":   round(cur_wpeb_std, 4),
            "now_ms":     now_ms,
        })
        await db.commit()

        logger.info(
            "基线更新完成 device={} mean={:.2f} std={:.2f} coef={:.3f} "
            "wpeb_mean={:.4f} wpeb_std={:.4f} valid_days={}",
            device_sn, cur_mean, cur_std, cur_coef,
            cur_wpeb_mean, cur_wpeb_std, valid_days,
        )


async def run_baseline_update() -> None:
    """对所有注册设备更新基线（设备列表从 device_sync_state 统一表获取）。"""
    async with AsyncSessionLocal() as db:
        # 从 device_sync_state 获取所有设备，替代原来查询 pet_skin_health_daily
        devices_sql = text(
            "SELECT device_sn FROM device_sync_state"
        )
        devices = (await db.execute(devices_sql)).fetchall()

    for row in devices:
        try:
            await _update_one(row.device_sn)
        except Exception:
            logger.exception("基线更新失败 device={}", row.device_sn)
