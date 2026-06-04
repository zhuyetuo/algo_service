"""
每日皮肤健康评估模块。

逻辑流程（与 demo_all.py 设计一致）：
1. 从 pet_dog_behavior.{device_sn} 汇总当日抓挠事件
2. 从 pet_dog_scratch_baseline.pet_skin_baseline 读取个体基线
3. 在 Z-score 层面进行温度修正（原始计数不做修改）
4. 根据 eval_phase 应用动态阈值
5. 跟踪连续异常天数
6. 计算 S1-S6 各维度分数及综合健康等级
7. 将结果写入 pet_dog_skin_assessment.{device_sn} 表
"""

import time
from dataclasses import dataclass

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from loguru import logger

from config import settings
from db.client import AsyncSessionLocal, get_session
from modules.inference.model import BehaviorLabel
router = APIRouter()


# ---------------------------------------------------------------------------
# 阶段阈值
# ---------------------------------------------------------------------------

@dataclass
class PhaseThreshold:
    z: float
    consec: int
    avg_z: float


def get_threshold(valid_days: int) -> PhaseThreshold | None:
    """预热期（valid_days<3）返回 None，不做评估。"""
    if valid_days < 3:
        return None
    if valid_days < 14:
        return PhaseThreshold(
            settings.phase1_threshold_z,
            settings.phase1_threshold_consec,
            settings.phase1_threshold_avgz,
        )
    if valid_days < 30:
        return PhaseThreshold(
            settings.phase2_threshold_z,
            settings.phase2_threshold_consec,
            settings.phase2_threshold_avgz,
        )
    return PhaseThreshold(
        settings.phase3_threshold_z,
        settings.phase3_threshold_consec,
        settings.phase3_threshold_avgz,
    )


def eval_phase(valid_days: int) -> int:
    """根据有效天数返回评估阶段编号（0-3）。"""
    if valid_days < 3:
        return 0
    if valid_days < 14:
        return 1
    if valid_days < 30:
        return 2
    return 3


# ---------------------------------------------------------------------------
# 各维度评分辅助函数
# ---------------------------------------------------------------------------

def _calc_s1(zscore: float) -> float:
    """S1 抓挠频次得分（满分25分）：Z-score 映射，z<=0 得0分。"""
    if zscore <= 0:
        return 0.0
    return min(zscore / 4.0 * 25.0, 25.0)


def _calc_s2(wpeb: float, wpeb_mean: float, wpeb_std: float) -> float:
    """
    S2 强度时长得分（满分25分）：W-PEB Z-score 映射。
    wpeb_std 最小值用 settings.baseline_std_floor_wpeb 防止除以零。
    """
    std_eff = max(wpeb_std, settings.baseline_std_floor_wpeb)
    z = (wpeb - wpeb_mean) / std_eff
    if z <= 0:
        return 0.0
    return min(z / 4.0 * 25.0, 25.0)


def _calc_s3(night_scratch_count: int, night_sleep_segments: int) -> float:
    """
    S3 睡眠质量得分（满分20分）。
    夜间抓挠次数：0→0分，1-2→5分，3-5→9分，6+→12分。
    睡眠碎片化（夜间睡眠段数）：1段→0分，2-3段→3分，4+→8分。
    """
    # 夜间抓挠次数得分
    if night_scratch_count == 0:
        scratch_score = 0
    elif night_scratch_count <= 2:
        scratch_score = 5
    elif night_scratch_count <= 5:
        scratch_score = 9
    else:
        scratch_score = 12

    # 睡眠碎片化得分
    if night_sleep_segments <= 1:
        frag_score = 0
    elif night_sleep_segments <= 3:
        frag_score = 3
    else:
        frag_score = 8

    return float(scratch_score + frag_score)


def _calc_s4(avg_temp: float | None, worn_loose_minutes: float) -> float:
    """
    S4 颈部体温得分（满分10分）。
    avg_temp 为空或 worn_loose_minutes>240 则返回0。
    <38°C→0分；38-39.2°C→2分；39.2-39.7°C→6分；>39.7°C→10分。
    """
    if avg_temp is None or worn_loose_minutes > 240:
        return 0.0
    if avg_temp < 38.0:
        return 0.0
    if avg_temp < 39.2:
        return 2.0
    if avg_temp <= 39.7:
        return 6.0
    return 10.0


def _calc_s6(env_temp: float | None, env_humidity: float | None) -> float:
    """
    S6 环境修正分（±5分）。
    env_temp>30→-2，env_temp<10→+1，env_humidity<40→+2，env_humidity>80→-1，
    结果 clamp(-5, 5)。
    """
    score = 0.0
    if env_temp is not None:
        if env_temp > 30:
            score -= 2
        elif env_temp < 10:
            score += 1
    if env_humidity is not None:
        if env_humidity < 40:
            score += 2
        elif env_humidity > 80:
            score -= 1
    return float(max(-5.0, min(5.0, score)))


def _score_to_level(score: float) -> int:
    """将总分（0-100）映射到健康等级 L0-L10，每10分一档。"""
    return min(int(score // 10), 10)


def _get_alert_level(consec: int) -> int:
    """
    根据连续异常天数返回告警等级。
    0=无告警，1=轻提醒（连续≥2天），2=建议就医（连续≥3天），3=紧急（连续≥5天）。
    """
    if consec >= 5:
        return 3
    if consec >= 3:
        return 2
    if consec >= 2:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 单设备日度评估
# ---------------------------------------------------------------------------

async def assess_device(db: AsyncSession, device_sn: str, stat_date_ts: int) -> None:
    """
    计算并写入单台设备的每日皮肤健康记录。

    stat_date_ts：该设备当地日期零点的 UTC 毫秒时间戳，由调用方传入（后端负责时区转换）。
    数据读取来源：pet_dog_behavior.{device_sn}、pet_dog_skin_assessment.{device_sn}、pet_dog_environment.{device_sn}。
    """
    day_end_ts = stat_date_ts + 86_400_000  # 当天结束毫秒时间戳（+24小时）

    # ── 1. 汇总当日抓挠事件（从 pet_dog_behavior.{device_sn} 读取） ──────────────────
    # SCRATCH 行为枚举值为 3，直接用整数过滤，无需 device_sn 参数
    scratch_sql = text(f"""
        SELECT ts_start, ts_end, confidence
        FROM   {settings.pg_schema_behavior}.{device_sn}
        WHERE  behavior   = {int(BehaviorLabel.SCRATCH)}
          AND  ts_start  >= :day_start
          AND  ts_start   < :day_end
        ORDER BY ts_start
    """)
    rows = (await db.execute(scratch_sql, {
        "day_start": stat_date_ts,
        "day_end":   day_end_ts,
    })).fetchall()

    scratch_count = len(rows)
    durations_ms  = [r.ts_end - r.ts_start for r in rows]
    scratch_dur   = sum(durations_ms)
    scratch_avg_dur = int(scratch_dur / scratch_count) if scratch_count else 0
    scratch_max_dur = max(durations_ms, default=0)

    # 夜间抓挠：stat_date_ts 为当地零点，夜间窗口为前一晚22:00到当天06:00
    night_start_ts = stat_date_ts - 2 * 3_600_000
    night_end_ts   = stat_date_ts + 6 * 3_600_000
    night_count = sum(
        1 for r in rows
        if night_start_ts <= r.ts_start < night_end_ts
    )

    # ── 2. 计算每个抓挠事件的 W-PEB（加权强度×时长积分） ─────────────────────
    wpeb_score = 0.0
    for r in rows:
        conf = float(r.confidence) if r.confidence is not None else 0.5
        dur_s = (r.ts_end - r.ts_start) / 1000.0  # 转换为秒

        # 置信度代理强度分级
        if conf < 0.65:
            intensity_weight = 1.0    # 轻度
        elif conf < 0.85:
            intensity_weight = 1.5    # 中度
        else:
            intensity_weight = 2.5    # 重度

        # 时长权重分级
        if dur_s < 3.0:
            duration_weight = 0.0     # 过短，不计入
        elif dur_s < 10.0:
            duration_weight = 1.0
        elif dur_s < 30.0:
            duration_weight = 1.5
        else:
            duration_weight = 2.0

        wpeb_score += intensity_weight * duration_weight

    # ── 3. 从行为表估算佩戴时长 ──────────────────────────────────────────
    wear_sql = text(f"""
        SELECT COALESCE(SUM(ts_end - ts_start), 0) AS wore_ms
        FROM   {settings.pg_schema_behavior}.{device_sn}
        WHERE  ts_start >= :day_start
          AND  ts_start  < :day_end
    """)
    wore_ms = (await db.execute(wear_sql, {
        "day_start": stat_date_ts, "day_end": day_end_ts,
    })).scalar() or 0
    wear_minutes = int(wore_ms / 60_000)
    worn_loose_minutes = 0.0

    # 佩戴时长低于阈值则标记为无效天
    data_quality = 0
    if wear_minutes < settings.min_wear_minutes:
        data_quality = 1  # 无效天，数据不足

    # ── 4. 读取基线（pet_dog_scratch_baseline.pet_skin_baseline，按 device_sn 主键） ────────
    bl_sql = text(f"""
        SELECT baseline_mean, baseline_std, temp_coef, valid_days,
               wpeb_mean, wpeb_std
        FROM   {settings.pg_schema_baseline}.pet_skin_baseline
        WHERE  device_sn = :sn
    """)
    try:
        bl = (await db.execute(bl_sql, {"sn": device_sn})).fetchone()
    except Exception:
        await db.rollback()
        bl = None

    if bl is None:
        # 尚无基线，以当日数据作为初始种子
        valid_days   = 1
        bl_mean      = float(scratch_count)
        bl_std       = settings.baseline_std_floor
        temp_coef    = 0.0
        wpeb_mean    = wpeb_score
        wpeb_std     = settings.baseline_std_floor_wpeb
    else:
        valid_days = bl.valid_days
        bl_mean    = float(bl.baseline_mean)
        bl_std     = max(float(bl.baseline_std), settings.baseline_std_floor)
        temp_coef  = float(bl.temp_coef)
        try:
            wpeb_mean = float(bl.wpeb_mean) if bl.wpeb_mean is not None else wpeb_score
            wpeb_std  = max(float(bl.wpeb_std) if bl.wpeb_std is not None else settings.baseline_std_floor_wpeb,
                            settings.baseline_std_floor_wpeb)
        except Exception:
            # wpeb_mean/wpeb_std 字段可能尚未存在
            wpeb_mean = wpeb_score
            wpeb_std  = settings.baseline_std_floor_wpeb

    phase  = eval_phase(valid_days)
    thresh = get_threshold(valid_days)

    # ── 5. 读取日均体温（从 pet_dog_skin_assessment.{device_sn} 读取已有记录） ────────
    # 确保皮肤评估表已存在
    await db.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {settings.pg_schema_assessment}.{device_sn} (
            stat_date_ts       bigint PRIMARY KEY,
            scratch_count      int            NOT NULL DEFAULT 0,
            scratch_duration   bigint         NOT NULL DEFAULT 0,
            scratch_avg_dur    int            NOT NULL DEFAULT 0,
            scratch_max_dur    int            NOT NULL DEFAULT 0,
            night_scratch_count int           NOT NULL DEFAULT 0,
            wpeb_score         decimal(10,4)  NOT NULL DEFAULT 0,
            avg_humidity       decimal(5,1),
            baseline_mean      decimal(6,2),
            baseline_std       decimal(6,2),
            temp_coef          decimal(5,3),
            temp_effect        decimal(6,2),
            zscore             decimal(6,2),
            avg_zscore         decimal(6,2),
            consec_abnormal    int            NOT NULL DEFAULT 0,
            eval_phase         smallint       NOT NULL DEFAULT 0,
            threshold_z        decimal(4,2),
            threshold_consec   smallint,
            threshold_avgz     decimal(4,2),
            valid_days         int            NOT NULL DEFAULT 0,
            is_abnormal        smallint       NOT NULL DEFAULT 0,
            alert_level        smallint       NOT NULL DEFAULT 0,
            alert_reason       varchar(256),
            s1_score           decimal(5,1)   NOT NULL DEFAULT 0,
            s2_score           decimal(5,1)   NOT NULL DEFAULT 0,
            s3_score           decimal(5,1)   NOT NULL DEFAULT 0,
            s4_score           decimal(5,1)   NOT NULL DEFAULT 0,
            s5_score           decimal(5,1)   NOT NULL DEFAULT 0,
            s6_score           decimal(5,1)   NOT NULL DEFAULT 0,
            total_score        decimal(6,1)   NOT NULL DEFAULT 0,
            health_level       smallint       NOT NULL DEFAULT 0,
            data_quality       smallint       NOT NULL DEFAULT 0,
            wear_minutes       int            NOT NULL DEFAULT 0,
            worn_loose_minutes decimal(8,1)   NOT NULL DEFAULT 0,
            created_at         bigint         NOT NULL,
            updated_at         bigint         NOT NULL
        )
    """))
    await db.commit()  # DDL must be committed before reads on this table

    # 从环境表读取当天颈部体温（由 env sync 预先写入）
    neck_temp_sql = text(f"""
        SELECT neck_temp FROM {settings.pg_schema_environment}.{device_sn}
        WHERE ts = :ts
    """)
    try:
        neck_row = (await db.execute(neck_temp_sql, {"ts": stat_date_ts})).fetchone()
        avg_temperature = float(neck_row.neck_temp) if neck_row and neck_row.neck_temp else None
    except Exception:
        await db.rollback()
        avg_temperature = None

    # ── 6. Z-score 计算（体温修正在 Z-score 层面进行，原始计数不变） ────────
    # 参考温度 20°C，温度效应从偏差中减去
    ref_temp    = 20.0
    temp_effect = round(temp_coef * (avg_temperature - ref_temp), 2) if avg_temperature else 0.0

    zscore = None
    avg_zscore = None
    is_abnormal = False

    # 连续异常天数：data_quality=1 时从上一天原样继承，不重置也不累加
    consec_abnormal = 0

    if data_quality == 1:
        # 数据不足：读取上一天的 consec_abnormal 原样保留（跳过本天）
        prev_consec_sql = text(f"""
            SELECT consec_abnormal FROM {settings.pg_schema_assessment}.{device_sn}
            WHERE  stat_date_ts < :ts
            ORDER BY stat_date_ts DESC
            LIMIT 1
        """)
        try:
            prev_consec_row = (await db.execute(prev_consec_sql, {"ts": stat_date_ts})).fetchone()
            consec_abnormal = int(prev_consec_row.consec_abnormal) if prev_consec_row else 0
        except Exception:
            await db.rollback()
            consec_abnormal = 0

    elif thresh is not None:
        # 正常数据天：计算 Z-score 并更新连续计数
        raw_deviation = scratch_count - bl_mean
        adjusted_dev  = raw_deviation - temp_effect
        zscore = round(adjusted_dev / bl_std, 3)

        # 读取近 N 天 z-score 计算滑动均值
        recent_sql = text(f"""
            SELECT zscore FROM {settings.pg_schema_assessment}.{device_sn}
            WHERE  stat_date_ts < :ts
              AND  data_quality  = 0
              AND  zscore IS NOT NULL
            ORDER BY stat_date_ts DESC
            LIMIT  :n
        """)
        recent_rows = (await db.execute(recent_sql, {
            "ts": stat_date_ts,
            "n": thresh.consec - 1,
        })).fetchall()

        recent_zscores = [float(r.zscore) for r in recent_rows] + [zscore]
        avg_zscore = round(sum(recent_zscores) / len(recent_zscores), 3)

        is_abnormal = zscore > thresh.z

        # 读取上一个有效天的连续计数
        prev_sql = text(f"""
            SELECT consec_abnormal FROM {settings.pg_schema_assessment}.{device_sn}
            WHERE  stat_date_ts < :ts
              AND  data_quality  = 0
            ORDER BY stat_date_ts DESC
            LIMIT 1
        """)
        prev_row = (await db.execute(prev_sql, {"ts": stat_date_ts})).fetchone()
        prev_consec = int(prev_row.consec_abnormal) if prev_row else 0

        # 异常则累加，正常则归零
        consec_abnormal = (prev_consec + 1) if is_abnormal else 0

    # ── 7. 告警等级（整数：0=无，1=轻提醒，2=建议就医，3=紧急） ─────────────
    alert_level = _get_alert_level(consec_abnormal) if is_abnormal else 0
    # 告警原因文本
    alert_reason = None
    if alert_level > 0 and thresh is not None:
        alert_reason = (
            f"连续{consec_abnormal}天z-score>{thresh.z}，"
            f"均值z={avg_zscore}"
        )

    # ── 8. 夜间睡眠段数（从 pet_dog_behavior.{device_sn} 查询，用于 S3） ────────────
    # SLEEP 行为枚举值为 2，直接用整数过滤
    night_sleep_segments = 0
    try:
        sleep_sql = text(f"""
            SELECT COUNT(*) AS seg_count
            FROM   {settings.pg_schema_behavior}.{device_sn}
            WHERE  behavior   = {int(BehaviorLabel.SLEEP)}
              AND  ts_start  >= :night_start
              AND  ts_start   < :night_end
        """)
        seg_row = (await db.execute(sleep_sql, {
            "night_start": night_start_ts,
            "night_end":   night_end_ts,
        })).fetchone()
        night_sleep_segments = int(seg_row.seg_count) if seg_row else 0
    except Exception:
        await db.rollback()
        night_sleep_segments = 0

    # ── 9. 读取环境数据（从 pet_dog_environment.{device_sn} 读取，用于 S6） ──────────
    env_temp = None
    env_humidity = None
    try:
        env_sql = text(f"""
            SELECT env_temp, env_humidity
            FROM   {settings.pg_schema_environment}.{device_sn}
            WHERE  ts = :day_ts
        """)
        env_row = (await db.execute(env_sql, {"day_ts": stat_date_ts})).fetchone()
        if env_row:
            env_temp     = float(env_row.env_temp) if env_row.env_temp is not None else None
            env_humidity = float(env_row.env_humidity) if env_row.env_humidity is not None else None
    except Exception:
        await db.rollback()
        # pet_dog_environment.{device_sn} 表不存在时 S6 为 0

    # ── 10. 各维度评分 ────────────────────────────────────────────────────────
    s1_score = _calc_s1(zscore if zscore is not None else 0.0)
    s2_score = _calc_s2(wpeb_score, wpeb_mean, wpeb_std)
    s3_score = _calc_s3(night_count, night_sleep_segments)
    s4_score = _calc_s4(avg_temperature, worn_loose_minutes)

    # S5：问卷分数，由外部写入，从已有行读取（默认0）
    s5_score = 0.0
    try:
        s5_sql = text(f"""
            SELECT s5_score FROM {settings.pg_schema_assessment}.{device_sn}
            WHERE stat_date_ts = :ts
        """)
        s5_row = (await db.execute(s5_sql, {"ts": stat_date_ts})).fetchone()
        s5_score = float(s5_row.s5_score) if s5_row and s5_row.s5_score is not None else 0.0
    except Exception:
        await db.rollback()
        s5_score = 0.0

    s6_score = _calc_s6(env_temp, env_humidity)

    # 综合总分 clamp(0, 100)
    total_score = float(max(0.0, min(100.0, s1_score + s2_score + s3_score + s4_score + s5_score + s6_score)))
    health_level = _score_to_level(total_score)

    # ── 11. Upsert 结果写入 pet_dog_skin_assessment.{device_sn} ─────────────────────
    # 无 device_sn 列，stat_date_ts 为主键，ON CONFLICT 按主键更新
    now_ms = int(time.time() * 1000)
    upsert_sql = text(f"""
        INSERT INTO {settings.pg_schema_assessment}.{device_sn} (
            stat_date_ts,
            scratch_count, scratch_duration, scratch_avg_dur, scratch_max_dur,
            night_scratch_count,
            baseline_mean, baseline_std, temp_coef,
            temp_effect, zscore, avg_zscore,
            consec_abnormal,
            eval_phase, threshold_z, threshold_consec, threshold_avgz,
            valid_days,
            is_abnormal, alert_level, alert_reason,
            data_quality, wear_minutes, worn_loose_minutes,
            wpeb_score,
            s1_score, s2_score, s3_score, s4_score, s5_score, s6_score,
            total_score, health_level,
            created_at, updated_at
        ) VALUES (
            :stat_date_ts,
            :scratch_count, :scratch_dur, :scratch_avg_dur, :scratch_max_dur,
            :night_count,
            :bl_mean, :bl_std, :temp_coef,
            :temp_effect, :zscore, :avg_zscore,
            :consec_abnormal,
            :phase, :thresh_z, :thresh_consec, :thresh_avgz,
            :valid_days,
            :is_abnormal, :alert_level, :alert_reason,
            :data_quality, :wear_minutes, :worn_loose_minutes,
            :wpeb_score,
            :s1_score, :s2_score, :s3_score, :s4_score, :s5_score, :s6_score,
            :total_score, :health_level,
            :now_ms, :now_ms
        )
        ON CONFLICT (stat_date_ts) DO UPDATE SET
            scratch_count        = EXCLUDED.scratch_count,
            scratch_duration     = EXCLUDED.scratch_duration,
            scratch_avg_dur      = EXCLUDED.scratch_avg_dur,
            scratch_max_dur      = EXCLUDED.scratch_max_dur,
            night_scratch_count  = EXCLUDED.night_scratch_count,
            temp_effect          = EXCLUDED.temp_effect,
            zscore               = EXCLUDED.zscore,
            avg_zscore           = EXCLUDED.avg_zscore,
            consec_abnormal      = EXCLUDED.consec_abnormal,
            is_abnormal          = EXCLUDED.is_abnormal,
            alert_level          = EXCLUDED.alert_level,
            alert_reason         = EXCLUDED.alert_reason,
            data_quality         = EXCLUDED.data_quality,
            wear_minutes         = EXCLUDED.wear_minutes,
            worn_loose_minutes   = EXCLUDED.worn_loose_minutes,
            wpeb_score           = EXCLUDED.wpeb_score,
            s1_score             = EXCLUDED.s1_score,
            s2_score             = EXCLUDED.s2_score,
            s3_score             = EXCLUDED.s3_score,
            s4_score             = EXCLUDED.s4_score,
            s6_score             = EXCLUDED.s6_score,
            total_score          = EXCLUDED.total_score,
            health_level         = EXCLUDED.health_level,
            updated_at           = EXCLUDED.updated_at
    """)
    await db.execute(upsert_sql, {
        "stat_date_ts":     stat_date_ts,
        "scratch_count":    scratch_count,
        "scratch_dur":      scratch_dur,
        "scratch_avg_dur":  scratch_avg_dur,
        "scratch_max_dur":  scratch_max_dur,
        "night_count":      night_count,
        "bl_mean":          bl_mean,
        "bl_std":           bl_std,
        "temp_coef":        temp_coef,
        "temp_effect":      temp_effect,
        "zscore":           zscore,
        "avg_zscore":       avg_zscore,
        "consec_abnormal":  consec_abnormal,
        "phase":            phase,
        "thresh_z":         thresh.z if thresh else None,
        "thresh_consec":    thresh.consec if thresh else None,
        "thresh_avgz":      thresh.avg_z if thresh else None,
        "valid_days":       valid_days,
        "is_abnormal":      int(is_abnormal),
        "alert_level":      alert_level,
        "alert_reason":     alert_reason,
        "data_quality":     data_quality,
        "wear_minutes":     wear_minutes,
        "worn_loose_minutes": worn_loose_minutes,
        "wpeb_score":       round(wpeb_score, 4),
        "s1_score":         round(s1_score, 2),
        "s2_score":         round(s2_score, 2),
        "s3_score":         round(s3_score, 2),
        "s4_score":         round(s4_score, 2),
        "s5_score":         round(s5_score, 2),
        "s6_score":         round(s6_score, 2),
        "total_score":      round(total_score, 2),
        "health_level":     health_level,
        "now_ms":           now_ms,
    })
    await db.commit()

    if alert_level > 0:
        logger.warning(
            "ALERT device={} stat_date_ts={} alert_level={} reason={}",
            device_sn, stat_date_ts, alert_level, alert_reason,
        )
    else:
        logger.info(
            "评估完成 device={} phase={} zscore={} consec={} alert_level={} "
            "total_score={:.2f} health_level={}",
            device_sn, phase, zscore, consec_abnormal, alert_level,
            total_score, health_level,
        )


# ---------------------------------------------------------------------------
# 批量调度任务入口
# ---------------------------------------------------------------------------

async def run_batch_assessment(stat_date_ts: int | None = None) -> None:
    """
    对所有注册设备执行每日评估。
    设备列表从 device_sync_state 获取（而非旧的 pet_behavior_record 表）。

    stat_date_ts：当地零点的 UTC 毫秒时间戳。
    若为 None，则使用当前 UTC 零点（适用于单时区部署）。
    """
    if stat_date_ts is None:
        now = int(time.time())
        stat_date_ts = (now // 86400) * 86400 * 1000

    async with AsyncSessionLocal() as db:
        # 从 device_sync_state 获取所有设备列表
        devices = (await db.execute(text(
            "SELECT device_sn FROM device_sync_state"
        ))).fetchall()

    for row in devices:
        async with AsyncSessionLocal() as db:
            try:
                await assess_device(db, row.device_sn, stat_date_ts)
            except Exception:
                logger.exception("评估失败 device={}", row.device_sn)


# ---------------------------------------------------------------------------
# REST 接口：查询单台设备最新评估报告
# ---------------------------------------------------------------------------

class AssessmentReport(BaseModel):
    device_sn: str
    stat_date_ts: int
    scratch_count: int
    zscore: float | None
    is_abnormal: bool
    alert_level: int
    alert_reason: str | None
    eval_phase: int
    valid_days: int
    confidence: float   # 基线置信度 0-1
    total_score: float
    health_level: int


@router.get("/report/{device_sn}", response_model=AssessmentReport)
async def get_report(
    device_sn: str,
    db: AsyncSession = Depends(get_session),
):
    """返回指定设备最新一天的评估报告（从 pet_dog_skin_assessment.{device_sn} 读取）。"""
    # 动态构造表名，device_sn 来自内部系统，安全
    sql = text(f"""
        SELECT '{device_sn}' AS device_sn,
               d.stat_date_ts,
               d.scratch_count, d.zscore,
               d.is_abnormal, d.alert_level, d.alert_reason,
               d.eval_phase, d.valid_days,
               b.confidence,
               d.total_score, d.health_level
        FROM   {settings.pg_schema_assessment}.{device_sn} d
        LEFT JOIN {settings.pg_schema_baseline}.pet_skin_baseline b ON b.device_sn = '{device_sn}'
        ORDER BY d.stat_date_ts DESC
        LIMIT 1
    """)
    row = (await db.execute(sql)).fetchone()
    if row is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="暂无评估数据")

    return AssessmentReport(
        device_sn=row.device_sn,
        stat_date_ts=row.stat_date_ts,
        scratch_count=row.scratch_count,
        zscore=row.zscore,
        is_abnormal=bool(row.is_abnormal),
        alert_level=int(row.alert_level) if row.alert_level is not None else 0,
        alert_reason=row.alert_reason,
        eval_phase=row.eval_phase,
        valid_days=row.valid_days,
        confidence=float(row.confidence or 0),
        total_score=float(row.total_score) if row.total_score is not None else 0.0,
        health_level=int(row.health_level) if row.health_level is not None else 0,
    )
