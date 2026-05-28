import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import numpy as np

from modules.inference.model import BehaviorLabel, get_classifier

router = APIRouter()


# 单次 IMU 采样数据结构
class ImuSample(BaseModel):
    ts_ms: int    # UTC 毫秒时间戳
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


# 行为事件输出结构
class BehaviorEvent(BaseModel):
    behavior_type: int   # BehaviorLabel 枚举值
    behavior_name: str   # 人类可读的行为名称
    start_time: int      # UTC 毫秒
    end_time: int        # UTC 毫秒
    confidence: float


# 推理请求结构
class InferenceRequest(BaseModel):
    device_sn: str
    # 完整拉取窗口内按时间顺序排列的 IMU 采样数据
    samples: list[ImuSample] = Field(..., min_length=1)


# 推理响应结构
class InferenceResponse(BaseModel):
    device_sn: str
    events: list[BehaviorEvent]
    scratch_count: int    # 便捷字段：本批次中抓挠事件数量


# 行为标签名称映射
_LABEL_NAMES = {
    BehaviorLabel.UNKNOWN:  "unknown",
    BehaviorLabel.MOVEMENT: "movement",
    BehaviorLabel.SLEEP:    "sleep",
    BehaviorLabel.SCRATCH:  "scratch",
}


@router.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    """接收 IMU 采样数据，返回行为事件列表及抓挠次数。"""
    try:
        clf = get_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # 按时间戳排序后取第一帧的时间戳作为基准
    samples_sorted = sorted(req.samples, key=lambda s: s.ts_ms)
    base_ts_ms = samples_sorted[0].ts_ms

    # 构建 (N, 6) 的 numpy 数组
    data = np.array(
        [[s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in samples_sorted],
        dtype=np.float32,
    )

    raw_events = clf.predict(data, base_ts_ms)

    # 将原始事件转换为响应模型格式
    events = [
        BehaviorEvent(
            behavior_type=e["behavior_type"],
            behavior_name=_LABEL_NAMES.get(e["behavior_type"], "unknown"),
            start_time=e["start_time"],
            end_time=e["end_time"],
            confidence=e["confidence"],
        )
        for e in raw_events
    ]

    # 统计抓挠事件数量
    scratch_count = sum(
        1 for e in events if e.behavior_type == BehaviorLabel.SCRATCH
    )

    return InferenceResponse(
        device_sn=req.device_sn,
        events=events,
        scratch_count=scratch_count,
    )
