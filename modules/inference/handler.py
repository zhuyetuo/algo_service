import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import numpy as np

from modules.inference.model import BehaviorLabel, get_classifier

router = APIRouter()


class ImuSample(BaseModel):
    ts_ms: int    # UTC millisecond timestamp of this sample
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class BehaviorEvent(BaseModel):
    behavior_type: int   # BehaviorLabel value
    behavior_name: str   # human-readable
    start_time: int      # UTC ms
    end_time: int        # UTC ms
    confidence: float


class InferenceRequest(BaseModel):
    device_sn: str
    # Chronological IMU samples for the full fetch window
    samples: list[ImuSample] = Field(..., min_length=1)


class InferenceResponse(BaseModel):
    device_sn: str
    events: list[BehaviorEvent]
    scratch_count: int    # convenience: number of scratch events in this batch


_LABEL_NAMES = {
    BehaviorLabel.UNKNOWN:  "unknown",
    BehaviorLabel.MOVEMENT: "movement",
    BehaviorLabel.SLEEP:    "sleep",
    BehaviorLabel.SCRATCH:  "scratch",
}


@router.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    try:
        clf = get_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    samples_sorted = sorted(req.samples, key=lambda s: s.ts_ms)
    base_ts_ms = samples_sorted[0].ts_ms

    data = np.array(
        [[s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in samples_sorted],
        dtype=np.float32,
    )

    raw_events = clf.predict(data, base_ts_ms)

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

    scratch_count = sum(
        1 for e in events if e.behavior_type == BehaviorLabel.SCRATCH
    )

    return InferenceResponse(
        device_sn=req.device_sn,
        events=events,
        scratch_count=scratch_count,
    )
