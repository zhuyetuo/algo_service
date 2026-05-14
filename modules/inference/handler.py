from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

import numpy as np

from db.client import get_session
from modules.inference.model import get_classifier, AXES

router = APIRouter()


class ImuSample(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class InferenceRequest(BaseModel):
    device_id: str
    # Raw IMU samples for the full fetch window (order matters — chronological)
    samples: list[ImuSample] = Field(..., min_length=1)


class InferenceResponse(BaseModel):
    device_id: str
    dominant_behavior: str
    distribution: dict[str, int]   # {"sleeping": 210, "walking": 90, ...}
    window_count: int


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    req: InferenceRequest,
    db: AsyncSession = Depends(get_session),
):
    try:
        clf = get_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = np.array([[s.ax, s.ay, s.az, s.gx, s.gy, s.gz] for s in req.samples],
                    dtype=np.float32)

    result = clf.predict_batch(data)

    return InferenceResponse(
        device_id=req.device_id,
        dominant_behavior=result["dominant"],
        distribution=result["distribution"],
        window_count=result["window_count"],
    )
