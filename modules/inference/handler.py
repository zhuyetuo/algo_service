from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.client import get_session
from modules.inference.model import get_model

router = APIRouter()


class InferenceRequest(BaseModel):
    session_id: str
    # Add frame/video payload fields here (e.g. base64 frames or a file key)


class InferenceResponse(BaseModel):
    session_id: str
    behavior: str
    confidence: float


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    req: InferenceRequest,
    db: AsyncSession = Depends(get_session),
):
    model = get_model()
    try:
        result = model.predict([])  # pass real frames here
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Model not implemented yet")
    return InferenceResponse(
        session_id=req.session_id,
        behavior=result.get("behavior", "unknown"),
        confidence=result.get("confidence", 0.0),
    )
