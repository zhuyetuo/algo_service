from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.client import AsyncSessionLocal, get_session

router = APIRouter()


class AssessmentResult(BaseModel):
    user_id: str
    score: float
    level: str  # e.g. "normal" | "mild" | "moderate" | "severe"


@router.get("/report/{user_id}", response_model=AssessmentResult)
async def get_assessment(
    user_id: str,
    db: AsyncSession = Depends(get_session),
):
    # TODO: fetch latest behavior data, compare against baseline, compute score
    return AssessmentResult(user_id=user_id, score=0.0, level="normal")


async def run_batch_assessment() -> None:
    """Scheduled job: compute assessments for all active users."""
    async with AsyncSessionLocal() as db:
        # TODO: iterate users, call compute logic, persist results
        pass
