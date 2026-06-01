from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from db.client import init_db, close_db
from log import setup_logging

setup_logging()
from modules.inference.handler import router as inference_router
from modules.assessment.evaluator import router as assessment_router
from scheduler.jobs import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    yield
    stop_scheduler()
    await close_db()


app = FastAPI(
    title="Algo Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(inference_router, prefix="/api/v1/inference", tags=["inference"])
app.include_router(assessment_router, prefix="/api/v1/assessment", tags=["assessment"])


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}
