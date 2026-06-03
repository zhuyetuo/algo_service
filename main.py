import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy import text

from config import settings
from db.client import init_db, close_db, AsyncSessionLocal
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
    result = {"status": "ok", "postgres": "ok", "tdengine": "ok"}

    # 检测 PostgreSQL
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
    except Exception as e:
        result["postgres"] = str(e)
        result["status"] = "degraded"

    # 检测 TDengine REST API
    try:
        url = f"http://{settings.td_host}:{settings.td_port}/rest/sql"
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                url,
                content="SELECT SERVER_VERSION()",
                auth=(settings.td_user, settings.td_password),
            )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")
    except Exception as e:
        result["tdengine"] = str(e)
        result["status"] = "degraded"

    return result
