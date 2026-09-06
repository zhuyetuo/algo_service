"""
训练任务表：label_infra 提交 /train 后立刻拿到 job_id，之后靠轮询
GET /train/{job_id} 查状态——跟 db/client.py 里 device_sync_state 那些表
一样用原始 SQL 建表（这个仓库里就没有用 ORM 声明式模型管这些运维表，
沿用现有风格，不额外引入一套新写法）。
"""

import json
import time

from sqlalchemy import text

from db.client import AsyncSessionLocal, engine

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


async def init_jobs_table() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS label_train_jobs (
                id           BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
                status       VARCHAR(16)  NOT NULL DEFAULT 'queued',
                dataset_spec TEXT         NOT NULL,
                model_type   VARCHAR(32)  NOT NULL DEFAULT 'rf',
                tag          VARCHAR(64),
                model_version VARCHAR(64),
                model_path   VARCHAR(500),
                metrics      TEXT,
                error        TEXT,
                created_at   BIGINT       NOT NULL,
                started_at   BIGINT,
                finished_at  BIGINT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))


async def create_job(dataset_spec: dict, model_type: str, tag: str | None) -> int:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("""
                INSERT INTO label_train_jobs (status, dataset_spec, model_type, tag, created_at)
                VALUES (:status, :dataset_spec, :model_type, :tag, :created_at)
            """),
            {
                "status": STATUS_QUEUED,
                "dataset_spec": json.dumps(dataset_spec, ensure_ascii=False),
                "model_type": model_type,
                "tag": tag,
                "created_at": int(time.time()),
            },
        )
        await db.commit()
        return result.lastrowid


async def mark_running(job_id: int) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("UPDATE label_train_jobs SET status=:status, started_at=:ts WHERE id=:id"),
            {"status": STATUS_RUNNING, "ts": int(time.time()), "id": job_id},
        )
        await db.commit()


async def mark_done(job_id: int, model_version: str, model_path: str, metrics: dict) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("""
                UPDATE label_train_jobs
                SET status=:status, model_version=:mv, model_path=:mp,
                    metrics=:metrics, finished_at=:ts
                WHERE id=:id
            """),
            {
                "status": STATUS_DONE,
                "mv": model_version,
                "mp": model_path,
                "metrics": json.dumps(metrics, ensure_ascii=False),
                "ts": int(time.time()),
                "id": job_id,
            },
        )
        await db.commit()


async def mark_failed(job_id: int, error: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("UPDATE label_train_jobs SET status=:status, error=:error, finished_at=:ts WHERE id=:id"),
            {"status": STATUS_FAILED, "error": error[:4000], "ts": int(time.time()), "id": job_id},
        )
        await db.commit()


async def get_job(job_id: int) -> dict | None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT * FROM label_train_jobs WHERE id=:id"), {"id": job_id}
        )
        row = result.mappings().first()
        if row is None:
            return None
        row = dict(row)
        row["dataset_spec"] = json.loads(row["dataset_spec"]) if row["dataset_spec"] else None
        row["metrics"] = json.loads(row["metrics"]) if row["metrics"] else None
        return row
