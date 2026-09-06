"""
label_infra 后端调用的两个接口：

  POST /infer            同步：给一个 NAS 相对路径的 IMU CSV，立刻返回预标注片段
                          （标注员点"AI预标注"时调，不经过训练任务队列）
  POST /train             提交训练任务，立刻返回 job_id，训练在后台跑
  GET  /train/{job_id}   轮询训练任务状态（跟 label_infra 那边"提交扫描任务→
                          轮询进度条"是同一个模式）

跟现有 /api/v1/inference/predict（原始采样点走 HTTP body、给设备端实时上报用）
是两条不同的路：这里的样本文件本来就在 label_infra 和 algo_service 共享的
NAS 上，没必要把整个 CSV 塞进请求体里再传一遍。
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from modules.inference.model import BehaviorLabel, get_classifier
from modules.label_pipeline import jobs_db
from modules.label_pipeline.csv_loader import ImuCsvError, load_imu_csv
from modules.label_pipeline.trainer import run_training_job

router = APIRouter()
_logger = logging.getLogger(__name__)

_LABEL_NAMES = {
    BehaviorLabel.UNKNOWN:  "unknown",
    BehaviorLabel.MOVEMENT: "movement",
    BehaviorLabel.SLEEP:    "sleep",
    BehaviorLabel.SCRATCH:  "scratch",
}


# ── /infer ──────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    path: str = Field(..., description="NAS_ROOT 下的相对路径，指向一份 IMU CSV")
    sample_id: int | None = Field(None, description="label_infra 那边的 sample.id，仅用于日志/回显关联，不参与推理")


class InferEvent(BaseModel):
    behavior_type: int
    behavior_name: str
    start_time: int   # 相对 CSV 第一行的 UTC 毫秒时间戳（绝对值，跟 CSV 里的时间戳同一个基准）
    end_time: int
    confidence: float


class InferResponse(BaseModel):
    sample_id: int | None
    path: str
    events: list[InferEvent]
    scratch_count: int


@router.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    try:
        data, base_ts_ms = load_imu_csv(req.path)
    except ImuCsvError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        clf = get_classifier()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    raw_events = clf.predict(data, base_ts_ms)
    events = [
        InferEvent(
            behavior_type=e["behavior_type"],
            behavior_name=_LABEL_NAMES.get(e["behavior_type"], "unknown"),
            start_time=e["start_time"],
            end_time=e["end_time"],
            confidence=e["confidence"],
        )
        for e in raw_events
    ]
    scratch_count = sum(1 for e in events if e.behavior_type == BehaviorLabel.SCRATCH)

    return InferResponse(
        sample_id=req.sample_id, path=req.path, events=events, scratch_count=scratch_count,
    )


# ── /train ──────────────────────────────────────────────────────────────

class DatasetSpec(BaseModel):
    date: str = Field(..., description="跟 train_custom.sh --date 一致，如 2026_8_20")
    extra_date: list[str] = Field(default_factory=list, description="跟 --extra_date 一致，格式 DATE:HZ，可传多个")
    missing_strategy: str | None = Field(None, description="none/drop/ffill/drop_window，缺省用 train_custom.sh 默认值")
    skip_syn: bool = Field(False, description="跳过合成数据方案B，只训练方案A（纯标注）")


class TrainRequest(BaseModel):
    dataset: DatasetSpec
    model_type: str = Field("rf", description="跟 train_custom.sh --model 一致，比如 rf/extra_trees/hist_gb")
    tag: str | None = Field(None, description="留空时 train_custom.sh 会按 missing_strategy 自动生成")


class TrainSubmitResponse(BaseModel):
    job_id: int
    status: str


class TrainStatusResponse(BaseModel):
    job_id: int
    status: str
    model_type: str
    tag: str | None
    model_version: str | None
    model_path: str | None
    metrics: dict | None
    error: str | None
    created_at: int
    started_at: int | None
    finished_at: int | None


@router.post("/train", response_model=TrainSubmitResponse)
async def submit_train(req: TrainRequest):
    job_id = await jobs_db.create_job(req.dataset.model_dump(), req.model_type, req.tag)
    asyncio.create_task(run_training_job(job_id, req.dataset.model_dump(), req.model_type, req.tag))
    return TrainSubmitResponse(job_id=job_id, status=jobs_db.STATUS_QUEUED)


@router.get("/train/{job_id}", response_model=TrainStatusResponse)
async def get_train_status(job_id: int):
    job = await jobs_db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"训练任务 #{job_id} 不存在")
    return TrainStatusResponse(
        job_id=job["id"],
        status=job["status"],
        model_type=job["model_type"],
        tag=job["tag"],
        model_version=job["model_version"],
        model_path=job["model_path"],
        metrics=job["metrics"],
        error=job["error"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        finished_at=job["finished_at"],
    )
