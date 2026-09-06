"""
后台执行训练任务：调用 imu_train 子模块里的 train_custom.sh，跑完之后从它自己
打印的"模型路径:"那几行里解析出产出的 .pkl 路径（比自己重新拼一遍
DATASET_TAG/HZ/MODEL_TYPE 的目录规则可靠——脚本内部命名规则变了这里也不用跟着改）。

不在这里做"训练完自动切换线上模型"这件事：model_path/metrics 只是写回
label_train_jobs 表供 label_infra 轮询展示，真正上线换模型仍然是运维手动
改 weights/ml_rf.pkl（或者以后单独做一个"发布"接口），训练和上线分开、
避免一次训练自动覆盖线上正在跑的模型。
"""

import asyncio
import json
import logging
import os
import re
import time

import httpx

from config import settings
from modules.label_pipeline import jobs_db

_logger = logging.getLogger(__name__)

_MODEL_PATH_RE = re.compile(r"^\s*(纯标注|带合成):\s*(\S+\.pkl)\s*$", re.MULTILINE)


def _imu_train_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, settings.imu_train_repo_dir)


def _build_command(dataset_spec: dict, model_type: str, tag: str | None) -> list[str]:
    cmd = ["bash", "train_custom.sh", "--date", dataset_spec["date"]]
    for extra in dataset_spec.get("extra_date", []):
        cmd += ["--extra_date", extra]
    if dataset_spec.get("missing_strategy"):
        cmd += ["--missing_strategy", dataset_spec["missing_strategy"]]
    if model_type:
        cmd += ["--model", model_type]
    if tag:
        cmd += ["--tag", tag]
    if dataset_spec.get("skip_syn"):
        cmd += ["--skip_syn"]
    return cmd


def _parse_model_path(stdout: str) -> str | None:
    matches = dict(_MODEL_PATH_RE.findall(stdout))
    return matches.get("带合成") or matches.get("纯标注")


def _load_metrics(pkl_path: str) -> dict:
    json_path = os.path.splitext(pkl_path)[0] + ".json"
    if not os.path.exists(json_path):
        return {}
    try:
        return json.loads(open(json_path, encoding="utf-8").read())
    except Exception as e:  # noqa: BLE001 元数据读取失败不应该让整个任务判失败
        _logger.warning("读取模型元数据失败 %s: %s", json_path, e)
        return {}


async def _notify_callback(job_id: int, status: str, payload: dict) -> None:
    if not settings.label_infra_callback_url:
        return
    url = settings.label_infra_callback_url.rstrip("/") + f"/{job_id}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={"job_id": job_id, "status": status, **payload})
    except Exception as e:  # noqa: BLE001 回调失败不影响任务本身的状态，label_infra 还能轮询兜底
        _logger.warning("训练完成回调 label_infra 失败（不影响任务状态，label_infra 可以轮询兜底）: %s", e)


async def run_training_job(job_id: int, dataset_spec: dict, model_type: str, tag: str | None) -> None:
    await jobs_db.mark_running(job_id)
    cmd = _build_command(dataset_spec, model_type, tag)
    _logger.info("训练任务 #%d 启动: %s", job_id, " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=_imu_train_dir(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await proc.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            raise RuntimeError(f"train_custom.sh 退出码 {proc.returncode}，末尾输出:\n{stdout[-2000:]}")

        model_path = _parse_model_path(stdout)
        if not model_path:
            raise RuntimeError(f"训练成功但没能从输出里解析出模型路径，末尾输出:\n{stdout[-2000:]}")

        metrics = _load_metrics(model_path)
        model_version = tag or f"{model_type}_{int(time.time())}"

        await jobs_db.mark_done(job_id, model_version, model_path, metrics)
        _logger.info("训练任务 #%d 完成: %s", job_id, model_path)
        await _notify_callback(job_id, jobs_db.STATUS_DONE, {
            "model_version": model_version, "model_path": model_path, "metrics": metrics,
        })

    except Exception as e:  # noqa: BLE001 后台任务异常不能让进程崩，落盘状态即可
        _logger.exception("训练任务 #%d 失败", job_id)
        await jobs_db.mark_failed(job_id, str(e))
        await _notify_callback(job_id, jobs_db.STATUS_FAILED, {"error": str(e)})
