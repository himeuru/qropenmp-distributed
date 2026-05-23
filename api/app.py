"""HTTP gateway in front of the QR worker pool.

Clients submit a matrix as base64-encoded little-endian doubles. The request
is pushed to a Redis-backed RQ queue; a worker container picks it up, runs the
C++ engine, and stores the result back in Redis. Clients poll GET /jobs/{id}
until status is finished or failed.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
QUEUE_NAME = "qr"
JOB_TIMEOUT = "10m"
MAX_N = 8192

redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(QUEUE_NAME, connection=redis_conn)

app = FastAPI(
    title="QR Compute Service",
    description="Distributed QR decomposition (Householder + OpenMP) behind a job queue.",
    version="0.1.0",
)


class JobRequest(BaseModel):
    n: int = Field(..., gt=0, le=MAX_N, description="Matrix dimension (n x n)")
    threads: int = Field(0, ge=0, le=128, description="OpenMP threads, 0 = engine default")
    matrix_b64: str = Field(..., description="Base64-encoded row-major matrix, n*n little-endian float64")


class JobSubmitted(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    queued_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    try:
        redis_conn.ping()
    except Exception as exc:
        raise HTTPException(503, f"redis unreachable: {exc}")
    return {"status": "ok", "queue": QUEUE_NAME, "pending": str(queue.count)}


@app.post("/jobs", response_model=JobSubmitted, status_code=202)
def submit_job(req: JobRequest) -> JobSubmitted:
    try:
        matrix_bytes = base64.b64decode(req.matrix_b64, validate=True)
    except Exception as exc:
        raise HTTPException(400, f"invalid base64: {exc}")

    expected_size = req.n * req.n * 8
    if len(matrix_bytes) != expected_size:
        raise HTTPException(
            400,
            f"matrix size mismatch: got {len(matrix_bytes)} bytes, expected n*n*8 = {expected_size}",
        )

    job = queue.enqueue(
        "worker.decompose",
        matrix_bytes,
        req.n,
        req.threads,
        job_timeout=JOB_TIMEOUT,
        result_ttl=3600,
    )
    return JobSubmitted(job_id=job.id, status=job.get_status())


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(404, f"job {job_id} not found")

    return JobStatus(
        job_id=job.id,
        status=job.get_status(),
        queued_at=_isoformat(job.enqueued_at),
        started_at=_isoformat(job.started_at),
        finished_at=_isoformat(job.ended_at),
        result=job.result if job.is_finished else None,
        error=str(job.exc_info) if job.is_failed and job.exc_info else None,
    )


def _isoformat(value) -> str | None:
    return value.isoformat() if value else None
