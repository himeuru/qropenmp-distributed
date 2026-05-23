"""HTTP gateway in front of the QR worker pool.

Clients submit a matrix as base64-encoded little-endian doubles. The request
is pushed to a Redis-backed RQ queue; a worker container picks it up, runs the
C++ engine, and stores the result back in Redis. Clients poll GET /jobs/{id}
until status is finished or failed.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from rq.command import send_stop_job_command
from rq.exceptions import InvalidJobOperation, NoSuchJobError
from rq.job import Job
from rq.registry import FinishedJobRegistry, StartedJobRegistry

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
QUEUE_NAME = "qr"
JOB_TIMEOUT = os.environ.get("JOB_TIMEOUT", "30m")
MAX_N = 8192
MAX_PENDING = int(os.environ.get("MAX_PENDING_JOBS", "4"))

redis_conn = Redis.from_url(REDIS_URL)
queue = Queue(QUEUE_NAME, connection=redis_conn)
started_registry = StartedJobRegistry(QUEUE_NAME, connection=redis_conn)
finished_registry = FinishedJobRegistry(QUEUE_NAME, connection=redis_conn)


def _check_capacity() -> None:
    """Bound the work the queue will accept so a flood of heavy matrices
    can't pile up and starve the host of CPU."""
    in_flight = queue.count + started_registry.count
    if in_flight >= MAX_PENDING:
        raise HTTPException(
            429,
            f"too many jobs in flight ({in_flight}/{MAX_PENDING}); wait for some to finish",
        )

app = FastAPI(
    title="QR Compute Service",
    description="Distributed QR decomposition (Householder + OpenMP) behind a job queue.",
    version="0.1.0",
)


class JobRequest(BaseModel):
    n: int = Field(..., gt=0, le=MAX_N, description="Matrix dimension (n x n)")
    threads: int = Field(0, ge=0, le=128, description="OpenMP threads, 0 = engine default")
    matrix_b64: str = Field(..., description="Base64-encoded row-major matrix, n*n little-endian float64")


class RandomJobRequest(BaseModel):
    n: int = Field(..., gt=0, le=MAX_N, description="Matrix dimension (n x n)")
    threads: int = Field(0, ge=0, le=128, description="OpenMP threads, 0 = engine default")
    seed: int = Field(42, description="RNG seed for the generated matrix")


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


def _summarise(job_id: str) -> dict | None:
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return None
    args = job.args or ()
    return {
        "id": job_id,
        "status": job.get_status(),
        "n": args[0] if len(args) > 0 else None,
        "threads": args[1] if len(args) > 1 else None,
        "enqueued_at": _isoformat(job.enqueued_at),
        "started_at": _isoformat(job.started_at),
    }


@app.get("/info")
def info() -> dict:
    """Diagnostic snapshot: what CPU resources the API sees and what the most
    recent finished job reported about the worker's OpenMP environment.
    Useful when the speedup numbers look worse than expected — the
    `observed_team_size` from the engine tells you whether OpenMP actually
    spun up the threads you asked for."""
    import sys as _sys

    api_block = {
        "cpu_count_logical": os.cpu_count(),
        "available_cpus": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        ),
        "python": _sys.version.split()[0],
        "max_pending_jobs": MAX_PENDING,
        "redis_url": REDIS_URL,
    }

    worker_block: dict | None = None
    try:
        for jid in finished_registry.get_job_ids():
            job = Job.fetch(jid, connection=redis_conn)
            if job.result and isinstance(job.result, dict) and "observed_team_size" in job.result:
                worker_block = {
                    "last_job_id": jid,
                    "n": job.result.get("n"),
                    "threads_requested": job.result.get("threads_used"),
                    "omp_max_threads": job.result.get("omp_max_threads"),
                    "omp_num_procs": job.result.get("omp_num_procs"),
                    "observed_team_size": job.result.get("observed_team_size"),
                    "elapsed_ms": job.result.get("elapsed_ms"),
                }
                break
    except Exception as exc:
        worker_block = {"error": str(exc)}

    return {"api": api_block, "worker_last_job": worker_block}


@app.get("/queue")
def queue_state() -> dict:
    """Snapshot of what the workers are currently doing. The UI polls this
    every couple of seconds to render a live view."""
    running = [j for j in (_summarise(i) for i in started_registry.get_job_ids()) if j]
    pending = [j for j in (_summarise(i) for i in queue.job_ids) if j]
    return {
        "cap": MAX_PENDING,
        "in_flight": len(running) + len(pending),
        "running": running,
        "queued": pending,
    }


@app.post("/jobs", response_model=JobSubmitted, status_code=202)
def submit_job(req: JobRequest) -> JobSubmitted:
    _check_capacity()
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


@app.post("/jobs/random", response_model=JobSubmitted, status_code=202)
def submit_random_job(req: RandomJobRequest) -> JobSubmitted:
    """Enqueue a job where the worker generates the matrix itself.

    Avoids shipping megabytes of base64 over the wire for the common case
    where the caller just wants `n`-sized random benchmark data.
    """
    _check_capacity()
    job = queue.enqueue(
        "worker.decompose_random",
        req.n,
        req.threads,
        req.seed,
        job_timeout=JOB_TIMEOUT,
        result_ttl=3600,
    )
    return JobSubmitted(job_id=job.id, status=job.get_status())


@app.delete("/jobs/{job_id}", status_code=204, response_class=Response)
def cancel_job(job_id: str):
    """Cancel a job and remove it from Redis so its slot in the cap is freed.

    Queued jobs are simply pulled off the queue. Running jobs are asked to
    stop via the worker control channel; the engine subprocess will be killed
    by the worker. Either way the job is then deleted so /health stops
    counting it as in-flight.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return  # already gone — idempotent

    if job.get_status() == "started":
        try:
            send_stop_job_command(redis_conn, job_id)
        except (InvalidJobOperation, Exception):
            pass  # best effort; we still delete below

    try:
        job.cancel()
    except InvalidJobOperation:
        pass
    try:
        job.delete()
    except Exception:
        pass


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
        error=_short_error(job.exc_info) if job.is_failed else None,
    )


def _short_error(exc_info: str | None) -> str | None:
    """Return the last meaningful line of a Python traceback so the UI can
    show a one-liner like 'JobTimeoutException: Task exceeded …' instead of
    twenty lines of stack frames."""
    if not exc_info:
        return None
    lines = [ln.strip() for ln in exc_info.strip().split("\n") if ln.strip()]
    return lines[-1] if lines else exc_info


def _isoformat(value) -> str | None:
    return value.isoformat() if value else None


# Web UI: serve the static page at /ui/ and redirect bare / to it. Keep the
# JSON API at the top of the router so /docs and /openapi.json still resolve.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def _root_redirect():
        return RedirectResponse(url="/ui/")
