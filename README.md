# qropenmp-distributed

QR decomposition (Householder + OpenMP) behind an HTTP API. Matrices are
submitted to FastAPI, queued in Redis, and picked up by a pool of workers
that run the C++ engine. Stack starts with one `docker compose up`.

The C++ engine is the one from [qropenmp](https://github.com/himeuru/qropenmp);
this repo wraps it into a small distributed service with a web UI.

## Architecture

```mermaid
flowchart LR
    Client -- HTTP --> API[FastAPI]
    API -- enqueue --> R[(Redis)]
    R -- pull --> W[worker]
    W -- subprocess --> E[qr-engine<br/>C++ + OpenMP]
    E --> W
    W --> R
    Client -- GET /jobs/:id --> API
```

Three Compose services: `api`, `worker`, `redis`. A `client` profile is
there for the CLI.

## Run

```bash
git clone https://github.com/himeuru/qropenmp-distributed.git
cd qropenmp-distributed
docker compose up -d --build
```

Open `http://localhost:8000/` for the web UI. Swagger at `/docs`.

Scale the worker pool:

```bash
docker compose up -d --scale worker=4
```

CLI alternative (no Python on the host needed):

```bash
docker compose run --rm client --n 512 --threads 4
```

## Web UI

Three tabs:

- **Single job** — pick `n` / `threads` / seed, submit one matrix, see
  the elapsed time and the diagonal of R for verification.
- **Benchmark** — run the same matrix at a list of thread counts and
  get a table plus a speedup chart with Amdahl overlay.
- **Reports** — four charts aggregated from your job history: Amdahl
  curves with real points overlaid, real-vs-Amdahl speedup per size,
  efficiency heatmap, and execution time on a log scale. **Run sweep**
  fires a `sizes × threads` grid so every chart fills up in one click.
  A **System info** panel shows what OpenMP actually got from the
  container (cores seen, requested team size, observed team size).

In-flight jobs survive a page refresh — the UI picks them back up via
`localStorage` instead of letting you accidentally submit duplicates.
The persistent **Cancel** button really cancels the job server-side
(queued or running) and frees its slot in the cap.

## API

| Method | Path | Body | Notes |
|---|---|---|---|
| GET    | `/health`        | — | service + queue status |
| GET    | `/info`          | — | API CPU count + OpenMP readout from the latest worker run |
| GET    | `/queue`         | — | running + queued jobs (used by the live panel) |
| POST   | `/jobs`          | `{n, threads, matrix_b64}` | client supplies the matrix |
| POST   | `/jobs/random`   | `{n, threads, seed}`       | worker generates the matrix |
| GET    | `/jobs/{id}`     | — | status + result |
| DELETE | `/jobs/{id}`     | — | cancel a queued or running job |

The result includes `elapsed_ms`, `threads_used`, the diagonal of R for
verification, and diagnostic fields from the engine
(`omp_num_procs`, `omp_max_threads`, `observed_team_size`). Full Q and
R aren't returned over the wire.

Submissions get `HTTP 429` when `queued + running` reaches the cap.

## Configuration

| Env var | Default | What it does |
|---|---|---|
| `REDIS_URL`         | `redis://redis:6379` | Redis connection |
| `MAX_PENDING_JOBS`  | `4`                  | Cap on queued + running jobs before 429 |
| `JOB_TIMEOUT`       | `30m`                | Per-job death penalty inside RQ |

Set them under the `api` / `worker` services in `docker-compose.yml`.

## Wire format

`/jobs` takes matrices as base64 of little-endian `float64`, row-major,
exactly `n*n*8` bytes. Internally the worker reframes that as a binary
stream to the engine:

```
stdin:  int32 n, int32 threads, double matrix[n*n]
stdout: double elapsed_ms,
        int32 n, int32 threads_requested,
        int32 omp_max_threads, int32 omp_num_procs, int32 observed_team_size,
        double diag_R[n]
```

## Layout

```
engine/      C++ kernel + CLI binary
api/
  static/    single-page UI (HTML / CSS / SVG, no build step)
  app.py     FastAPI routes
worker/      RQ worker, shells out to the engine
client/      containerised Python CLI
.github/     CI: builds images and runs an end-to-end smoke test
```

Engine builds standalone:

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
```

## Notes

- **Docker Desktop / WSL2 performance.** Memory-bound kernels (QR is one) run
  about 2–3× slower inside a Linux container on Windows than they would on
  the host directly. The *relative* speedup numbers stay honest — both the
  baseline and the parallel runs pay the same penalty — but absolute times
  look bigger than they would natively.
- **Diagnosing OpenMP weirdness.** If a thread count doesn't produce the
  speedup you expect, open Reports → System info. The `observed_team_size`
  field is what `omp_get_num_threads()` reports from inside the parallel
  region; if it's lower than `threads_requested`, libgomp isn't giving you
  the team you asked for and the speedup numbers will be off.
- **Large matrices.** `n = 8192` with `threads = 1` is multi-minute work
  even on a fast machine. Bump `JOB_TIMEOUT` in `docker-compose.yml` if
  jobs hit the 30-minute death penalty, or just use higher thread counts.
- **Logs.** `docker compose logs -f worker` is usually where the real
  failure reason lives when a job ends as `failed`.
