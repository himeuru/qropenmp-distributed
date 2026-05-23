"""Submit a random matrix to the QR service and wait for the result.

Usage:
    python submit.py --n 512 --threads 4
    python submit.py --api http://localhost:8000 --n 1024 --threads 8

Only depends on numpy + stdlib. The matrix is generated locally, base64-encoded,
and posted as JSON; the script then polls until the worker finishes.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

import numpy as np


def random_matrix_bytes(n: int, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    A = rng.uniform(-10.0, 10.0, size=(n, n)).astype(np.float64, copy=False)
    return A.tobytes()


def http_post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def http_get(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="QR service client")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=512, help="Matrix dimension")
    parser.add_argument("--threads", type=int, default=0, help="OpenMP threads (0 = engine default)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the matrix")
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Polling interval in seconds")
    parser.add_argument("--timeout", type=float, default=600.0, help="Overall timeout in seconds")
    args = parser.parse_args()

    print(f"Generating {args.n}x{args.n} matrix (seed={args.seed})...")
    matrix_bytes = random_matrix_bytes(args.n, args.seed)
    matrix_b64 = base64.b64encode(matrix_bytes).decode()

    print(f"POST {args.api}/jobs  (threads={args.threads}, {len(matrix_bytes)//1024} KiB)")
    submitted = http_post(
        f"{args.api}/jobs",
        {"n": args.n, "threads": args.threads, "matrix_b64": matrix_b64},
    )
    job_id = submitted["job_id"]
    print(f"job_id = {job_id}")

    deadline = time.monotonic() + args.timeout
    last_status = None
    while True:
        if time.monotonic() > deadline:
            print(f"Timed out after {args.timeout}s", file=sys.stderr)
            return 2

        try:
            status = http_get(f"{args.api}/jobs/{job_id}")
        except urllib.error.URLError as exc:
            print(f"  poll error: {exc}", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        if status["status"] != last_status:
            print(f"  status: {status['status']}")
            last_status = status["status"]

        if status["status"] == "finished":
            result = status["result"]
            print()
            print(f"  n            = {result['n']}")
            print(f"  threads used = {result['threads_used']}")
            print(f"  elapsed      = {result['elapsed_ms']:.2f} ms")
            print(f"  diag(R)[:5]  = {[round(x, 4) for x in result['diag_r_head'][:5]]}")
            return 0

        if status["status"] == "failed":
            print(f"Job failed:\n{status.get('error') or '(no error info)'}", file=sys.stderr)
            return 1

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
