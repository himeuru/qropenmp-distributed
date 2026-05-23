"""RQ worker entry point.

`decompose` is the task function that the API enqueues. It shells out to the
C++ engine binary with the matrix on stdin and parses the binary result back.
The Dockerfile installs `qr-engine` into /usr/local/bin so we just call it by
name; for local development the ENGINE_BIN env var can point elsewhere.
"""

from __future__ import annotations

import os
import struct
import subprocess

import numpy as np

ENGINE_BIN = os.environ.get("ENGINE_BIN", "/usr/local/bin/qr-engine")

# Header layout matches main.cpp:
#   double elapsed_ms, int32 n, int32 threads_used → 16 bytes
_HEADER_FMT = "<dii"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def decompose(matrix_bytes: bytes, n: int, threads: int) -> dict:
    if len(matrix_bytes) != n * n * 8:
        raise ValueError(f"matrix bytes ({len(matrix_bytes)}) != n*n*8 ({n * n * 8})")

    payload = struct.pack("<ii", n, threads) + matrix_bytes

    proc = subprocess.run(
        [ENGINE_BIN],
        input=payload,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"qr-engine exited with {proc.returncode}: {proc.stderr.decode(errors='replace')}"
        )

    out = proc.stdout
    if len(out) < _HEADER_SIZE:
        raise RuntimeError(f"engine returned {len(out)} bytes, expected at least {_HEADER_SIZE}")

    elapsed_ms, n_out, threads_used = struct.unpack(_HEADER_FMT, out[:_HEADER_SIZE])
    diag_bytes = out[_HEADER_SIZE : _HEADER_SIZE + n_out * 8]
    diag_r = list(struct.unpack(f"<{n_out}d", diag_bytes))

    return {
        "n": n_out,
        "threads_used": threads_used,
        "elapsed_ms": elapsed_ms,
        "diag_r_head": diag_r[:8],
        "diag_r_tail": diag_r[-8:],
        "diag_r_count": len(diag_r),
    }


def decompose_random(n: int, threads: int, seed: int = 42) -> dict:
    """Generate a random n x n matrix on the worker, then decompose it.

    Keeps large payloads off the network and the queue — only n/threads/seed
    travel from the client, and the matrix is materialised right next to the
    engine. Useful for the web UI's "submit" button and benchmark mode.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-10.0, 10.0, size=(n, n)).astype(np.float64, copy=False)
    return decompose(matrix.tobytes(), n, threads)
