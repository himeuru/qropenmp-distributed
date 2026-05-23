# Stage 1 — build the C++ engine.
FROM debian:bookworm-slim AS engine-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ cmake make \
    && rm -rf /var/lib/apt/lists/*

COPY engine /engine
RUN cmake -S /engine -B /build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j

# Stage 2 — Python worker runtime with the engine binary copied in.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=engine-build /build/qr-engine /usr/local/bin/qr-engine

WORKDIR /app
COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker/ .

CMD ["rq", "worker", "qr", "--url", "redis://redis:6379"]
