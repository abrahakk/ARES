FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rl/ ./rl/
COPY robots/ ./robots/
COPY simulation/ ./simulation/
COPY coordination/ ./coordination/
COPY config/ ./config/
COPY data/ ./data/
COPY training/ ./training/
COPY evaluation/ ./evaluation/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:${PYTHONPATH}"


VOLUME ["/app/models", "/app/logs", "/app/data"]

ENTRYPOINT ["python3"]
CMD ["scripts/train.py", "--config", "config/training.yaml"]
