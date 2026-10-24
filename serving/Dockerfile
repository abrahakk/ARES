FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY rl/networks ./rl/networks
COPY rl/environments ./rl/environments
COPY rl/communication ./rl/communication
COPY rl/__init__.py ./rl/__init__.py
COPY robots/ ./robots/
COPY coordination/ ./coordination/
COPY missions/ ./missions/
COPY monitoring/metrics.py ./monitoring/metrics.py
COPY monitoring/__init__.py ./monitoring/__init__.py
COPY config/ ./config/
COPY models/onnx/ ./models/onnx/

ENV PYTHONUNBUFFERED=1 \
    ARES_MODEL_PATH=/app/models/onnx/ares_policy.onnx

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER nobody

CMD ["uvicorn", "serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
