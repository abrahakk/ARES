rom __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from coordination.mission_manager import MissionManager
from coordination.redis_client import RedisClient
from coordination.state_manager import StateManager
from coordination.task_manager import TaskManager
from serving.app.routes import inference, missions, robots
from serving.app.schemas import HealthResponse
from serving.inference.model_manager import ModelManager

logger = logging.getLogger(__name__)


def _load_system_config() -> dict:
    config_path = os.environ.get("ARES_SYSTEM_CONFIG", "config/system.yaml")
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("system config not found at %s - using built-in defaults", config_path)
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    system_cfg = _load_system_config()
    api_cfg = system_cfg.get("api", {})

    model_path = os.environ.get(
        "ARES_MODEL_PATH",
        system_cfg.get("model_registry", {}).get("onnx_root", "models/onnx") + "/ares_policy.onnx",
    )

    model_manager = ModelManager(model_path)
    try:
        model_manager.load()
    except Exception as exc: 
        logger.error("failed to load model from %s: %s - /predict will return 503 until this is fixed", model_path, exc)

    redis_client = RedisClient()
    state_manager = StateManager(redis_client)
    task_manager = TaskManager(redis_client)
    mission_manager = MissionManager(redis_client)

    app.state.model_manager = model_manager
    app.state.redis_client = redis_client
    app.state.state_manager = state_manager
    app.state.task_manager = task_manager
    app.state.mission_manager = mission_manager

    logger.info("ARES inference service started (model_loaded=%s)", model_manager.is_loaded)
    yield

    await redis_client.close()
    logger.info("ARES inference service shut down")


app = FastAPI(
    title="ARES Inference API",
    description="Serves the ARES cooperative exploration policy for prediction, and exposes "
                 "robot state and mission lifecycle endpoints backed by the coordination layer.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(inference.router, tags=["inference"])
app.include_router(robots.router, tags=["robots"])
app.include_router(missions.router, tags=["missions"])

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    model_manager: ModelManager = app.state.model_manager
    return HealthResponse(status="ok", model_loaded=model_manager.is_loaded)

