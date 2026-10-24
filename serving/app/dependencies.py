from __future__ import annotations

from fastapi import Request

from coordination.mission_manager import MissionManager
from coordination.redis_client import RedisClient
from coordination.state_manager import StateManager
from coordination.task_manager import TaskManager
from serving.inference.model_manager import ModelManager


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_redis_client(request: Request) -> RedisClient:
    return request.app.state.redis_client


def get_state_manager(request: Request) -> StateManager:
    return request.app.state.state_manager


def get_task_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager


def get_mission_manager(request: Request) -> MissionManager:
    return request.app.state.mission_manager
