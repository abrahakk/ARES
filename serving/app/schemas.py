from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from rl.environments.observation_space import OCCUPANCY_GRID_DIM


class RobotPose(BaseModel):
    x: float
    y: float
    theta: float = Field(ge=-3.15, le=3.15)


class MessagePayload(BaseModel):
    sender_id: str
    message_type: str
    payload: Dict = Field(default_factory=dict)
    timestamp: Optional[float] = None


class Observation(BaseModel):
    robot_id: str
    pose: RobotPose
    local_occupancy: List[float] = Field(min_length=OCCUPANCY_GRID_DIM, max_length=OCCUPANCY_GRID_DIM)
    detected_obstacles: List[List[float]] = Field(default_factory=list)
    messages_received: List[MessagePayload] = Field(default_factory=list)
    battery_level: float = Field(ge=0.0, le=1.0)


class ActionResponse(BaseModel):
    robot_id: str
    linear_velocity: float
    angular_velocity: float
    should_communicate: bool
    value_estimate: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RobotStateResponse(BaseModel):
    robot_id: str
    pose: RobotPose
    status: str
    battery_level: float
    last_updated: float
    collision_count: int


class MissionStartRequest(BaseModel):
    scenario_name: str = "exploration"
    robot_ids: List[str]


class MissionStartResponse(BaseModel):
    mission_id: str
    status: str
    robot_ids: List[str]


class MissionInfoResponse(BaseModel):
    mission_id: str
    status: str
    robot_ids: List[str]
    failed_robot_ids: List[str]


class RobotFailureReport(BaseModel):
    robot_id: str

