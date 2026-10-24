

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class RobotStatus(str, Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    NAVIGATING_TO_GOAL = "navigating_to_goal"
    COMMUNICATING = "communicating"
    RETURNING = "returning"
    FAILED = "failed"
    DISCONNECTED = "disconnected"


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # radians, expected range [-pi, pi]

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.theta)


@dataclass
class RobotState:
    

    robot_id: str
    pose: Pose = field(default_factory=Pose)
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    battery_level: float = 1.0  # 0.0 - 1.0
    status: RobotStatus = RobotStatus.IDLE
    current_goal: Optional[tuple] = None  # (x, y) frontier/goal being pursued
    last_updated: float = field(default_factory=time.time)
    collision_count: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    coverage_contribution: float = 0.0  # sq. meters uniquely explored by this robot

    def touch(self) -> None:
        self.last_updated = time.time()

    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        
        return (time.time() - self.last_updated) > max_age_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "RobotState":
        data = dict(data)
        pose_data = data.get("pose", {})
        pose = Pose(**pose_data) if isinstance(pose_data, dict) else Pose()
        data["pose"] = pose
        if "status" in data and not isinstance(data["status"], RobotStatus):
            data["status"] = RobotStatus(data["status"])
        return cls(**data)
