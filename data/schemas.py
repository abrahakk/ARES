from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeStep:


    step: int
    robot_id: str
    pose: List[float]  
    action: List[float]
    reward: float
    reward_breakdown: Dict[str, float]
    coverage_fraction: float
    messages_sent: int
    messages_received: int
    collided: bool


@dataclass
class TrajectoryRecord:
    

    episode_id: str
    experiment_name: str
    condition: str  
    seed: int
    environment_name: str
    steps: List[EpisodeStep] = field(default_factory=list)
    outcome: str = "unkno
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))
EvaluationRun:
    @classmethod
    def from_json_line(cls, line: str) -> "TrajectoryRecord":
        data = json.loads(line)
        data["steps"] = [EpisodeStep(**s) for s in data.get("steps", [])]
        return cls(**data)


@dataclass
class EvaluationRun:

    run_name: str
    robot_id: str
    step: int
    observation: List[float]
    action: List[float]
    reward: float
    next_observation: List[float]
    done: bool

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json_line(cls, line: str) -> "ExperienceRecord":
        return cls(**json.loads(line))


@dataclass
class EvaluationRun:
    
    experiment_name: str
    condition: str
    seed: int
    environment_name: str
    coverage: float
    mission_completed: bool
    completion_time_steps: Optional[int]
    collision_count: int
    path_efficiency: float
    energy_consumption: float
    communication_messages: int
    mean_episode_reward: float
    reward_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json_line(cls, line: str) -> "EvaluationRun":
        return cls(**json.loads(line))
