from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from simulation.scenarios.exploration import ExplorationScenario
from simulation.environments.dynamic import DYNAMIC_ENV_CONFIG


@dataclass
class RobotFailureScenario(ExplorationScenario):
    name: str = "robot_failure"
    environment: "EnvironmentConfig" = field(default_factory=lambda: DYNAMIC_ENV_CONFIG)
    robot_ids: List[str] = field(default_factory=lambda: ["robot_01", "robot_02", "robot_03"])

    failure_step: int = 500       
    failure_robot_id: str = "robot_02"  

    def __post_init__(self) -> None:
        if self.failure_robot_id not in self.robot_ids:
            raise ValueError(
                f"failure_robot_id={self.failure_robot_id!r} is not in robot_ids={self.robot_ids!r}"
            )

    def should_trigger_failure(self, current_step: int) -> bool:
        return current_step == self.failure_step
