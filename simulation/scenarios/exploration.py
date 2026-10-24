from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from simulation.environments.static import EnvironmentConfig, STATIC_ENV_CONFIG
from rl.environments.reward import RewardConfig


@dataclass
class ExplorationScenario:
    name: str = "exploration"
    environment: EnvironmentConfig = field(default_factory=lambda: STATIC_ENV_CONFIG)
    robot_ids: List[str] = field(default_factory=lambda: ["robot_01", "robot_02"])
    spawn_positions: Optional[List[Tuple[float, float, float]]] = None  
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    max_steps: int = 2000
    coverage_success_threshold: float = 0.9

    def build_env_kwargs(self) -> dict:
        
        return {
            "robot_ids": self.robot_ids,
            "world_bounds_m": self.environment.world_bounds_m,
            "reward_config": self.reward_config,
            "max_steps": self.max_steps,
            "coverage_success_threshold": self.coverage_success_threshold,
        }

    def default_spawn_positions(self) -> List[Tuple[float, float, float]]:
        
        if self.spawn_positions is not None:
            return self.spawn_positions

        xmin, ymin, xmax, _ymax = self.environment.world_bounds_m
        n = len(self.robot_ids)
        spacing = (xmax - xmin) / max(n + 1, 1)
        return [(xmin + spacing * (i + 1), ymin + 0.5, 0.0) for i in range(n)]
