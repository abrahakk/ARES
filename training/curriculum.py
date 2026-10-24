from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

from simulation.environments.dynamic import DYNAMIC_ENV_CONFIG
from simulation.environments.static import EnvironmentConfig, STATIC_ENV_CONFIG

logger = logging.getLogger(__name__)

_SIMPLE_ENV_CONFIG = EnvironmentConfig(
    name="simple",
    world_file="simulation/webots/worlds/simple.wbt",
    world_bounds_m=(-4.0, -4.0, 4.0, 4.0),
    description="Open room, no obstacles - infrastructure check only, per proposal Section 8.1.",
)

_TRAINABLE_STAGES = [_SIMPLE_ENV_CONFIG, STATIC_ENV_CONFIG, DYNAMIC_ENV_CONFIG]


@dataclass
class CurriculumManager:

    advancement_threshold: float = 0.7
    episodes_to_average: int = 20

    _stage_index: int = field(default=0, init=False)
    _recent_coverage: List[float] = field(default_factory=list, init=False)

    @property
    def current_stage(self) -> EnvironmentConfig:
        return _TRAINABLE_STAGES[self._stage_index]

    @property
    def is_final_stage(self) -> bool:
        return self._stage_index == len(_TRAINABLE_STAGES) - 1

    def report_episode_coverage(self, coverage_fraction: float) -> bool:
       
        self._recent_coverage.append(coverage_fraction)
        if len(self._recent_coverage) > self.episodes_to_average:
            self._recent_coverage.pop(0)

        if self.is_final_stage or len(self._recent_coverage) < self.episodes_to_average:
            return False

        mean_recent = sum(self._recent_coverage) / len(self._recent_coverage)
        if mean_recent >= self.advancement_threshold:
            self._stage_index += 1
            self._recent_coverage.clear()
            logger.info(
                "curriculum advanced to stage '%s' (mean coverage over last %d episodes: %.3f >= threshold %.3f)",
                self.current_stage.name, self.episodes_to_average, mean_recent, self.advancement_threshold,
            )
            return True
        return False

    @classmethod
    def from_config(cls, path: str = "config/simulation.yaml") -> "CurriculumManager":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        webots_cfg = data.get("webots", {})
        return cls(advancement_threshold=webots_cfg.get("coverage_success_threshold", 0.9) * 0.75)
