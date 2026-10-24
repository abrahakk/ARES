from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from simulation.scenarios.exploration import ExplorationScenario

TARGET_DETECTION_RADIUS_M = 0.5
TARGET_INFORMATION_BONUS = 5.0  


@dataclass
class SearchRescueScenario(ExplorationScenario):
    name: str = "search_rescue"
    target_positions: List[Tuple[float, float]] = field(default_factory=list)
    detection_radius_m: float = TARGET_DETECTION_RADIUS_M
    information_bonus: float = TARGET_INFORMATION_BONUS

    def check_targets_found(
        self, robot_positions: dict, already_found: set
    ) -> set:
        
        found = set(already_found)
        for target_index, (tx, ty) in enumerate(self.target_positions):
            if target_index in found:
                continue
            for (rx, ry) in robot_positions.values():
                if ((rx - tx) ** 2 + (ry - ty) ** 2) ** 0.5 <= self.detection_radius_m:
                    found.add(target_index)
                    break
        return found
