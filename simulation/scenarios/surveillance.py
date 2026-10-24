from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from simulation.scenarios.exploration import ExplorationScenario

DEFAULT_REVISIT_INTERVAL_STEPS = 300


@dataclass
class SurveillanceScenario(ExplorationScenario):
    name: str = "surveillance"
    revisit_interval_steps: int = DEFAULT_REVISIT_INTERVAL_STEPS
    staleness_bonus_scale: float = 0.5

    def compute_staleness_bonus(
        self, cell: Tuple[int, int], last_observed_step: Dict[Tuple[int, int], int], current_step: int
    ) -> float:
        last_seen = last_observed_step.get(cell)
        if last_seen is None:
            return 0.0

        steps_since = current_step - last_seen
        overdue_fraction = max(0.0, (steps_since - self.revisit_interval_steps) / self.revisit_interval_steps)
        return min(overdue_fraction, 1.0) * self.staleness_bonus_scale
