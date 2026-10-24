from __future__ import annotations

import logging
from typing import List, Tuple

from coordination.task_manager import TaskManager
from rl.coordination.formation import FormationController
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


class MissionPlanner:
    def __init__(self, task_manager: TaskManager, grid_divisions: int = 3) -> None:
        self.task_manager = task_manager
        self.grid_divisions = grid_divisions
        self.formation = FormationController()

    def initial_frontiers(self, scenario: ExplorationScenario) -> List[Point]:

        xmin, ymin, xmax, ymax = scenario.environment.world_bounds_m
        width = xmax - xmin
        height = ymax - ymin

        frontiers: List[Point] = []
        for i in range(self.grid_divisions):
            for j in range(self.grid_divisions):
                x = xmin + width * (i + 0.5) / self.grid_divisions
                y = ymin + height * (j + 0.5) / self.grid_divisions
                frontiers.append((x, y))
        return frontiers

    async def seed_mission(self, scenario: ExplorationScenario) -> List[Point]:

        frontiers = self.initial_frontiers(scenario)
        for index, (x, y) in enumerate(frontiers):
            frontier_id = f"grid_{index:03d}"
            await self.task_manager.register_frontier(frontier_id, (x, y))

        logger.info("seeded mission '%s' with %d initial frontiers", scenario.name, len(frontiers))
        return frontiers

    def spawn_positions(self, scenario: ExplorationScenario) -> List[Tuple[float, float, float]]:

        return scenario.default_spawn_positions()
