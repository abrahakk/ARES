from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

CELL_SIZE_M = 0.2


class CoverageMap:
    def __init__(self, world_bounds_m: Tuple[float, float, float, float], cell_size_m: float = CELL_SIZE_M) -> None:
        xmin, ymin, xmax, ymax = world_bounds_m
        self.xmin, self.ymin = xmin, ymin
        self.cell_size_m = cell_size_m
        self.width = max(1, int(math.ceil((xmax - xmin) / cell_size_m)))
        self.height = max(1, int(math.ceil((ymax - ymin) / cell_size_m)))
        self.total_cells = self.width * self.height

        self._observed = np.zeros((self.height, self.width), dtype=bool)
        self._per_robot_observed: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._observed[:] = False
        self._per_robot_observed = {}

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int((x - self.xmin) / self.cell_size_m)
        cy = int((y - self.ymin) / self.cell_size_m)
        cx = min(max(cx, 0), self.width - 1)
        cy = min(max(cy, 0), self.height - 1)
        return cx, cy

    def update_from_robot(
        self,
        robot_id: str,
        pose: Tuple[float, float, float],
        obstacle_points_local: List[Tuple[float, float]],
    ) -> Tuple[float, float]:

        if robot_id not in self._per_robot_observed:
            self._per_robot_observed[robot_id] = np.zeros((self.height, self.width), dtype=bool)

        x0, y0, theta = pose
        cells_seen_this_step = set()


        cells_seen_this_step.add(self._world_to_cell(x0, y0))


        sensing_radius_m = 1.0
        steps = 8
        for i in range(steps):
            angle = theta + (2 * math.pi * i / steps)
            wx = x0 + sensing_radius_m * math.cos(angle)
            wy = y0 + sensing_radius_m * math.sin(angle)
            cells_seen_this_step.add(self._world_to_cell(wx, wy))

        newly_covered_team = 0
        newly_covered_self = 0
        for (cx, cy) in cells_seen_this_step:
            if not self._per_robot_observed[robot_id][cy, cx]:
                newly_covered_self += 1
                self._per_robot_observed[robot_id][cy, cx] = True
            if not self._observed[cy, cx]:
                newly_covered_team += 1
                self._observed[cy, cx] = True

        cell_area = self.cell_size_m ** 2
        newly_covered_area_m2 = newly_covered_self * cell_area

        joint_entropy_reduction = newly_covered_team * cell_area

        return newly_covered_area_m2, joint_entropy_reduction

    def coverage_fraction(self) -> float:
        return float(self._observed.sum()) / float(self.total_cells)

    def per_robot_coverage_fraction(self, robot_id: str) -> float:
        if robot_id not in self._per_robot_observed:
            return 0.0
        return float(self._per_robot_observed[robot_id].sum()) / float(self.total_cells)

    def duplicated_coverage_fraction(self) -> float:
        if not self._per_robot_observed:
            return 0.0
        stacked = np.stack(list(self._per_robot_observed.values()), axis=0)
        observed_by_count = stacked.sum(axis=0)
        duplicated = (observed_by_count > 1).sum()
        return float(duplicated) / float(self.total_cells)
