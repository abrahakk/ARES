from __future__ import annotations

import math
from typing import Dict, List, Tuple

Point = Tuple[float, float]


class FormationController:
    def __init__(self, min_spacing_m: float = 0.5) -> None:
        self.min_spacing_m = min_spacing_m

    def compute_rendezvous_point(self, robot_positions: Dict[str, Point]) -> Point:

        if not robot_positions:
            return (0.0, 0.0)
        xs = [p[0] for p in robot_positions.values()]
        ys = [p[1] for p in robot_positions.values()]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def assign_formation_targets(
        self, robot_ids: List[str], rendezvous_point: Point
    ) -> Dict[str, Point]:
        n = len(robot_ids)
        if n == 0:
            return {}
        if n == 1:
            return {robot_ids[0]: rendezvous_point}

        circumference_needed = n * self.min_spacing_m
        radius = max(circumference_needed / (2 * math.pi), self.min_spacing_m)

        targets: Dict[str, Point] = {}
        for i, robot_id in enumerate(robot_ids):
            angle = 2 * math.pi * i / n
            x = rendezvous_point[0] + radius * math.cos(angle)
            y = rendezvous_point[1] + radius * math.sin(angle)
            targets[robot_id] = (x, y)
        return targets
