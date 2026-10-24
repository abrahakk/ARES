from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

Point = Tuple[float, float]


@dataclass
class FrontierPathPlanner:


    frontier_reach_threshold_m: float = 0.3
    _claimed_frontiers: Set[Point] = field(default_factory=set, init=False)

    def choose_frontier(
        self,
        current_pos: Point,
        known_frontiers: List[Point],
    ) -> Optional[Point]:
        available = [f for f in known_frontiers if f not in self._claimed_frontiers]
        if not available:
            return None
        return min(available, key=lambda f: self._distance(current_pos, f))

    def claim_frontier(self, frontier: Point) -> None:
        self._claimed_frontiers.add(frontier)

    def release_frontier(self, frontier: Point) -> None:
        self._claimed_frontiers.discard(frontier)

    def has_reached(self, current_pos: Point, target: Point) -> bool:
        return self._distance(current_pos, target) <= self.frontier_reach_threshold_m

    def velocity_toward(
        self,
        current_pos: Point,
        current_theta: float,
        target: Point,
        max_linear: float = 0.2,
        max_angular: float = 1.5,
    ) -> Tuple[float, float]:

        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        target_theta = math.atan2(dy, dx)
        heading_error = self._normalize_angle(target_theta - current_theta)

        angular = max(-max_angular, min(max_angular, 2.0 * heading_error))

        linear = max_linear * max(0.0, 1.0 - abs(heading_error) / math.pi)
        return linear, angular

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

