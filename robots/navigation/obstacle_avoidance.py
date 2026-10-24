from __future__ import annotations

from typing import List, Optional, Tuple

DEFAULT_SAFE_DISTANCE_M = 0.25
DEFAULT_CRITICAL_DISTANCE_M = 0.12


class ObstacleAvoidance:
    def __init__(
        self,
        safe_distance_m: float = DEFAULT_SAFE_DISTANCE_M,
        critical_distance_m: float = DEFAULT_CRITICAL_DISTANCE_M,
    ) -> None:
        self.safe_distance_m = safe_distance_m
        self.critical_distance_m = critical_distance_m

    def filter_command(
        self,
        linear: float,
        angular: float,
        obstacle_points: List[Tuple[float, float]],
    ) -> Tuple[float, float, bool]:
        
        nearest = self._nearest_forward_obstacle(obstacle_points)
        if nearest is None:
            return linear, angular, False

        distance, bearing = nearest

        if distance <= self.critical_distance_m and linear > 0:
            return 0.0, angular, True

        if distance <= self.safe_distance_m:
            span = max(self.safe_distance_m - self.critical_distance_m, 1e-6)
            scale = max(0.0, (distance - self.critical_distance_m) / span)
            adjusted_linear = min(linear, linear * scale)
            steering_bias = -1.0 if bearing >= 0 else 1.0
            adjusted_angular = angular + steering_bias * (1.0 - scale)
            return adjusted_linear, adjusted_angular, True

        return linear, angular, False

    @staticmethod
    def _nearest_forward_obstacle(
        obstacle_points: List[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        forward_points = [(x, y) for (x, y) in obstacle_points if x > 0]
        if not forward_points:
            return None

        nearest_x, nearest_y = min(
            forward_points, key=lambda pt: (pt[0] ** 2 + pt[1] ** 2) ** 0.5
        )
        distance = (nearest_x ** 2 + nearest_y ** 2) ** 0.5
        return distance, nearest_y
