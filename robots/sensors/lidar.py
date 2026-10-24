from __future__ import annotations

import logging
import math
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_GRID_SIZE = 5            
DEFAULT_GRID_RESOLUTION_M = 0.2  
DEFAULT_MAX_RANGE_M = 3.0


class Lidar:
    def __init__(
        self,
        webots_robot,
        timestep_ms: int,
        device_name: str = "lidar",
        grid_size: int = DEFAULT_GRID_SIZE,
        grid_resolution_m: float = DEFAULT_GRID_RESOLUTION_M,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ) -> None:
        self.grid_size = grid_size
        self.grid_resolution_m = grid_resolution_m
        self.max_range_m = max_range_m

        self._device = None
        if webots_robot is not None:
            self._device = webots_robot.getDevice(device_name)
            self._device.enable(timestep_ms)

    def _raw_ranges(self) -> List[float]:
        if self._device is None:
            return []
        return list(self._device.getRangeImage())

    def get_obstacle_points(self) -> List[Tuple[float, float]]:
 
        ranges = self._raw_ranges()
        if not ranges:
            return []

        n = len(ranges)
        fov = 2 * math.pi  
        points: List[Tuple[float, float]] = []
        for i, r in enumerate(ranges):
            if r >= self.max_range_m or math.isinf(r) or math.isnan(r):
                continue
            angle = -fov / 2 + fov * (i / max(n - 1, 1))
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            points.append((x, y))
        return points

    def get_local_occupancy_grid(self) -> List[float]:
        
        half = self.grid_size // 2
        grid = [[0.0] * self.grid_size for _ in range(self.grid_size)]

        for x, y in self.get_obstacle_points():
            gx = half + int(round(x / self.grid_resolution_m))
            gy = half + int(round(y / self.grid_resolution_m))
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                grid[gy][gx] = 1.0

        return [cell for row in grid for cell in row]
