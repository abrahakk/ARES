from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class GPS:
    def __init__(self, webots_robot, timestep_ms: int, device_name: str = "gps") -> None:
        self._device = None
        if webots_robot is not None:
            self._device = webots_robot.getDevice(device_name)
            self._device.enable(timestep_ms)

    def get_position(self) -> Tuple[float, float]:
        """Returns (x, y) in world coordinates, meters."""
        if self._device is None:
            return (0.0, 0.0)
        x, y, _z = self._device.getValues()
        return (x, y)
