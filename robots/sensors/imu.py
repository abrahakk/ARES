from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class IMU:
    def __init__(self, webots_robot, timestep_ms: int, device_name: str = "inertial unit") -> None:
        self._device = None
        if webots_robot is not None:
            self._device = webots_robot.getDevice(device_name)
            self._device.enable(timestep_ms)

    def get_yaw(self) -> float:
        """Returns heading in radians, [-pi, pi]."""
        if self._device is None:
            return 0.0
        _roll, _pitch, yaw = self._device.getRollPitchYaw()
        return yaw
