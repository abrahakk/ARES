from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Camera:
    def __init__(self, webots_robot, timestep_ms: int, device_name: str = "camera") -> None:
        self._device = None
        self._timestep_ms = timestep_ms
        if webots_robot is not None:
            self._device = webots_robot.getDevice(device_name)
            self._device.enable(timestep_ms)

    @property
    def resolution(self) -> Optional[Tuple[int, int]]:
        if self._device is None:
            return None
        return (self._device.getWidth(), self._device.getHeight())

    def get_frame(self):

        if self._device is None:
            return None
        return self._device.getImage()

    def save_frame(self, path: str) -> bool:

        if self._device is None:
            logger.debug("stub mode: would save camera frame to %s", path)
            return False
        self._device.saveImage(path, quality=90)
        return True
