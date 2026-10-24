from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_WHEEL_RADIUS_M = 0.033
DEFAULT_AXLE_LENGTH_M = 0.16
DEFAULT_MAX_WHEEL_SPEED_RAD_S = 18.84  


class LocalController:
    

    def __init__(
        self,
        webots_robot,
        wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
        axle_length_m: float = DEFAULT_AXLE_LENGTH_M,
        max_wheel_speed_rad_s: float = DEFAULT_MAX_WHEEL_SPEED_RAD_S,
        left_motor_name: str = "left wheel motor",
        right_motor_name: str = "right wheel motor",
    ) -> None:
        self.wheel_radius_m = wheel_radius_m
        self.axle_length_m = axle_length_m
        self.max_wheel_speed_rad_s = max_wheel_speed_rad_s

        self._left_motor = None
        self._right_motor = None
        if webots_robot is not None:
            self._left_motor = webots_robot.getDevice(left_motor_name)
            self._right_motor = webots_robot.getDevice(right_motor_name)
            for motor in (self._left_motor, self._right_motor):
                motor.setPosition(float("inf"))  
                motor.setVelocity(0.0)

    def set_velocity(self, linear: float, angular: float) -> None:
        
        w_left = (linear - angular * self.axle_length_m / 2.0) / self.wheel_radius_m
        w_right = (linear + angular * self.axle_length_m / 2.0) / self.wheel_radius_m

        w_left = max(-self.max_wheel_speed_rad_s, min(self.max_wheel_speed_rad_s, w_left))
        w_right = max(-self.max_wheel_speed_rad_s, min(self.max_wheel_speed_rad_s, w_right))

        if self._left_motor is not None and self._right_motor is not None:
            self._left_motor.setVelocity(w_left)
            self._right_motor.setVelocity(w_right)
        else:
            logger.debug("stub mode: would set wheel speeds L=%.2f R=%.2f", w_left, w_right)

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)
