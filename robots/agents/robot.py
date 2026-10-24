
from __future__ import annotations

import logging
from typing import Any, Dict, List

from robots.agents.robot_state import RobotState, RobotStatus, Pose
from robots.communication.communication_manager import CommunicationManager
from robots.communication.message import Message
from robots.navigation.local_controller import LocalController
from robots.navigation.obstacle_avoidance import ObstacleAvoidance
from robots.sensors.camera import Camera
from robots.sensors.gps import GPS
from robots.sensors.imu import IMU
from robots.sensors.lidar import Lidar

logger = logging.getLogger(__name__)

try:
    from controller import Robot as WebotsRobot  
except ImportError:  
    WebotsRobot = None
    logger.warning("Webots `controller` module not found - Robot will run in stub mode.")


DEFAULT_TIMESTEP_MS = 32


class Robot:
    
    def __init__(
        self,
        robot_id: str,
        timestep_ms: int = DEFAULT_TIMESTEP_MS,
        enable_safety_override: bool = True,
    ) -> None:
        self.robot_id = robot_id
        self.timestep_ms = timestep_ms
        self.enable_safety_override = enable_safety_override

        self._webots_robot = WebotsRobot() if WebotsRobot is not None else None

        # Sensors
        self.gps = GPS(self._webots_robot, timestep_ms)
        self.imu = IMU(self._webots_robot, timestep_ms)
        self.lidar = Lidar(self._webots_robot, timestep_ms)
        self.camera = Camera(self._webots_robot, timestep_ms)

        # Navigation
        self.local_controller = LocalController(self._webots_robot)
        self.obstacle_avoidance = ObstacleAvoidance()

        # Communication
        self.communication = CommunicationManager(robot_id=robot_id)

        # State
        self.state = RobotState(robot_id=robot_id)
        self._inbox: List[Message] = []

    
    def step(self) -> bool:
        if self._webots_robot is None:
            return True
        return self._webots_robot.step(self.timestep_ms) != -1

    
    def get_observation(self) -> Dict[str, Any]:
        x, y = self.gps.get_position()
        theta = self.imu.get_yaw()
        self.state.pose = Pose(x=x, y=y, theta=theta)
        self.state.touch()

        occupancy = self.lidar.get_local_occupancy_grid()
        obstacles = self.lidar.get_obstacle_points()
        messages = self._drain_inbox()

        return {
            "robot_id": self.robot_id,
            "pose": self.state.pose.as_tuple(),
            "local_occupancy": occupancy,
            "detected_obstacles": obstacles,
            "messages_received": [m.to_dict() for m in messages],
            "battery_level": self.state.battery_level,
        }

    def _drain_inbox(self) -> List[Message]:
        messages, self._inbox = self._inbox, []
        self.state.messages_received += len(messages)
        return messages

    def receive_message(self, message: Message) -> None:
         self._inbox.append(message)

   
    def apply_action(self, action: Dict[str, Any]) -> None:
        
        linear = float(action.get("linear_velocity", 0.0))
        angular = float(action.get("angular_velocity", 0.0))

        if self.enable_safety_override:
            linear, angular, overridden = self.obstacle_avoidance.filter_command(
                linear, angular, self.lidar.get_obstacle_points()
            )
            if overridden:
                logger.debug("robot=%s safety override engaged", self.robot_id)

        self.local_controller.set_velocity(linear, angular)
        self.state.linear_velocity = linear
        self.state.angular_velocity = angular
        self.state.status = RobotStatus.EXPLORING

        if action.get("should_communicate"):
            self._broadcast(action.get("message_payload") or {})

    def _broadcast(self, payload: dict) -> None:
        message = self.communication.build_message(payload)
        self.communication.send(message)
        self.state.messages_sent += 1

    def mark_failed(self) -> None:
        self.state.status = RobotStatus.FAILED
        self.local_controller.set_velocity(0.0, 0.0)
        logger.info("robot=%s marked as failed", self.robot_id)

