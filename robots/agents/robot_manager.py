
from __future__ import annotations

import logging
from typing import Dict, Iterable, List

from robots.agents.robot import Robot
from robots.agents.robot_state import RobotState, RobotStatus

logger = logging.getLogger(__name__)


class RobotManager:
    
    def __init__(self, robot_ids: Iterable[str]) -> None:
        self._robot_ids: List[str] = list(robot_ids)
        self.robots: Dict[str, Robot] = {}

    def spawn_all(self, **robot_kwargs) -> None:
        for robot_id in self._robot_ids:
            self.robots[robot_id] = Robot(robot_id=robot_id, **robot_kwargs)
        logger.info("spawned %d robots: %s", len(self.robots), self._robot_ids)

    def step(self) -> bool:
        still_running = True
        for robot in self.robots.values():
            still_running = robot.step() and still_running
        return still_running

    def get_observations(self) -> Dict[str, dict]:
        return {
            robot_id: robot.get_observation()
            for robot_id, robot in self.robots.items()
            if robot.state.status != RobotStatus.FAILED
        }

    def apply_actions(self, actions: Dict[str, dict]) -> None:
        for robot_id, action in actions.items():
            robot = self.robots.get(robot_id)
            if robot is None:
                logger.warning("apply_actions: unknown robot_id=%s", robot_id)
                continue
            if robot.state.status == RobotStatus.FAILED:
                continue
            robot.apply_action(action)

  
    def get_states(self) -> Dict[str, RobotState]:
        return {robot_id: robot.state for robot_id, robot in self.robots.items()}

    def active_robot_ids(self) -> List[str]:
        return [
            robot_id
            for robot_id, robot in self.robots.items()
            if robot.state.status != RobotStatus.FAILED
        ]

    
    def fail_robot(self, robot_id: str) -> None:
        robot = self.robots.get(robot_id)
        if robot is None:
            raise KeyError(f"no such robot: {robot_id}")
        robot.mark_failed()

    def deliver_message(self, robot_id: str, message) -> None:
        robot = self.robots.get(robot_id)
        if robot is not None and robot.state.status != RobotStatus.FAILED:
            robot.receive_message(message)
