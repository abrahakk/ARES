from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging
from typing import List, Optional, Tuple  

from simulation.environments.randomized import apply_domain_randomization, DEFAULT_RANDOMIZATION  
from simulation.scenarios.exploration import ExplorationScenario  
from simulation.scenarios.robot_failure import RobotFailureScenario  

logger = logging.getLogger(__name__)

try:
    from controller import Supervisor
except ImportError: 
    Supervisor = None
    logger.warning("Webots `controller` module not found - SupervisorController will run in stub mode.")


ARES_ROBOT_PROTO_TEMPLATE = (
    'DEF {robot_id} ARESRobot {{ '
    'translation {x} {y} 0.05 rotation 0 0 1 {theta} '
    'name "{robot_id}" controller "robot_controller" '
    'controllerArgs [] }}'
)


class SupervisorController:
    def __init__(self) -> None:
        self._supervisor = Supervisor() if Supervisor is not None else None
        self._timestep_ms = 32
        self._spawned_robot_defs: List[str] = []
        self._root_children = None
        if self._supervisor is not None:
            self._root_children = self._supervisor.getRoot().getField("children")


    def reset_episode(
        self,
        scenario: ExplorationScenario,
        randomize: bool = True,
    ) -> None:

        self._remove_spawned_robots()

        if self._supervisor is not None:
            self._supervisor.simulationResetPhysics()

        spawn_positions = scenario.default_spawn_positions()
        if randomize:
            xy_only = [(x, y) for x, y, _theta in spawn_positions]
            jittered = apply_domain_randomization(xy_only, DEFAULT_RANDOMIZATION)
            spawn_positions = [
                (jx, jy, theta) for (jx, jy), (_x, _y, theta) in zip(jittered, spawn_positions)
            ]

        for robot_id, (x, y, theta) in zip(scenario.robot_ids, spawn_positions):
            self._spawn_robot(robot_id, x, y, theta)

        logger.info("episode reset: spawned %d robots for scenario '%s'", len(scenario.robot_ids), scenario.name)

    def step(self) -> bool:
        if self._supervisor is None:
            return True
        return self._supervisor.step(self._timestep_ms) != -1

    def _spawn_robot(self, robot_id: str, x: float, y: float, theta: float) -> None:
        if self._root_children is None:
            logger.debug("stub mode: would spawn robot %s at (%.2f, %.2f, %.2f)", robot_id, x, y, theta)
            self._spawned_robot_defs.append(robot_id)
            return

        node_string = ARES_ROBOT_PROTO_TEMPLATE.format(robot_id=robot_id, x=x, y=y, theta=theta)
        self._root_children.importMFNodeFromString(-1, node_string)
        self._spawned_robot_defs.append(robot_id)

    def _remove_spawned_robots(self) -> None:
        if self._supervisor is None:
            self._spawned_robot_defs.clear()
            return

        for robot_id in self._spawned_robot_defs:
            node = self._supervisor.getFromDef(robot_id)
            if node is not None:
                node.remove()
        self._spawned_robot_defs.clear()



    def inject_failure(self, robot_id: str) -> None:

        if self._supervisor is None:
            logger.debug("stub mode: would inject failure for robot %s", robot_id)
            return

        node = self._supervisor.getFromDef(robot_id)
        if node is None:
            logger.warning("inject_failure: no such robot DEF=%s", robot_id)
            return
        controller_field = node.getField("controller")
        if controller_field is not None:
            controller_field.setSFString("<none>")
        logger.info("injected failure for robot %s", robot_id)


def main() -> None:
    from simulation.scenarios.exploration import ExplorationScenario

    scenario_name = os.environ.get("ARES_SCENARIO", "exploration")
    if scenario_name == "robot_failure":
        scenario: ExplorationScenario = RobotFailureScenario()
    else:
        scenario = ExplorationScenario()

    controller = SupervisorController()
    controller.reset_episode(scenario)

    current_step = 0
    while controller.step():
        if isinstance(scenario, RobotFailureScenario) and scenario.should_trigger_failure(current_step):
            controller.inject_failure(scenario.failure_robot_id)
        current_step += 1


if __name__ == "__main__":
    main()
