from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robots.agents.robot import Robot  
from robots.navigation.path_planning import FrontierPathPlanner  
from rl.environments.observation_space import flatten_observation  


def _run_policy_mode(robot: Robot, model_path: str) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    while robot.step():
        observation = robot.get_observation()
        vector = flatten_observation(observation).reshape(1, -1).astype(np.float32)
        action_out, _value_out = session.run(None, {input_name: vector})
        action = action_out[0]

        robot.apply_action(
            {
                "linear_velocity": float(action[0]),
                "angular_velocity": float(action[1]),
                "should_communicate": bool(action[2] > 0.5),
                "message_payload": {},
            }
        )


def _run_frontier_baseline_mode(robot: Robot) -> None:
    """Drives the robot with the non-learned frontier planner - used for the Experiment 2 baseline runs."""
    planner = FrontierPathPlanner()
    current_target = None

    while robot.step():
        observation = robot.get_observation()
        x, y, theta = observation["pose"]

        frontiers = [(x + 2.0, y), (x, y + 2.0), (x - 2.0, y), (x, y - 2.0)]

        if current_target is None or planner.has_reached((x, y), current_target):
            current_target = planner.choose_frontier((x, y), frontiers)

        if current_target is None:
            robot.apply_action({"linear_velocity": 0.0, "angular_velocity": 0.0, "should_communicate": False})
            continue

        linear, angular = planner.velocity_toward((x, y), theta, current_target)
        robot.apply_action({"linear_velocity": linear, "angular_velocity": angular, "should_communicate": False})


def main() -> None:
    robot_id = os.environ.get("ARES_ROBOT_ID", "robot_01")
    mode = os.environ.get("ARES_CONTROLLER_MODE", "frontier_baseline")
    model_path = os.environ.get("ARES_MODEL_PATH", "models/onnx/ares_policy.onnx")

    robot = Robot(robot_id=robot_id)

    if mode == "policy":
        _run_policy_mode(robot, model_path)
    elif mode == "frontier_baseline":
        _run_frontier_baseline_mode(robot)
    else:
        raise ValueError(f"unknown ARES_CONTROLLER_MODE={mode!r}")


if __name__ == "__main__":
    main()
