from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

ACTION_DIM = 3 

LINEAR_VELOCITY_INDEX = 0
ANGULAR_VELOCITY_INDEX = 1
COMMUNICATE_INDEX = 2


def build_action_space(
    max_linear_velocity: float = 0.3,
    max_angular_velocity: float = 2.0,
) -> Dict[str, Tuple[float, float]]:
    return {
        "linear_velocity": (-max_linear_velocity, max_linear_velocity),
        "angular_velocity": (-max_angular_velocity, max_angular_velocity),
        "should_communicate": (0.0, 1.0),
    }


def action_vector_to_dict(action: np.ndarray) -> dict:
        return {
        "linear_velocity": float(action[LINEAR_VELOCITY_INDEX]),
        "angular_velocity": float(action[ANGULAR_VELOCITY_INDEX]),
        "should_communicate": bool(action[COMMUNICATE_INDEX] > 0.5),
    }
