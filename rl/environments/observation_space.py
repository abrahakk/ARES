
from __future__ import annotations

from typing import Dict, List

import numpy as np


OCCUPANCY_GRID_DIM = 25
POSE_DIM = 3            
MESSAGE_SUMMARY_DIM = 8  
BATTERY_DIM = 1

OBSERVATION_DIM = POSE_DIM + OCCUPANCY_GRID_DIM + MESSAGE_SUMMARY_DIM + BATTERY_DIM


def build_observation_space() -> Dict[str, tuple]:
        return {
        "pose": (POSE_DIM, (-np.inf, np.inf)),
        "local_occupancy": (OCCUPANCY_GRID_DIM, (0.0, 1.0)),
        "message_summary": (MESSAGE_SUMMARY_DIM, (-np.inf, np.inf)),
        "battery_level": (BATTERY_DIM, (0.0, 1.0)),
    }


def flatten_observation(observation: dict, message_summary: List[float] | None = None) -> np.ndarray:
    pose = observation.get("pose", (0.0, 0.0, 0.0))
    occupancy = observation.get("local_occupancy", [0.0] * OCCUPANCY_GRID_DIM)
    battery = [observation.get("battery_level", 1.0)]

    if message_summary is None:
        message_summary = [0.0] * MESSAGE_SUMMARY_DIM

    vector = list(pose) + list(occupancy) + list(message_summary) + battery
    if len(vector) != OBSERVATION_DIM:
        raise ValueError(
            f"flattened observation has length {len(vector)}, expected {OBSERVATION_DIM}. "
            "Check that local_occupancy / message_summary sizes match the constants in this module."
        )
    return np.asarray(vector, dtype=np.float32)
