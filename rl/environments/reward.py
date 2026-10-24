from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml


@dataclass
class RewardConfig:
    w_coverage: float = 1.0
    w_information: float = 0.5
    w_cooperation: float = 0.3
    w_collision: float = 2.0
    w_energy: float = 0.05
    w_communication: float = 0.02

    @classmethod
    def from_yaml(cls, path: str) -> "RewardConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def ablated(self, remove_term: str) -> "RewardConfig":

        field_name = f"w_{remove_term}"
        if field_name not in self.__dataclass_fields__:
            raise ValueError(f"unknown reward term '{remove_term}'")
        kwargs = self.__dict__.copy()
        kwargs[field_name] = 0.0
        return RewardConfig(**kwargs)


def compute_reward(
    newly_covered_area_m2: float,
    joint_entropy_reduction: float,
    cooperation_event: bool,
    collided: bool,
    distance_traveled_m: float,
    message_sent: bool,
    config: RewardConfig,
) -> Tuple[float, Dict[str, float]]:

    r_coverage = newly_covered_area_m2
    r_information = joint_entropy_reduction
    r_cooperation = 1.0 if cooperation_event else 0.0
    r_collision = 1.0 if collided else 0.0
    r_energy = distance_traveled_m
    r_communication = 1.0 if message_sent else 0.0

    total = (
        config.w_coverage * r_coverage
        + config.w_information * r_information
        + config.w_cooperation * r_cooperation
        - config.w_collision * r_collision
        - config.w_energy * r_energy
        - config.w_communication * r_communication
    )

    breakdown = {
        "r_coverage": r_coverage,
        "r_information": r_information,
        "r_cooperation": r_cooperation,
        "r_collision": r_collision,
        "r_energy": r_energy,
        "r_communication": r_communication,
        "total": total,
    }
    return total, breakdown


def aggregate_episode_breakdown(step_breakdowns: List[Dict[str, float]]) -> Dict[str, float]:

    if not step_breakdowns:
        return {}
    keys = step_breakdowns[0].keys()
    return {k: sum(step[k] for step in step_breakdowns) for k in keys}
