from dataclasses import dataclass
import random
from typing import List, Tuple


@dataclass(frozen=True)
class RandomizationConfig:
    obstacle_position_jitter_m: float = 0.15
    lidar_noise_std_m: float = 0.02
    gps_noise_std_m: float = 0.01
    imu_noise_std_rad: float = 0.01
    wheel_speed_noise_fraction: float = 0.03


DEFAULT_RANDOMIZATION = RandomizationConfig()


def apply_domain_randomization(
    obstacle_positions: List[Tuple[float, float]],
    config: RandomizationConfig = DEFAULT_RANDOMIZATION,
) -> List[Tuple[float, float]]:
    randomized = []
    for x, y in obstacle_positions:
        dx = random.uniform(-config.obstacle_position_jitter_m, config.obstacle_position_jitter_m)
        dy = random.uniform(-config.obstacle_position_jitter_m, config.obstacle_position_jitter_m)
        randomized.append((x + dx, y + dy))
    return randomized
