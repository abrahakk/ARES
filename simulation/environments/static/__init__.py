from dataclasses import dataclass, field
from typing import Tuple

from robots.communication.protocol import ProtocolConfig


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    world_file: str
    world_bounds_m: Tuple[float, float, float, float]  
    protocol_config: ProtocolConfig = field(default_factory=ProtocolConfig)
    description: str = ""


STATIC_ENV_CONFIG = EnvironmentConfig(
    name="static",
    world_file="simulation/webots/worlds/maze.wbt",
    world_bounds_m=(-4.0, -4.0, 4.0, 4.0),
    protocol_config=ProtocolConfig(
        max_bandwidth_bytes_per_step=2048,
        packet_loss_probability=0.0,
        max_range_meters=15.0,
    ),
    description=(
        "A room subdivided by fixed walls and furniture-scale obstacles "
        "into several connected sub-regions. Robots must decide which "
        "sub-region to explore and coordinate to avoid sending two "
        "robots to the same region. Perfect communication channel - "
        "the confound this environment deliberately does NOT introduce "
        "is communication degradation, which is reserved for the "
        "dynamic environment."
    ),
)
