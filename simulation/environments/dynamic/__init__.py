from robots.communication.protocol import ProtocolConfig
from simulation.environments.static import EnvironmentConfig

DYNAMIC_ENV_CONFIG = EnvironmentConfig(
    name="dynamic",
    world_file="simulation/webots/worlds/dynamic.wbt",
    world_bounds_m=(-5.0, -5.0, 5.0, 5.0),
    protocol_config=ProtocolConfig(
        max_bandwidth_bytes_per_step=1024,
        packet_loss_probability=0.05,
        max_range_meters=10.0,
    ),
    description=(
        "The static-obstacle layout with moving obstacles, configurable "
        "sensor noise, and a lossy, bandwidth-limited communication "
        "channel. Used for the communication-constraints experiment "
        "(sweep packet_loss_probability and max_bandwidth_bytes_per_step "
        "away from these defaults) and the robot-failure experiment."
    ),
)


def communication_sweep_configs() -> list:
    loss_levels = [0.0, 0.05, 0.15, 0.30, 0.50, 0.75]
    return [
        ProtocolConfig(
            max_bandwidth_bytes_per_step=DYNAMIC_ENV_CONFIG.protocol_config.max_bandwidth_bytes_per_step,
            packet_loss_probability=loss,
            max_range_meters=DYNAMIC_ENV_CONFIG.protocol_config.max_range_meters,
        )
        for loss in loss_levels
    ]
