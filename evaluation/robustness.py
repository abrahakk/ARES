from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List

from evaluation.evaluate_policy import PolicyEvaluator
from robots.communication.protocol import ProtocolConfig
from simulation.environments.dynamic import DYNAMIC_ENV_CONFIG, communication_sweep_configs
from simulation.scenarios.robot_failure import RobotFailureScenario

logger = logging.getLogger(__name__)


@dataclass
class CommunicationSweepPoint:
    packet_loss_probability: float
    max_bandwidth_bytes_per_step: int
    mean_coverage: float
    mean_collision_rate: float
    mean_duplicated_coverage: float


def run_communication_sweep(
    act_fns: dict,
    experiment_name: str = "communication_constraints",
    seeds: List[int] = (0, 1, 2, 3, 4),
    protocol_configs: List[ProtocolConfig] = None,
) -> List[CommunicationSweepPoint]:

    configs = protocol_configs or communication_sweep_configs()
    results: List[CommunicationSweepPoint] = []

    for protocol_config in configs:
        condition_name = f"loss_{protocol_config.packet_loss_probability:.2f}"


        from simulation.scenarios.exploration import ExplorationScenario
        from dataclasses import replace

        env_config = replace(DYNAMIC_ENV_CONFIG, protocol_config=protocol_config)
        scenario = ExplorationScenario(environment=env_config)

        evaluator = PolicyEvaluator(scenario, experiment_name, condition_name)
        runs = evaluator.run(act_fns, seeds=list(seeds))

        if not runs:
            continue

        mean_coverage = sum(r.coverage for r in runs) / len(runs)
        mean_collision = sum(r.collision_count for r in runs) / len(runs)

        results.append(
            CommunicationSweepPoint(
                packet_loss_probability=protocol_config.packet_loss_probability,
                max_bandwidth_bytes_per_step=protocol_config.max_bandwidth_bytes_per_step,
                mean_coverage=mean_coverage,
                mean_collision_rate=mean_collision,
                mean_duplicated_coverage=0.0,  
            )
        )
        logger.info(
            "comm sweep: loss=%.2f -> mean_coverage=%.3f mean_collisions=%.2f",
            protocol_config.packet_loss_probability, mean_coverage, mean_collision,
        )

    return results


@dataclass
class RobotFailureResult:
    failure_robot_id: str
    failure_step: int
    mean_coverage_with_failure: float
    mean_coverage_without_failure: float
    coverage_recovery_ratio: float  

def run_robot_failure_sweep(
    act_fns: dict,
    experiment_name: str = "robot_failure",
    seeds: List[int] = (0, 1, 2, 3, 4),
) -> RobotFailureResult:
    
    from simulation.scenarios.exploration import ExplorationScenario

    failure_scenario = RobotFailureScenario()
    baseline_scenario = ExplorationScenario(
        environment=failure_scenario.environment,
        robot_ids=failure_scenario.robot_ids,
    )

    failure_evaluator = PolicyEvaluator(failure_scenario, experiment_name, "with_failure")
    baseline_evaluator = PolicyEvaluator(baseline_scenario, experiment_name, "without_failure")

    failure_runs = failure_evaluator.run(act_fns, seeds=list(seeds))
    baseline_runs = baseline_evaluator.run(act_fns, seeds=list(seeds))

    mean_with = sum(r.coverage for r in failure_runs) / len(failure_runs) if failure_runs else 0.0
    mean_without = sum(r.coverage for r in baseline_runs) / len(baseline_runs) if baseline_runs else 0.0
    ratio = mean_with / mean_without if mean_without > 0 else 0.0

    logger.info(
        "robot failure sweep: with_failure=%.3f without_failure=%.3f recovery_ratio=%.3f",
        mean_with, mean_without, ratio,
    )

    return RobotFailureResult(
        failure_robot_id=failure_scenario.failure_robot_id,
        failure_step=failure_scenario.failure_step,
        mean_coverage_with_failure=mean_with,
        mean_coverage_without_failure=mean_without,
        coverage_recovery_ratio=ratio,
    )
