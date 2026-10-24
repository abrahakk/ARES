from __future__ import annotations

from typing import List

from data.schemas import TrajectoryRecord


def coverage_fraction(record: TrajectoryRecord) -> float:

    if not record.steps:
        return 0.0
    return record.steps[-1].coverage_fraction


def mission_completion(record: TrajectoryRecord, success_threshold: float = 0.9) -> bool:

    return coverage_fraction(record) >= success_threshold


def collision_rate(record: TrajectoryRecord) -> float:

    if not record.steps:
        return 0.0
    robot_ids = {s.robot_id for s in record.steps}
    total_collisions = sum(1 for s in record.steps if s.collided)
    return total_collisions / max(len(robot_ids), 1)


def path_efficiency(record: TrajectoryRecord) -> float:
    if not record.steps:
        return 0.0

    total_distance = 0.0
    by_robot: dict = {}
    for step in record.steps:
        prev = by_robot.get(step.robot_id)
        if prev is not None:
            dx = step.pose[0] - prev[0]
            dy = step.pose[1] - prev[1]
            total_distance += (dx ** 2 + dy ** 2) ** 0.5
        by_robot[step.robot_id] = step.pose

    if total_distance <= 0:
        return 0.0

    final_coverage = coverage_fraction(record)
    theoretical_minimum = (final_coverage ** 0.5) * 4.0  
    return min(1.0, theoretical_minimum / total_distance)


def energy_consumption(record: TrajectoryRecord) -> float:

    total_distance = 0.0
    by_robot: dict = {}
    for step in record.steps:
        prev = by_robot.get(step.robot_id)
        if prev is not None:
            dx = step.pose[0] - prev[0]
            dy = step.pose[1] - prev[1]
            total_distance += (dx ** 2 + dy ** 2) ** 0.5
        by_robot[step.robot_id] = step.pose
    return total_distance


def communication_overhead(record: TrajectoryRecord) -> int:

    return sum(s.messages_sent for s in record.steps)


def communication_coverage_correlation(records: List[TrajectoryRecord]) -> float:

    if len(records) < 2:
        return 0.0

    comm = [communication_overhead(r) for r in records]
    cov = [coverage_fraction(r) for r in records]

    n = len(records)
    mean_comm = sum(comm) / n
    mean_cov = sum(cov) / n

    covariance = sum((c - mean_comm) * (v - mean_cov) for c, v in zip(comm, cov))
    comm_var = sum((c - mean_comm) ** 2 for c in comm)
    cov_var = sum((v - mean_cov) ** 2 for v in cov)

    denominator = (comm_var * cov_var) ** 0.5
    if denominator == 0:
        return 0.0
    return covariance / denominator


def mean_episode_reward(record: TrajectoryRecord) -> float:

    if not record.steps:
        return 0.0
    return sum(s.reward for s in record.steps) / len(record.steps)


def reward_breakdown_totals(record: TrajectoryRecord) -> dict:

    totals: dict = {}
    for step in record.steps:
        for key, value in step.reward_breakdown.items():
            totals[key] = totals.get(key, 0.0) + value
    return totals
