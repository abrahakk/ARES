from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorkerRolloutResult:
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    final_coverage: float


def _collect_one_rollout(args: Tuple) -> WorkerRolloutResult:
  
    env_factory, act_fn_factory, robot_id, max_steps, seed = args

    np.random.seed(seed)
    env = env_factory()
    act_fn = act_fn_factory()

    observations_list, actions_list, rewards_list, dones_list = [], [], [], []

    observations = env.reset()
    steps = 0
    while steps < max_steps:
        obs = observations.get(robot_id)
        if obs is None:
            break
        action = act_fn(obs)
        observations, rewards, dones, _infos = env.step({robot_id: action})

        observations_list.append(obs)
        actions_list.append(action)
        rewards_list.append(rewards.get(robot_id, 0.0))
        dones_list.append(dones.get(robot_id, False))
        steps += 1

        if dones.get(robot_id, False):
            break

    final_coverage = env.coverage_map.coverage_fraction()
    return WorkerRolloutResult(observations_list, actions_list, rewards_list, dones_list, final_coverage)


def collect_parallel_rollouts(
    env_factory: Callable,
    act_fn_factory: Callable,
    robot_id: str,
    num_workers: int,
    max_steps_per_rollout: int,
    base_seed: int = 0,
) -> List[WorkerRolloutResult]:

    tasks = [
        (env_factory, act_fn_factory, robot_id, max_steps_per_rollout, base_seed + i)
        for i in range(num_workers)
    ]

    logger.info("collecting %d parallel rollouts (max_steps=%d each)", num_workers, max_steps_per_rollout)
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_collect_one_rollout, tasks)

    mean_coverage = sum(r.final_coverage for r in results) / len(results) if results else 0.0
    logger.info("parallel rollout collection complete: mean_coverage=%.3f across %d workers", mean_coverage, num_workers)
    return results
