from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

from rl.agents.single_agent.ppo import SingleAgentPPO
from rl.environments.observation_space import OBSERVATION_DIM
from rl.utils.checkpoints import CheckpointManager
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario
from training.train_single_agent import train_single_agent

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    hyperparameters: Dict[str, Any]
    best_coverage: float


def grid_search(
    search_space: Dict[str, List[Any]],
    reduced_env_steps: int = 20_000,
    base_run_name: str = "hp_search",
) -> List[SearchResult]:
 
    keys = list(search_space.keys())
    combinations = list(itertools.product(*search_space.values()))
    logger.info("grid search: %d combinations across %d hyperparameters", len(combinations), len(keys))

    results = []
    for i, values in enumerate(combinations):
        hyperparameters = dict(zip(keys, values))
        run_name = f"{base_run_name}_grid_{i:03d}"
        coverage = _run_trial(hyperparameters, reduced_env_steps, run_name)
        results.append(SearchResult(hyperparameters, coverage))
        logger.info("trial %d/%d: %s -> coverage=%.3f", i + 1, len(combinations), hyperparameters, coverage)

    return sorted(results, key=lambda r: r.best_coverage, reverse=True)


def random_search(
    search_space: Dict[str, List[Any]],
    num_trials: int,
    reduced_env_steps: int = 20_000,
    base_run_name: str = "hp_search",
    seed: int = 0,
) -> List[SearchResult]:
    """Samples `num_trials` combinations uniformly at random from search_space, with replacement."""
    rng = random.Random(seed)
    keys = list(search_space.keys())

    results = []
    for i in range(num_trials):
        hyperparameters = {key: rng.choice(search_space[key]) for key in keys}
        run_name = f"{base_run_name}_random_{i:03d}"
        coverage = _run_trial(hyperparameters, reduced_env_steps, run_name)
        results.append(SearchResult(hyperparameters, coverage))
        logger.info("trial %d/%d: %s -> coverage=%.3f", i + 1, num_trials, hyperparameters, coverage)

    return sorted(results, key=lambda r: r.best_coverage, reverse=True)


def _run_trial(hyperparameters: Dict[str, Any], reduced_env_steps: int, run_name: str) -> float:
 
    defaults = dict(hidden_dim=256, learning_rate=3e-4, clip_epsilon=0.2, value_loss_coef=0.5,
                     entropy_coef=0.01, max_grad_norm=0.5, num_epochs=4, minibatch_size=64)
    defaults.update(hyperparameters)

    algorithm = SingleAgentPPO(OBSERVATION_DIM, **defaults)
    scenario = ExplorationScenario(environment=STATIC_ENV_CONFIG, robot_ids=["robot_01"])
    checkpoint_manager = CheckpointManager(root_dir=f"models/checkpoints/_hp_search/{run_name}")

    return train_single_agent(
        algorithm, scenario, total_env_steps=reduced_env_steps, run_name=run_name,
        checkpoint_manager=checkpoint_manager, save_interval=reduced_env_steps,  
        log_interval_episodes=9999,  
    )
