from __future__ import annotations

import argparse
import logging

import numpy as np

from evaluation.evaluate_policy import PolicyEvaluator
from experiments._common import add_common_args, build_uniform_act_fns, configure_logging, random_placeholder_act_fn
from rl.environments.observation_space import OBSERVATION_DIM
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exploration_comparison"
CONDITION_NAME = "hierarchical_low_level_only"
ROBOT_IDS = ["robot_01", "robot_02"]


def _load_low_level_only_act_fn(checkpoint_path: str, observation_dim: int, device: str = "cpu"):

    import torch

    from rl.agents.hierarchical.low_level_policy import LowLevelPolicy

    policy = LowLevelPolicy(observation_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["low_level_policy"] if "low_level_policy" in checkpoint else checkpoint)
    policy.eval()

    def act_fn(observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        goal_tensor = torch.zeros((1, 2), dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _log_prob, _value = policy.act(obs_tensor, goal_tensor, deterministic=True)
        return action.squeeze(0).cpu().numpy()

    return act_fn


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    scenario = ExplorationScenario(environment=STATIC_ENV_CONFIG, robot_ids=ROBOT_IDS)

    if args.checkpoint:
        act_fn = _load_low_level_only_act_fn(args.checkpoint, OBSERVATION_DIM, device=args.device)
    else:
        logger.warning("No --checkpoint given: running with a random placeholder policy (pipeline validation only).")
        act_fn = random_placeholder_act_fn()

    logger.warning(
        "This run evaluates the low-level policy only, goal fixed at 'hold' - "
        "see this script's module docstring for the integration limitation."
    )

    evaluator = PolicyEvaluator(scenario, EXPERIMENT_NAME, CONDITION_NAME)
    runs = evaluator.run(build_uniform_act_fns(ROBOT_IDS, act_fn), seeds=args.seeds)

    mean_coverage = sum(r.coverage for r in runs) / len(runs) if runs else float("nan")
    logger.info("hierarchical (low-level only): mean_coverage=%.3f across %d seeds", mean_coverage, len(runs))


if __name__ == "__main__":
    main()
