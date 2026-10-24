from __future__ import annotations

import argparse
import logging

from evaluation.evaluate_policy import PolicyEvaluator
from experiments._common import (
    add_common_args,
    build_uniform_act_fns,
    configure_logging,
    load_actor_act_fn,
    random_placeholder_act_fn,
)
from rl.environments.observation_space import OBSERVATION_DIM
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exploration_comparison"
CONDITION_NAME = "single_robot"


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    scenario = ExplorationScenario(
        environment=STATIC_ENV_CONFIG,
        robot_ids=["robot_01"],
    )

    if args.checkpoint:
        act_fn = load_actor_act_fn(args.checkpoint, OBSERVATION_DIM, device=args.device)
    else:
        logger.warning("No --checkpoint given: running with a random placeholder policy (pipeline validation only).")
        act_fn = random_placeholder_act_fn()

    act_fns = build_uniform_act_fns(scenario.robot_ids, act_fn)

    evaluator = PolicyEvaluator(scenario, EXPERIMENT_NAME, CONDITION_NAME)
    runs = evaluator.run(act_fns, seeds=args.seeds)

    mean_coverage = sum(r.coverage for r in runs) / len(runs) if runs else float("nan")
    completed = sum(1 for r in runs if r.mission_completed)
    logger.info(
        "single_robot baseline: %d/%d seeds completed, mean_coverage=%.3f",
        completed, len(runs), mean_coverage,
    )


if __name__ == "__main__":
    main()
