from __future__ import annotations

import argparse
import logging

from evaluation.compare_baselines import compare_all_metrics
from evaluation.evaluate_policy import PolicyEvaluator
from experiments._common import (
    add_common_args,
    build_uniform_act_fns,
    configure_logging,
    load_actor_act_fn,
    make_independent_act_fn,
    random_placeholder_act_fn,
)
from rl.environments.observation_space import OBSERVATION_DIM
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exploration_comparison"
ROBOT_IDS = ["robot_01", "robot_02"]


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    scenario = ExplorationScenario(environment=STATIC_ENV_CONFIG, robot_ids=ROBOT_IDS)

    if args.checkpoint:
        base_act_fn = load_actor_act_fn(args.checkpoint, OBSERVATION_DIM, device=args.device)
    else:
        logger.warning("No --checkpoint given: running with a random placeholder policy (pipeline validation only).")
        base_act_fn = random_placeholder_act_fn()

    independent_act_fn = make_independent_act_fn(base_act_fn)

    logger.info("Running condition: independent (Experiment 2)")
    independent_evaluator = PolicyEvaluator(scenario, EXPERIMENT_NAME, "independent")
    independent_evaluator.run(build_uniform_act_fns(ROBOT_IDS, independent_act_fn), seeds=args.seeds)

    logger.info("Running condition: cooperative_marl (Experiment 3)")
    cooperative_evaluator = PolicyEvaluator(scenario, EXPERIMENT_NAME, "cooperative_marl")
    cooperative_evaluator.run(build_uniform_act_fns(ROBOT_IDS, base_act_fn), seeds=args.seeds)

    logger.info("Comparing independent vs. cooperative_marl (RQ1):")
    for comparison in compare_all_metrics(EXPERIMENT_NAME, "independent", "cooperative_marl"):
        logger.info("  %s", comparison.summary_line())


if __name__ == "__main__":
    main()
