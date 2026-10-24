from __future__ import annotations

import argparse
import logging

from evaluation.scalability import run_scalability_sweep, summarize_scaling_trend
from experiments._common import add_common_args, configure_logging, load_actor_act_fn, random_placeholder_act_fn
from rl.environments.observation_space import OBSERVATION_DIM

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--robot-counts", type=int, nargs="+", default=[2, 4, 6, 8])
    args = parser.parse_args()

    if args.checkpoint:
        shared_act_fn = load_actor_act_fn(args.checkpoint, OBSERVATION_DIM, device=args.device)
    else:
        logger.warning("No --checkpoint given: running with a random placeholder policy (pipeline validation only).")
        shared_act_fn = random_placeholder_act_fn()


    def act_fn_factory(_robot_id: str):
        return shared_act_fn

    results = run_scalability_sweep(act_fn_factory, robot_counts=args.robot_counts, seeds=args.seeds)

    logger.info("Scalability sweep results (RQ6):")
    for point in results:
        logger.info(
            "  num_robots=%d -> mean_coverage=%.3f mean_completion_steps=%s",
            point.num_robots, point.mean_coverage, point.mean_completion_time_steps,
        )

    logger.info(summarize_scaling_trend(results))


if __name__ == "__main__":
    main()
