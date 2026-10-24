from __future__ import annotations

import argparse
import logging

from evaluation.robustness import run_communication_sweep
from experiments._common import add_common_args, build_uniform_act_fns, configure_logging, load_actor_act_fn, random_placeholder_act_fn
from rl.environments.observation_space import OBSERVATION_DIM

logger = logging.getLogger(__name__)

ROBOT_IDS = ["robot_01", "robot_02"]


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    if args.checkpoint:
        act_fn = load_actor_act_fn(args.checkpoint, OBSERVATION_DIM, device=args.device)
    else:
        logger.warning("No --checkpoint given: running with a random placeholder policy (pipeline validation only).")
        act_fn = random_placeholder_act_fn()

    act_fns = build_uniform_act_fns(ROBOT_IDS, act_fn)

    results = run_communication_sweep(act_fns, seeds=args.seeds)

    logger.info("Communication sweep results (RQ3):")
    for point in results:
        logger.info(
            "  loss=%.2f bandwidth=%d -> mean_coverage=%.3f mean_collision_rate=%.2f",
            point.packet_loss_probability, point.max_bandwidth_bytes_per_step,
            point.mean_coverage, point.mean_collision_rate,
        )

    if len(results) >= 2:
        coverage_drop = results[0].mean_coverage - results[-1].mean_coverage
        logger.info(
            "Coverage dropped %.3f from perfect communication to the most degraded condition tested. "
            "Whether this constitutes a 'graceful' or 'sharp' degradation (per RQ3) should be judged "
            "against the full per-point trend above, not this single number alone.",
            coverage_drop,
        )


if __name__ == "__main__":
    main()
