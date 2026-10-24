from __future__ import annotations

import argparse
import logging

from evaluation.robustness import run_robot_failure_sweep
from experiments._common import add_common_args, build_uniform_act_fns, configure_logging, load_actor_act_fn, random_placeholder_act_fn
from rl.environments.observation_space import OBSERVATION_DIM

logger = logging.getLogger(__name__)

ROBOT_IDS = ["robot_01", "robot_02", "robot_03"]  


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

    result = run_robot_failure_sweep(act_fns, seeds=args.seeds)

    logger.info(
        "Robot failure result: failed robot=%s at step=%d -> "
        "coverage_with_failure=%.3f coverage_without_failure=%.3f recovery_ratio=%.3f",
        result.failure_robot_id, result.failure_step,
        result.mean_coverage_with_failure, result.mean_coverage_without_failure,
        result.coverage_recovery_ratio,
    )

    if result.coverage_recovery_ratio < 0.7:
        logger.warning(
            "Recovery ratio below 0.7 - the team did not substantially compensate for the failed "
            "robot. Worth inspecting data/trajectories/robot_failure__with_failure.jsonl for the "
            "specific seeds where recovery was weakest before treating this as a settled result."
        )


if __name__ == "__main__":
    main()
