from __future__ import annotations

import argparse
import logging

from evaluation.evaluate_policy import PolicyEvaluator
from rl.environments.observation_space import OBSERVATION_DIM
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)


def _build_act_fn(checkpoint_path: str, device: str):
    import numpy as np
    import torch

    from rl.networks.actor import ActorNetwork

    actor = ActorNetwork(OBSERVATION_DIM).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"] if "actor" in checkpoint else checkpoint)
    actor.eval()

    def act_fn(observation: "np.ndarray") -> "np.ndarray":
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _value = actor.export_forward(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    return act_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name to write results under (data/evaluation/<experiment>.jsonl).")
    parser.add_argument("--condition", type=str, default="default", help="Condition label within the experiment.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    robot_ids = [f"robot_{i+1:02d}" for i in range(args.num_robots)]
    scenario = ExplorationScenario(environment=STATIC_ENV_CONFIG, robot_ids=robot_ids)

    if args.checkpoint:
        act_fn = _build_act_fn(args.checkpoint, args.device)
    else:
        logger.warning("No --checkpoint given - using a random placeholder policy. See experiments/_common.py's "
                        "docstring: results from this run are pipeline validation only, not a real finding.")
        from experiments._common import random_placeholder_act_fn
        act_fn = random_placeholder_act_fn()

    act_fns = {robot_id: act_fn for robot_id in robot_ids}

    evaluator = PolicyEvaluator(scenario, args.experiment, args.condition)
    runs = evaluator.run(act_fns, seeds=args.seeds)

    if not runs:
        logger.warning("no evaluation runs produced")
        return

    mean_coverage = sum(r.coverage for r in runs) / len(runs)
    completed = sum(1 for r in runs if r.mission_completed)
    logger.info(
        "experiment=%s condition=%s: %d/%d seeds completed, mean_coverage=%.3f",
        args.experiment, args.condition, completed, len(runs), mean_coverage,
    )


if __name__ == "__main__":
    main()
