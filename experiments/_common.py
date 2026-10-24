from __future__ import annotations

import argparse
import logging
from typing import Callable, Dict, List

import numpy as np

from rl.environments.action_space import build_action_space

logger = logging.getLogger(__name__)

ActFunction = Callable[[np.ndarray], np.ndarray]

_BOUNDS = build_action_space()
_LINEAR_BOUND = _BOUNDS["linear_velocity"][1]
_ANGULAR_BOUND = _BOUNDS["angular_velocity"][1]


def random_placeholder_act_fn(seed: int = 0) -> ActFunction:
    rng = np.random.default_rng(seed)

    def act_fn(observation: np.ndarray) -> np.ndarray:
        linear = rng.uniform(-_LINEAR_BOUND, _LINEAR_BOUND)
        angular = rng.uniform(-_ANGULAR_BOUND, _ANGULAR_BOUND)
        should_communicate = float(rng.random() < 0.1)
        return np.array([linear, angular, should_communicate], dtype=np.float32)

    return act_fn


def load_actor_act_fn(checkpoint_path: str, observation_dim: int, device: str = "cpu") -> ActFunction:
    
    import torch  

    from rl.networks.actor import ActorNetwork

    actor = ActorNetwork(observation_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"] if "actor" in checkpoint else checkpoint)
    actor.eval()

    def act_fn(observation: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _value = actor.export_forward(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    return act_fn


def build_uniform_act_fns(robot_ids: List[str], act_fn: ActFunction) -> Dict[str, ActFunction]:
    
    return {robot_id: act_fn for robot_id in robot_ids}


def make_independent_act_fn(base_act_fn: ActFunction) -> ActFunction:

    def act_fn(observation: np.ndarray) -> np.ndarray:
        action = base_act_fn(observation).copy()
        action[2] = 0.0
        return action

    return act_fn


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a trained checkpoint (rl.utils.checkpoints.CheckpointManager format). "
             "If omitted, runs with random_placeholder_act_fn() for pipeline validation only.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Seeds to evaluate. Defaults to 5, matching the proposal's minimum seed count (Section 12.5).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
