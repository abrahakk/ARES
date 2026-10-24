from __future__ import annotations

import argparse
import logging

import numpy as np
import yaml

from rl.environments.ares_env import AresEnv
from rl.environments.observation_space import OBSERVATION_DIM
from rl.environments.reward import RewardConfig
from rl.utils.checkpoints import CheckpointManager
from simulation.environments.static import STATIC_ENV_CONFIG
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)

ALGORITHM_CHOICES = [
    "single_agent_ppo", "single_agent_dqn", "single_agent_sac",
    "mappo", "maddpg", "qmix", "hierarchical",
]


def build_algorithm(name: str, robot_ids, training_cfg: dict, device: str):

    if name == "single_agent_ppo":
        from rl.agents.single_agent.ppo import SingleAgentPPO
        cfg = training_cfg["single_agent_ppo"]
        return SingleAgentPPO(OBSERVATION_DIM, hidden_dim=cfg["hidden_dim"], learning_rate=cfg["learning_rate"],
                               clip_epsilon=cfg["clip_epsilon"], value_loss_coef=cfg["value_loss_coef"],
                               entropy_coef=cfg["entropy_coef"], max_grad_norm=cfg["max_grad_norm"],
                               num_epochs=cfg["num_epochs"], minibatch_size=cfg["minibatch_size"], device=device)
    if name == "single_agent_dqn":
        from rl.agents.single_agent.dqn import SingleAgentDQN
        cfg = training_cfg["single_agent_dqn"]
        return SingleAgentDQN(OBSERVATION_DIM, learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
                               epsilon_start=cfg["epsilon_start"], epsilon_end=cfg["epsilon_end"],
                               epsilon_decay_steps=cfg["epsilon_decay_steps"],
                               target_update_interval=cfg["target_update_interval"],
                               batch_size=cfg["batch_size"], device=device)
    if name == "single_agent_sac":
        from rl.agents.single_agent.sac import SingleAgentSAC
        cfg = training_cfg["single_agent_sac"]
        return SingleAgentSAC(OBSERVATION_DIM, hidden_dim=cfg["hidden_dim"], learning_rate=cfg["learning_rate"],
                               gamma=cfg["gamma"], tau=cfg["tau"], alpha=cfg["alpha"],
                               batch_size=cfg["batch_size"], device=device)
    if name == "mappo":
        from rl.agents.multi_agent.mappo import MAPPO
        cfg = training_cfg["mappo"]
        return MAPPO(robot_ids, OBSERVATION_DIM, hidden_dim=cfg["hidden_dim"], actor_lr=cfg["actor_lr"],
                      critic_lr=cfg["critic_lr"], clip_epsilon=cfg["clip_epsilon"],
                      entropy_coef=cfg["entropy_coef"], max_grad_norm=cfg["max_grad_norm"],
                      num_epochs=cfg["num_epochs"], device=device)
    if name == "maddpg":
        from rl.agents.multi_agent.maddpg import MADDPG
        cfg = training_cfg["maddpg"]
        return MADDPG(robot_ids, OBSERVATION_DIM, hidden_dim=cfg["hidden_dim"], actor_lr=cfg["actor_lr"],
                       critic_lr=cfg["critic_lr"], gamma=cfg["gamma"], tau=cfg["tau"],
                       exploration_noise_std=cfg["exploration_noise_std"], batch_size=cfg["batch_size"], device=device)
    if name == "qmix":
        from rl.agents.multi_agent.qmix import QMIX
        cfg = training_cfg["qmix"]
        return QMIX(robot_ids, OBSERVATION_DIM, learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
                     target_update_interval=cfg["target_update_interval"], batch_size=cfg["batch_size"], device=device)
    if name == "hierarchical":
        from rl.agents.hierarchical.hierarchical_agent import HierarchicalAgent
        cfg = training_cfg["hierarchical"]
        return HierarchicalAgent(OBSERVATION_DIM, max_candidates=cfg["max_candidates"],
                                  decision_interval=cfg["decision_interval_steps"], hidden_dim=cfg["hidden_dim"],
                                  high_level_lr=cfg["high_level_lr"], low_level_lr=cfg["low_level_lr"], device=device)
    raise ValueError(f"unknown algorithm: {name!r} (choices: {ALGORITHM_CHOICES})")


def train_single_agent(algorithm, scenario: ExplorationScenario, total_env_steps: int, run_name: str,
                        checkpoint_manager: CheckpointManager, save_interval: int) -> None:
    """Training loop for the single-agent algorithms - one robot, direct AresEnv interaction."""
    env = AresEnv(**scenario.build_env_kwargs())
    observations = env.reset()
    robot_id = scenario.robot_ids[0]

    episode_reward = 0.0
    best_coverage = 0.0
    steps_done = 0

    while steps_done < total_env_steps:
        obs = observations.get(robot_id)
        if obs is None:
            observations = env.reset()
            continue

        if hasattr(algorithm, "act"):
            result = algorithm.act(obs)
            action, log_prob, value = result if isinstance(result, tuple) and len(result) == 3 else (result, None, None)
        else:
            raise RuntimeError("algorithm has no .act() method")

        actions = {robot_id: action}
        observations, rewards, dones, infos = env.step(actions)
        reward = rewards.get(robot_id, 0.0)
        episode_reward += reward
        steps_done += 1

        if hasattr(algorithm, "store_transition"):
            if log_prob is not None:  
                algorithm.store_transition(obs, action, log_prob, value, reward, dones.get(robot_id, False))
            else:  
                next_obs = observations.get(robot_id, obs)
                algorithm.store_transition(obs, action, reward, next_obs, dones.get(robot_id, False))

        if hasattr(algorithm, "buffer") and hasattr(algorithm.buffer, "__len__"):
            if len(algorithm.buffer) >= getattr(algorithm, "minibatch_size", 64) and hasattr(algorithm, "update"):
                try:
                    algorithm.update(last_value=0.0)
                except TypeError:
                    algorithm.update()

        if dones.get(robot_id, False):
            coverage = infos.get(robot_id, {}).get("coverage_fraction", 0.0)
            best_coverage = max(best_coverage, coverage)
            logger.info("step=%d episode_reward=%.3f coverage=%.3f", steps_done, episode_reward, coverage)
            episode_reward = 0.0
            observations = env.reset()

        if steps_done % save_interval == 0:
            checkpoint_manager.save(
                algorithm.state_dict(), run_name=run_name, step=steps_done,
                metrics={"coverage_fraction": best_coverage},
                is_best=(best_coverage >= checkpoint_manager.__dict__.get("_best_so_far", 0.0)),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config/training.yaml")
    parser.add_argument("--rewards-config", type=str, default="config/rewards.yaml")
    parser.add_argument("--scenario", type=str, default="exploration", choices=["exploration"])
    parser.add_argument("--algorithm", type=str, required=True, choices=ALGORITHM_CHOICES)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    with open(args.config) as f:
        training_cfg = yaml.safe_load(f)

    reward_config = RewardConfig.from_yaml(args.rewards_config)

    is_single = args.algorithm.startswith("single_agent")
    robot_ids = ["robot_01"] if is_single else [f"robot_{i+1:02d}" for i in range(args.num_robots)]

    scenario = ExplorationScenario(environment=STATIC_ENV_CONFIG, robot_ids=robot_ids, reward_config=reward_config)

    run_name = args.run_name or args.algorithm
    algorithm = build_algorithm(args.algorithm, robot_ids, training_cfg, args.device)
    checkpoint_manager = CheckpointManager()

    if is_single:
        total_steps = training_cfg[args.algorithm].get("total_env_steps", 100_000)
        save_interval = training_cfg["checkpointing"]["save_interval_env_steps"]
        logger.info("training %s for %d env steps (run_name=%s)", args.algorithm, total_steps, run_name)
        train_single_agent(algorithm, scenario, total_steps, run_name, checkpoint_manager, save_interval)
    else:
        logger.warning(
            "Multi-agent/hierarchical training loops (%s) require the full multi-robot rollout "
            "and per-condition buffering described in rl/agents/multi_agent/ and rl/agents/hierarchical/ - "
            "this script currently wires up the algorithm and scenario correctly but the multi-agent "
            "rollout loop itself is not yet implemented here. See training/train_marl.py / "
            "training/train_hrl.py for where that loop belongs.",
            args.algorithm,
        )

    logger.info("training run '%s' finished (or handed off - see warnings above)", run_name)


if __name__ == "__main__":
    main()
