from __future__ import annotations

import logging
from typing import List

import numpy as np

from rl.environments.ares_env import AresEnv
from rl.utils.checkpoints import CheckpointManager
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)


def train_marl(
    algorithm_name: str,
    algorithm,
    scenario: ExplorationScenario,
    total_env_steps: int,
    run_name: str,
    checkpoint_manager: CheckpointManager,
    save_interval: int,
    log_interval_episodes: int = 10,
) -> float:
    if algorithm_name == "mappo":
        return _train_mappo(algorithm, scenario, total_env_steps, run_name, checkpoint_manager, save_interval, log_interval_episodes)
    if algorithm_name == "maddpg":
        return _train_maddpg(algorithm, scenario, total_env_steps, run_name, checkpoint_manager, save_interval, log_interval_episodes)
    if algorithm_name == "qmix":
        return _train_qmix(algorithm, scenario, total_env_steps, run_name, checkpoint_manager, save_interval, log_interval_episodes)
    raise ValueError(f"train_marl does not support algorithm_name={algorithm_name!r} (expected mappo/maddpg/qmix)")


def _train_mappo(algorithm, scenario, total_env_steps, run_name, checkpoint_manager, save_interval, log_interval_episodes) -> float:
    env = AresEnv(**scenario.build_env_kwargs())
    observations = env.reset()
    robot_ids = scenario.robot_ids

    episode_rewards = {rid: 0.0 for rid in robot_ids}
    episode_count = 0
    best_coverage = 0.0
    steps_done = 0

    while steps_done < total_env_steps:
        if not observations:
            observations = env.reset()
            episode_rewards = {rid: 0.0 for rid in robot_ids}
            continue

        act_results = algorithm.act(observations)  # {robot_id: {"action": ..., "log_prob": ...}}
        actions = {rid: r["action"] for rid, r in act_results.items()}

        next_observations, rewards, dones, infos = env.step(actions)
        steps_done += 1

        
        joint_obs = np.stack([observations[rid] for rid in robot_ids if rid in observations])
        joint_actions = np.stack([actions[rid] for rid in robot_ids if rid in actions])
        shared_value = algorithm.compute_centralized_value(joint_obs, joint_actions)

        for rid in list(actions.keys()):
            reward = rewards.get(rid, 0.0)
            episode_rewards[rid] = episode_rewards.get(rid, 0.0) + reward
            algorithm.store_transition(
                rid, observations[rid], actions[rid], act_results[rid]["log_prob"],
                shared_value, reward, dones.get(rid, False),
            )

        observations = next_observations

        buffer_lengths = [len(b) for b in algorithm.buffers.values()]
        if buffer_lengths and min(buffer_lengths) >= 512:
            last_values = {rid: 0.0 for rid in robot_ids}
            algorithm.update(last_values)

        if dones and all(dones.values()):
            episode_count += 1
            coverage = next(iter(infos.values()), {}).get("coverage_fraction", 0.0) if infos else 0.0
            best_coverage = max(best_coverage, coverage)
            if episode_count % log_interval_episodes == 0:
                mean_reward = sum(episode_rewards.values()) / max(len(episode_rewards), 1)
                logger.info(
                    "run=%s step=%d episode=%d mean_episode_reward=%.3f coverage=%.3f best_coverage=%.3f",
                    run_name, steps_done, episode_count, mean_reward, coverage, best_coverage,
                )
            observations = env.reset()
            episode_rewards = {rid: 0.0 for rid in robot_ids}

        if steps_done % save_interval == 0:
            checkpoint_manager.save(
                algorithm.state_dict(), run_name=run_name, step=steps_done,
                metrics={"coverage_fraction": best_coverage}, is_best=False,
                config_path="config/training.yaml",
            )

    logger.info("MAPPO run '%s' finished: %d steps, %d episodes, best_coverage=%.3f",
                run_name, steps_done, episode_count, best_coverage)
    return best_coverage


def _train_maddpg(algorithm, scenario, total_env_steps, run_name, checkpoint_manager, save_interval, log_interval_episodes) -> float:
    env = AresEnv(**scenario.build_env_kwargs())
    observations = env.reset()
    robot_ids: List[str] = scenario.robot_ids

    episode_count = 0
    best_coverage = 0.0
    steps_done = 0

    while steps_done < total_env_steps:
        if len(observations) < len(robot_ids):
            observations = env.reset()
            continue

        actions = algorithm.act(observations)
        next_observations, rewards, dones, infos = env.step(actions)
        steps_done += 1
        if len(next_observations) == len(robot_ids): 
            joint_obs = np.stack([observations[rid] for rid in robot_ids])
            joint_actions = np.stack([actions[rid] for rid in robot_ids])
            joint_next_obs = np.stack([next_observations[rid] for rid in robot_ids])
            joint_rewards = np.array([rewards.get(rid, 0.0) for rid in robot_ids], dtype=np.float32)
            joint_dones = np.array([float(dones.get(rid, False)) for rid in robot_ids], dtype=np.float32)

            algorithm.store_transition(joint_obs, joint_actions, joint_rewards, joint_next_obs, joint_dones)
            algorithm.update()

        observations = next_observations

        if dones and all(dones.values()):
            episode_count += 1
            coverage = next(iter(infos.values()), {}).get("coverage_fraction", 0.0) if infos else 0.0
            best_coverage = max(best_coverage, coverage)
            if episode_count % log_interval_episodes == 0:
                logger.info("run=%s step=%d episode=%d coverage=%.3f best_coverage=%.3f",
                            run_name, steps_done, episode_count, coverage, best_coverage)
            observations = env.reset()

        if steps_done % save_interval == 0:
            checkpoint_manager.save(
                {"actors": {rid: a.state_dict() for rid, a in algorithm.actors.items()}},
                run_name=run_name, step=steps_done,
                metrics={"coverage_fraction": best_coverage}, is_best=False,
                config_path="config/training.yaml",
            )

    logger.info("MADDPG run '%s' finished: %d steps, %d episodes, best_coverage=%.3f",
                run_name, steps_done, episode_count, best_coverage)
    return best_coverage


def _train_qmix(algorithm, scenario, total_env_steps, run_name, checkpoint_manager, save_interval, log_interval_episodes) -> float:
    env = AresEnv(**scenario.build_env_kwargs())
    observations = env.reset()
    robot_ids: List[str] = scenario.robot_ids

    episode_count = 0
    best_coverage = 0.0
    steps_done = 0

    from rl.agents.single_agent.dqn import ACTION_TABLE

    while steps_done < total_env_steps:
        if len(observations) < len(robot_ids):
            observations = env.reset()
            continue

        action_dicts = algorithm.act(observations)  
        next_observations, rewards, dones, infos = env.step(
            {rid: np.array(action_dicts[rid], dtype=np.float32) for rid in action_dicts}
        )
        steps_done += 1

        if len(next_observations) == len(robot_ids):
            joint_obs = np.stack([observations[rid] for rid in robot_ids])
            joint_next_obs = np.stack([next_observations[rid] for rid in robot_ids])
            action_indices = np.array(
                [ACTION_TABLE.index(action_dicts[rid]) for rid in robot_ids], dtype=np.int64
            )

            team_reward = float(np.mean([rewards.get(rid, 0.0) for rid in robot_ids]))
            team_done = float(all(dones.get(rid, False) for rid in robot_ids))

            algorithm.store_transition(joint_obs, action_indices, team_reward, joint_next_obs, team_done)
            algorithm.update()

        observations = next_observations

        if dones and all(dones.values()):
            episode_count += 1
            coverage = next(iter(infos.values()), {}).get("coverage_fraction", 0.0) if infos else 0.0
            best_coverage = max(best_coverage, coverage)
            if episode_count % log_interval_episodes == 0:
                logger.info("run=%s step=%d episode=%d coverage=%.3f best_coverage=%.3f",
                            run_name, steps_done, episode_count, coverage, best_coverage)
            observations = env.reset()

        if steps_done % save_interval == 0:
            checkpoint_manager.save(
                {"agent_q": algorithm.agent_q.state_dict(), "mixer": algorithm.mixer.state_dict()},
                run_name=run_name, step=steps_done,
                metrics={"coverage_fraction": best_coverage}, is_best=False,
                config_path="config/training.yaml",
            )

    logger.info("QMIX run '%s' finished: %d steps, %d episodes, best_coverage=%.3f",
                run_name, steps_done, episode_count, best_coverage)
    return best_coverage
