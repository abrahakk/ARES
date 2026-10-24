from __future__ import annotations

import logging

from rl.environments.ares_env import AresEnv
from rl.utils.checkpoints import CheckpointManager
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)


def train_single_agent(
    algorithm,
    scenario: ExplorationScenario,
    total_env_steps: int,
    run_name: str,
    checkpoint_manager: CheckpointManager,
    save_interval: int,
    log_interval_episodes: int = 10,
) -> float:

    env = AresEnv(**scenario.build_env_kwargs())
    observations = env.reset()
    robot_id = scenario.robot_ids[0]

    episode_reward = 0.0
    episode_count = 0
    best_coverage = 0.0
    steps_done = 0

    while steps_done < total_env_steps:
        obs = observations.get(robot_id)
        if obs is None:
            observations = env.reset()
            continue

        act_result = algorithm.act(obs)
        is_ppo_style = isinstance(act_result, tuple) and len(act_result) == 3
        if is_ppo_style:
            action, log_prob, value = act_result
        else:
            action = act_result

        observations, rewards, dones, infos = env.step({robot_id: action})
        reward = rewards.get(robot_id, 0.0)
        done = dones.get(robot_id, False)
        episode_reward += reward
        steps_done += 1

        if is_ppo_style:
            algorithm.store_transition(obs, action, log_prob, value, reward, done)
            if hasattr(algorithm, "buffer") and len(algorithm.buffer) >= 2048:
                algorithm.update(last_value=0.0 if done else value)
        else:
            next_obs = observations.get(robot_id, obs)
            algorithm.store_transition(obs, action, reward, next_obs, done)
            if hasattr(algorithm, "update"):
                algorithm.update()

        if done:
            episode_count += 1
            coverage = infos.get(robot_id, {}).get("coverage_fraction", 0.0)
            best_coverage = max(best_coverage, coverage)

            if episode_count % log_interval_episodes == 0:
                logger.info(
                    "run=%s step=%d episode=%d episode_reward=%.3f coverage=%.3f best_coverage=%.3f",
                    run_name, steps_done, episode_count, episode_reward, coverage, best_coverage,
                )

            episode_reward = 0.0
            observations = env.reset()

        if steps_done % save_interval == 0:
            checkpoint_manager.save(
                algorithm.state_dict(), run_name=run_name, step=steps_done,
                metrics={"coverage_fraction": best_coverage},
                is_best=(best_coverage >= checkpoint_manager.__dict__.get("_best_so_far", {}).get(run_name, 0.0)),
                config_path="config/training.yaml",
            )

    logger.info("training run '%s' finished: %d steps, %d episodes, best_coverage=%.3f",
                run_name, steps_done, episode_count, best_coverage)
    return best_coverage
