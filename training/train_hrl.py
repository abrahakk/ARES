from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from rl.agents.hierarchical.hierarchical_agent import HierarchicalAgent
from rl.environments.ares_env import AresEnv
from rl.utils.checkpoints import CheckpointManager
from simulation.scenarios.exploration import ExplorationScenario

logger = logging.getLogger(__name__)


def _mock_known_frontiers(robot_pose: Tuple[float, float, float], world_bounds_m) -> List[Tuple[float, float]]:

    x, y, _theta = robot_pose
    xmin, ymin, xmax, ymax = world_bounds_m
    candidates = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    return [(max(xmin, min(xmax, cx)), max(ymin, min(ymax, cy))) for cx, cy in candidates]


def train_hrl(
    agent: HierarchicalAgent,
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
    world_bounds = scenario.environment.world_bounds_m

    episode_reward = 0.0
    episode_count = 0
    best_coverage = 0.0
    steps_done = 0

    while steps_done < total_env_steps:
        obs = observations.get(robot_id)
        if obs is None:
            observations = env.reset()
            continue

        robot_state = env.robot_manager.robots[robot_id].state
        robot_pose = robot_state.pose.as_tuple()
        frontiers = _mock_known_frontiers(robot_pose, world_bounds)

        agent.maybe_select_goal(obs, robot_pose, frontiers)
        action, log_prob, value, _goal_offset = agent.act(obs)

        observations, rewards, dones, infos = env.step({robot_id: action})
        reward = rewards.get(robot_id, 0.0)
        done = dones.get(robot_id, False)
        episode_reward += reward
        steps_done += 1

        agent.store_low_level_transition(obs, action, log_prob, value, reward, done)
        agent.accumulate_high_level_reward(reward)

        if len(agent.low_level_buffer) >= agent.decision_interval:
            agent.update_low_level(last_value=0.0 if done else value)

        if len(agent.high_level_buffer) >= 32:
            agent.update_high_level(last_value=0.0)

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
                agent.state_dict(), run_name=run_name, step=steps_done,
                metrics={"coverage_fraction": best_coverage}, is_best=False,
                config_path="config/training.yaml",
            )

    logger.info("HRL run '%s' finished: %d steps, %d episodes, best_coverage=%.3f",
                run_name, steps_done, episode_count, best_coverage)
    return best_coverage
