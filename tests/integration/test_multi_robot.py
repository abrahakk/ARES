from __future__ import annotations

import numpy as np

from rl.environments.ares_env import AresEnv
from rl.environments.observation_space import OBSERVATION_DIM


def _zero_action_for_every_robot(observations: dict) -> dict:
    return {robot_id: np.zeros(3, dtype=np.float32) for robot_id in observations}


def test_ares_env_reset_returns_one_observation_per_robot():
    env = AresEnv(robot_ids=["robot_01", "robot_02"], world_bounds_m=(-4.0, -4.0, 4.0, 4.0), max_steps=50)
    observations = env.reset()

    assert set(observations.keys()) == {"robot_01", "robot_02"}
    for obs in observations.values():
        assert obs.shape == (OBSERVATION_DIM,)


def test_ares_env_step_returns_matching_keys_across_all_dicts():
    env = AresEnv(robot_ids=["robot_01", "robot_02"], world_bounds_m=(-4.0, -4.0, 4.0, 4.0), max_steps=50)
    observations = env.reset()
    actions = _zero_action_for_every_robot(observations)

    next_observations, rewards, dones, infos = env.step(actions)

    assert set(next_observations.keys()) == {"robot_01", "robot_02"}
    assert set(rewards.keys()) == {"robot_01", "robot_02"}
    assert set(dones.keys()) == {"robot_01", "robot_02"}
    assert set(infos.keys()) == {"robot_01", "robot_02"}


def test_ares_env_episode_terminates_at_max_steps():
    env = AresEnv(robot_ids=["robot_01"], world_bounds_m=(-2.0, -2.0, 2.0, 2.0), max_steps=5)
    observations = env.reset()

    done = False
    steps_taken = 0
    for _ in range(10):  
        if not observations:
            break
        actions = _zero_action_for_every_robot(observations)
        observations, rewards, dones, infos = env.step(actions)
        steps_taken += 1
        if dones and all(dones.values()):
            done = True
            break

    assert done is True
    assert steps_taken <= 5


def test_ares_env_coverage_map_accumulates_across_steps():
    env = AresEnv(robot_ids=["robot_01"], world_bounds_m=(-3.0, -3.0, 3.0, 3.0), max_steps=20)
    env.reset()

    coverage_before = env.coverage_map.coverage_fraction()
    for _ in range(5):
        env.step({"robot_01": np.array([0.1, 0.0, 0.0], dtype=np.float32)})
    coverage_after = env.coverage_map.coverage_fraction()

    
    assert coverage_after >= coverage_before
    assert 0.0 <= coverage_after <= 1.0


def test_ares_env_episode_summary_has_entry_per_robot():
    env = AresEnv(robot_ids=["robot_01", "robot_02"], world_bounds_m=(-3.0, -3.0, 3.0, 3.0), max_steps=10)
    observations = env.reset()
    for _ in range(3):
        actions = _zero_action_for_every_robot(observations)
        observations, _rewards, _dones, _infos = env.step(actions)
        if not observations:
            break

    summary = env.episode_summary()
    assert set(summary.keys()) == {"robot_01", "robot_02"}
