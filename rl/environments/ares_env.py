from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from robots.agents.robot_manager import RobotManager
from rl.coordination.cooperative_exploration import CoverageMap
from rl.environments.action_space import action_vector_to_dict
from rl.environments.observation_space import flatten_observation, OBSERVATION_DIM
from rl.environments.reward import RewardConfig, compute_reward, aggregate_episode_breakdown

logger = logging.getLogger(__name__)


class AresEnv:
    def __init__(
        self,
        robot_ids: List[str],
        world_bounds_m: Tuple[float, float, float, float],  
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 2000,
        coverage_success_threshold: float = 0.9,
        webots_kwargs: Optional[dict] = None,
    ) -> None:
        self.robot_ids = robot_ids
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.coverage_success_threshold = coverage_success_threshold

        self.robot_manager = RobotManager(robot_ids)
        self.robot_manager.spawn_all(**(webots_kwargs or {}))

        self.coverage_map = CoverageMap(world_bounds_m)

        self._step_count = 0
        self._episode_breakdowns: Dict[str, List[dict]] = {rid: [] for rid in robot_ids}
        self._last_pose: Dict[str, Tuple[float, float]] = {}


    def reset(self) -> Dict[str, np.ndarray]:
        self._step_count = 0
        self._episode_breakdowns = {rid: [] for rid in self.robot_ids}
        self.coverage_map.reset()
        self._last_pose = {}

        raw_obs = self.robot_manager.get_observations()
        return {rid: flatten_observation(obs) for rid, obs in raw_obs.items()}

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        action_dicts = {rid: action_vector_to_dict(a) for rid, a in actions.items()}
        self.robot_manager.apply_actions(action_dicts)
        sim_running = self.robot_manager.step()
        self._step_count += 1

        raw_obs = self.robot_manager.get_observations()
        states = self.robot_manager.get_states()

        observations: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        dones: Dict[str, bool] = {}
        infos: Dict[str, dict] = {}
        episode_timeout = self._step_count >= self.max_steps
        coverage_fraction = self.coverage_map.coverage_fraction()
        mission_complete = coverage_fraction >= self.coverage_success_threshold

        for rid in self.robot_ids:
            if rid not in raw_obs:
                continue

            state = states[rid]
            pose = state.pose.as_tuple()
            prev_pose = self._last_pose.get(rid, (pose[0], pose[1]))
            distance_traveled = float(np.hypot(pose[0] - prev_pose[0], pose[1] - prev_pose[1]))
            self._last_pose[rid] = (pose[0], pose[1])

            newly_covered, entropy_reduction = self.coverage_map.update_from_robot(
                rid, pose, raw_obs[rid]["detected_obstacles"]
            )

            reward, breakdown = compute_reward(
                newly_covered_area_m2=newly_covered,
                joint_entropy_reduction=entropy_reduction,
                cooperation_event=self._detect_cooperation_event(rid, raw_obs[rid]),
                collided=state.collision_count > 0 and distance_traveled < 1e-3,
                distance_traveled_m=distance_traveled,
                message_sent=action_dicts[rid]["should_communicate"],
                config=self.reward_config,
            )

            self._episode_breakdowns[rid].append(breakdown)
            observations[rid] = flatten_observation(raw_obs[rid])
            rewards[rid] = reward
            dones[rid] = episode_timeout or mission_complete or not sim_running
            infos[rid] = {"coverage_fraction": coverage_fraction, **breakdown}

        return observations, rewards, dones, infos

    def episode_summary(self) -> Dict[str, dict]:
        
        return {rid: aggregate_episode_breakdown(steps) for rid, steps in self._episode_breakdowns.items()}

    def _detect_cooperation_event(self, robot_id: str, observation: dict) -> bool:
        
        return len(observation.get("messages_received", [])) > 0

    @property
    def observation_dim(self) -> int:
        return OBSERVATION_DIM
