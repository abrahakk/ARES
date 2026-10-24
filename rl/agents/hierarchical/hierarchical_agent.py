from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.agents.hierarchical.high_level_policy import HighLevelPolicy, GOAL_REGROUP, GOAL_HOLD
from rl.agents.hierarchical.low_level_policy import LowLevelPolicy
from rl.utils.normalization import RunningNormalizer
from rl.utils.replay_buffer import RolloutBuffer

SPECIAL_GOAL_OFFSETS = {
    GOAL_REGROUP: (0.0, 0.0),
    GOAL_HOLD: (0.0, 0.0),
}


class HierarchicalAgent:
    def __init__(
        self,
        observation_dim: int,
        max_candidates: int = 8,
        decision_interval: int = 20,
        hidden_dim: int = 256,
        high_level_lr: float = 3e-4,
        low_level_lr: float = 3e-4,
        device: str = "cpu",
    ) -> None:
        self.decision_interval = decision_interval
        self.max_candidates = max_candidates
        self.device = device

        self.high_level_policy = HighLevelPolicy(observation_dim, max_candidates, hidden_dim).to(device)
        self.low_level_policy = LowLevelPolicy(observation_dim, hidden_dim).to(device)

        self.high_level_optimizer = torch.optim.Adam(self.high_level_policy.parameters(), lr=high_level_lr)
        self.low_level_optimizer = torch.optim.Adam(self.low_level_policy.parameters(), lr=low_level_lr)

        self.normalizer = RunningNormalizer(shape=(observation_dim,))
        self.high_level_buffer = RolloutBuffer()
        self.low_level_buffer = RolloutBuffer()

        self._steps_since_decision = 0
        self._current_goal_offset: Tuple[float, float] = (0.0, 0.0)
        self._current__goal_index: Optional[int] = None
    def build_candidates(
        self, robot_pose: Tuple[float, float, float], known_frontiers: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        
        x0, y0, _theta = robot_pose
        num_frontier_slots = self.max_candidates - len(SPECIAL_GOAL_OFFSETS)
        frontiers = known_frontiers[:num_frontier_slots]

        offsets = np.zeros((self.max_candidates, 2), dtype=np.float32)
        mask = np.ones(self.max_candidates, dtype=bool)  
        labels: List = [None] * self.max_candidates

        idx = 0
        for fx, fy in frontiers:
            offsets[idx] = (fx - x0, fy - y0)
            mask[idx] = False
            labels[idx] = (fx, fy)
            idx += 1

        for name, offset in SPECIAL_GOAL_OFFSETS.items():
            offsets[idx] = offset
            mask[idx] = False
            labels[idx] = name
            idx += 1

        return offsets, mask, labels


    def maybe_select_goal(
        self,
        observation: np.ndarray,
        robot_pose: Tuple[float, float, float],
        known_frontiers: List[Tuple[float, float]],
        deterministic: bool = False,
    ) -> None:

        if self._steps_since_decision % self.decision_interval != 0:
            self._steps_since_decision += 1
            return

        offsets, mask, labels = self.build_candidates(robot_pose, known_frontiers)
        normalized = self.normalizer.normalize(observation.reshape(1, -1))
        obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        offsets_tensor = torch.as_tensor(offsets, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            goal_index, log_prob, value = self.high_level_policy.select_goal(
                obs_tensor, offsets_tensor, mask_tensor, deterministic=deterministic
            )

        idx = int(goal_index.item())
        self._current_goal_index = idx
        self._current_goal_offset = tuple(offsets[idx].tolist())
        self._steps_since_decision = 1  # reset the interval counter

        self.high_level_buffer.add(
            observation, idx, log_prob.item(), value.item(), reward=0.0, done=False
        )

    def act(self, observation: np.ndarray, deterministic: bool = False):
        normalized = self.normalizer.normalize(observation.reshape(1, -1))
        obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        goal_tensor = torch.as_tensor(self._current_goal_offset, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value = self.low_level_policy.act(obs_tensor, goal_tensor, deterministic=deterministic)

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item(), self._current_goal_offset

    def store_low_level_transition(self, observation, action, log_prob, value, reward, done) -> None:
        self.normalizer.update(observation.reshape(1, -1))
        self.low_level_buffer.add(observation, action, log_prob, value, reward, done)

    def accumulate_high_level_reward(self, reward: float) -> None:

        if self.high_level_buffer.rewards:
            self.high_level_buffer.rewards[-1] += reward

    def state_dict(self) -> dict:
        return {
            "high_level_policy": self.high_level_policy.state_dict(),
            "low_level_policy": self.low_level_policy.state_dict(),
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.high_level_policy.load_state_dict(state["high_level_policy"])
        self.low_level_policy.load_state_dict(state["low_level_policy"])
        self.normalizer.load_state_dict(state["normalizer"])
