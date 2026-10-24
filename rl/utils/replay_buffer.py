from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class RolloutBuffer:


    gamma: float = 0.99
    gae_lambda: float = 0.95

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, observation, action, log_prob, value, reward, done) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:

        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        values = self.values + [last_value]

        for t in reversed(range(n)):
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def as_tensors(self, last_value: float, device: str = "cpu") -> Dict[str, torch.Tensor]:
        returns, advantages = self.compute_returns_and_advantages(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return {
            "observations": torch.as_tensor(np.array(self.observations), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=device),
            "old_log_probs": torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        }

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class ReplayBuffer:


    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, done) -> None:
        self._buffer.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        obs, act, rew, next_obs, done = zip(*batch)
        return {
            "observations": torch.as_tensor(np.array(obs), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.array(act), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(np.array(rew), dtype=torch.float32, device=device),
            "next_observations": torch.as_tensor(np.array(next_obs), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.array(done), dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return len(self._buffer)
