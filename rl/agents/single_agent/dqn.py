from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.utils.replay_buffer import ReplayBuffer
from rl.utils.normalization import RunningNormalizer

LINEAR_BINS = [-0.3, -0.15, 0.0, 0.15, 0.3]
ANGULAR_BINS = [-0.2, -0.1, 0.0, 0.1, 0.2]
COMM_BINS = [0.0, 1.0]


def _build_action_table() -> List[Tuple[float, float, float]]:
    table = []
    for lin in LINEAR_BINS:
        for ang in ANGULAR_BINS:
            for comm in COMM_BINS:
                table.append((lin, ang, comm))
    return table


ACTION_TABLE = _build_action_table()
NUM_DISCRETE_ACTIONS = len(ACTION_TABLE)


class _QNetwork(nn.Module):
    def __init__(self, observation_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleAgentDQN:
    def __init__(
        self,
        observation_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_update_interval: int = 500,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.batch_size = batch_size

        self.q_network = _QNetwork(observation_dim, NUM_DISCRETE_ACTIONS).to(device)
        self.target_network = _QNetwork(observation_dim, NUM_DISCRETE_ACTIONS).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.normalizer = RunningNormalizer(shape=(observation_dim,))
        self.buffer = ReplayBuffer()

        self._train_steps = 0

    def _epsilon(self) -> float:
        fraction = min(1.0, self._train_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        normalized = self.normalizer.normalize(observation.reshape(1, -1))
        if not deterministic and random.random() < self._epsilon():
            action_index = random.randrange(NUM_DISCRETE_ACTIONS)
        else:
            obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
            action_index = int(torch.argmax(q_values, dim=-1).item())
        return np.array(ACTION_TABLE[action_index], dtype=np.float32)

    def store_transition(self, observation, action, reward, next_observation, done) -> None:
        self.normalizer.update(observation.reshape(1, -1))
        action_index = ACTION_TABLE.index(tuple(np.round(action, 4).tolist())) if isinstance(action, np.ndarray) else action
        self.buffer.add(observation, [action_index], reward, next_observation, done)

    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size, device=self.device)
        actions = batch["actions"].long().squeeze(-1)

        q_values = self.q_network(batch["observations"]).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_values = self.target_network(batch["next_observations"]).max(dim=-1).values
            targets = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
