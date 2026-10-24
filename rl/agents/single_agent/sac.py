from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rl.utils.replay_buffer import ReplayBuffer
from rl.utils.normalization import RunningNormalizer

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class _GaussianPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)


class _QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SingleAgentSAC:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.policy = _GaussianPolicy(observation_dim, action_dim, hidden_dim).to(device)
        self.q1 = _QNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.q2 = _QNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.q1_target = _QNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.q2_target = _QNetwork(observation_dim, action_dim, hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=learning_rate
        )

        self.normalizer = RunningNormalizer(shape=(observation_dim,))
        self.buffer = ReplayBuffer()

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        normalized = self.normalizer.normalize(observation.reshape(1, -1))
        obs_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(obs_tensor)
        return action.squeeze(0).cpu().numpy()

    def store_transition(self, observation, action, reward, next_observation, done) -> None:
        self.normalizer.update(observation.reshape(1, -1))
        self.buffer.add(observation, action, reward, next_observation, done)

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, device=self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(batch["next_observations"])
            q1_next = self.q1_target(batch["next_observations"], next_action)
            q2_next = self.q2_target(batch["next_observations"], next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = batch["rewards"] + self.gamma * (1.0 - batch["dones"]) * min_q_next

        q1_pred = self.q1(batch["observations"], batch["actions"])
        q2_pred = self.q2(batch["observations"], batch["actions"])
        q_loss = nn.functional.mse_loss(q1_pred, q_target) + nn.functional.mse_loss(q2_pred, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_action, log_prob = self.policy.sample(batch["observations"])
        q1_new = self.q1(batch["observations"], new_action)
        q2_new = self.q2(batch["observations"], new_action)
        policy_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data)
