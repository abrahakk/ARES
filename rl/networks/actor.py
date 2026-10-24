from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli


class ActorNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 256,
        max_linear_velocity: float = 0.3,
        max_angular_velocity: float = 2.0,
    ) -> None:
        super().__init__()
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.motion_mean_head = nn.Linear(hidden_dim, 2)
        self.motion_log_std = nn.Parameter(torch.zeros(2) - 0.5)


        self.comm_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor):
        features = self.backbone(observation)

        motion_mean = torch.tanh(self.motion_mean_head(features))
        motion_std = torch.exp(self.motion_log_std).expand_as(motion_mean)
        motion_dist = Normal(motion_mean, motion_std)

        comm_logit = self.comm_head(features)
        comm_dist = Bernoulli(logits=comm_logit)

        value = self.value_head(features)
        return motion_dist, comm_dist, value

    def act(self, observation: torch.Tensor, deterministic: bool = False):
        
        motion_dist, comm_dist, value = self.forward(observation)

        if deterministic:
            motion_raw = motion_dist.mean
            comm = (torch.sigmoid(comm_dist.logits) > 0.5).float()
        else:
            motion_raw = motion_dist.rsample()
            comm = comm_dist.sample()

        log_prob = motion_dist.log_prob(motion_raw).sum(-1) + comm_dist.log_prob(comm).squeeze(-1)

        linear = torch.tanh(motion_raw[..., 0]) * self.max_linear_velocity
        angular = torch.tanh(motion_raw[..., 1]) * self.max_angular_velocity
        action = torch.stack([linear, angular, comm.squeeze(-1)], dim=-1)

        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor):
      
        motion_dist, comm_dist, value = self.forward(observation)

      
        linear_raw = torch.atanh(torch.clamp(action[..., 0] / self.max_linear_velocity, -0.999, 0.999))
        angular_raw = torch.atanh(torch.clamp(action[..., 1] / self.max_angular_velocity, -0.999, 0.999))
        motion_raw = torch.stack([linear_raw, angular_raw], dim=-1)
        comm = action[..., 2:3]

        log_prob = motion_dist.log_prob(motion_raw).sum(-1) + comm_dist.log_prob(comm).squeeze(-1)
        entropy = motion_dist.entropy().sum(-1) + comm_dist.entropy().squeeze(-1)

        return log_prob, entropy, value.squeeze(-1)

    def export_forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      
        features = self.backbone(observation)
        motion_mean = torch.tanh(self.motion_mean_head(features))
        linear = motion_mean[..., 0] * self.max_linear_velocity
        angular = motion_mean[..., 1] * self.max_angular_velocity
        comm = (torch.sigmoid(self.comm_head(features)) > 0.5).float().squeeze(-1)
        action = torch.stack([linear, angular, comm], dim=-1)
        value = self.value_head(features).squeeze(-1)
        return action, value
