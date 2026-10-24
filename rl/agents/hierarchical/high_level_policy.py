from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

GOAL_REGROUP = "regroup"
GOAL_HOLD = "hold"


class HighLevelPolicy(nn.Module):


    def __init__(self, observation_dim: int, max_candidates: int = 8, hidden_dim: int = 256) -> None:
        super().__init__()
        self.max_candidates = max_candidates

        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
        )

        self.candidate_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2), nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor, candidate_offsets: torch.Tensor, candidate_mask: torch.Tensor):
        obs_features = self.observation_encoder(observation)  
        obs_expanded = obs_features.unsqueeze(1).expand(-1, self.max_candidates, -1)

        candidate_features = self.candidate_encoder(candidate_offsets)  
        combined = torch.cat([obs_expanded, candidate_features], dim=-1)
        scores = self.score_head(combined).squeeze(-1)  

        scores = scores.masked_fill(candidate_mask, float("-inf"))
        dist = Categorical(logits=scores)
        value = self.value_head(obs_features).squeeze(-1)
        return dist, value

    def select_goal(
        self,
        observation: torch.Tensor,
        candidate_offsets: torch.Tensor,
        candidate_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(observation, candidate_offsets, candidate_mask)
        goal_index = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(goal_index)
        return goal_index, log_prob, value
