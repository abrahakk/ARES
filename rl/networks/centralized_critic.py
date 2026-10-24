from __future__ import annotations

import torch
import torch.nn as nn

from rl.networks.attention import MultiHeadAttentionEncoder


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        per_agent_dim = observation_dim + action_dim

        self.per_agent_encoder = nn.Sequential(
            nn.Linear(per_agent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = MultiHeadAttentionEncoder(
            embed_dim=hidden_dim, num_heads=num_attention_heads
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_observations: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:

        per_agent = torch.cat([joint_observations, joint_actions], dim=-1)
        encoded = self.per_agent_encoder(per_agent)         
        pooled = self.attention(encoded)                    
        return self.value_head(pooled).squeeze(-1)
