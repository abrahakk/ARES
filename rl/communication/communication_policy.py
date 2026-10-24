from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from robots.communication.message import Message
from rl.communication.message_encoder import MessageEncoder, ENCODED_MESSAGE_DIM
from rl.networks.attention import MultiHeadAttentionEncoder
from rl.environments.observation_space import MESSAGE_SUMMARY_DIM


class CommunicationPolicy(nn.Module):

    def __init__(self, hidden_dim: int = 64, num_heads: int = 2) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(ENCODED_MESSAGE_DIM, hidden_dim),
            nn.ReLU(),
        )
        self.attention = MultiHeadAttentionEncoder(embed_dim=hidden_dim, num_heads=num_heads)
        self.output_projection = nn.Linear(hidden_dim, MESSAGE_SUMMARY_DIM)

    def forward(self, encoded_messages: torch.Tensor) -> torch.Tensor:
        
        projected = self.project(encoded_messages)
        return self.output_projection(self.attention(projected))

    @staticmethod
    def zeros_summary(device: str = "cpu") -> torch.Tensor:
        return torch.zeros(MESSAGE_SUMMARY_DIM, device=device)

    def encode_inbox(self, messages: List[Message], device: str = "cpu") -> torch.Tensor:
        
        if not messages:
            return self.zeros_summary(device=device)

        encoded = MessageEncoder.encode_batch(messages)  
        tensor = torch.as_tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
        summary = self.forward(tensor)  
        return summary.squeeze(0)
