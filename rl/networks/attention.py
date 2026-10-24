from __future__ import annotations

import torch
import torch.nn as nn


class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, items: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = items.shape[0]
        query = self.query.expand(batch_size, -1, -1)  

        pooled, _attn_weights = self.attention(
            query, items, items, key_padding_mask=key_padding_mask
        )
        pooled = pooled.squeeze(1)  
        return self.norm(pooled)
