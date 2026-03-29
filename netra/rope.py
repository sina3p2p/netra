import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputes and caches the RoPE sin/cos frequency table."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        pair_idx = torch.arange(0, dim // 2, dtype=torch.float32)
        inv_freq = base ** (-2.0 * pair_idx / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = t[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE to x of shape (batch, n_heads, seq_len, d_rope).
    cos/sin have shape (seq_len, d_rope // 2).
    """
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
