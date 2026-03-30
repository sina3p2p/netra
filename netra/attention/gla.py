import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention (Yang et al., 2024).

    Replaces softmax attention with a gated linear recurrence:
      h_t = α_t · h_{t-1} + k_t ⊗ v_t    (state update with forget gate)
      o_t = q_t^T · h_t                    (readout)

    The data-dependent forget gate α controls how much past context
    to retain, giving better quality than vanilla linear attention.

    Training uses the parallel form (O(n²) materialized decay matrix).
    This is efficient for seq_len ≤ 2048; for longer sequences a
    chunk-wise algorithm would be needed.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        d_inner = config.n_heads * config.d_head

        self.W_q = nn.Linear(config.d_model, d_inner, bias=False)
        self.W_k = nn.Linear(config.d_model, d_inner, bias=False)
        self.W_v = nn.Linear(config.d_model, d_inner, bias=False)
        self.W_gate = nn.Linear(config.d_model, d_inner, bias=False)
        self.W_alpha = nn.Linear(config.d_model, config.n_heads, bias=True)
        self.W_o = nn.Linear(d_inner, config.d_model, bias=False)

        self.group_norm = nn.GroupNorm(config.n_heads, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.W_q(x).view(B, S, H, D).transpose(1, 2)
        k = self.W_k(x).view(B, S, H, D).transpose(1, 2)
        v = self.W_v(x).view(B, S, H, D).transpose(1, 2)

        q = F.silu(q)
        k = F.silu(k)

        # Forget gates in log-space: log(α) ∈ (-∞, 0]
        log_alpha = -F.softplus(self.W_alpha(x))           # (B, S, H)
        log_alpha = log_alpha.transpose(1, 2).unsqueeze(-1) # (B, H, S, 1)

        # Cumulative log-gates → pairwise decay matrix
        cum_log = torch.cumsum(log_alpha, dim=2)            # (B, H, S, 1)
        # decay[i,j] = exp(cum_log[i] - cum_log[j]),  ∈ [0, 1] for j ≤ i
        decay = cum_log - cum_log.transpose(2, 3)           # (B, H, S, S)

        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=x.device), diagonal=1
        )
        decay = torch.exp(decay + causal_mask)

        # Gated linear attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * decay
        out = torch.matmul(attn, v)                         # (B, H, S, D)

        # Per-head GroupNorm + output gate
        out = out.transpose(1, 2).contiguous().view(B, S, H * D)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = out * torch.sigmoid(self.W_gate(x))

        return self.W_o(out)
