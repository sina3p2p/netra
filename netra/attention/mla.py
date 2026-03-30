import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..norm import RMSNorm
from ..rope import apply_rotary_emb


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) from DeepSeek V2/V3.

    Q path:  hidden -> W_dq (compress to d_q_latent) -> W_uq (expand to n_heads * d_nope)
                                                      + W_qr (expand to n_heads * d_rope, then RoPE)
    KV path: hidden -> W_dkv (compress to d_kv_latent)  [<-- this is what gets cached]
             latent -> W_uk (expand to n_heads * d_nope)
                    + W_kr (expand to n_heads * d_rope, then RoPE)
             latent -> W_uv (expand to n_heads * d_head)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_nope = config.d_nope
        self.d_rope = config.d_rope
        self.d_kv_latent = config.d_kv_latent
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Q path
        self.W_dq = nn.Linear(config.d_model, config.d_q_latent, bias=False)
        self.q_norm = RMSNorm(config.d_q_latent, eps=config.norm_eps)
        self.W_uq = nn.Linear(config.d_q_latent, config.n_heads * config.d_nope, bias=False)
        self.W_qr = nn.Linear(config.d_q_latent, config.n_heads * config.d_rope, bias=False)

        # KV path
        self.W_dkv = nn.Linear(config.d_model, config.d_kv_latent, bias=False)
        self.kv_norm = RMSNorm(config.d_kv_latent, eps=config.norm_eps)
        self.W_uk = nn.Linear(config.d_kv_latent, config.n_heads * config.d_nope, bias=False)
        self.W_kr = nn.Linear(config.d_kv_latent, config.n_heads * config.d_rope, bias=False)
        self.W_uv = nn.Linear(config.d_kv_latent, config.n_heads * config.d_head, bias=False)

        # Output
        self.W_o = nn.Linear(config.n_heads * config.d_head, config.d_model, bias=False)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        B, S, _ = x.shape

        # Q: compress -> expand -> split nope + rope portions
        q_latent = self.q_norm(self.W_dq(x))
        q_nope = self.W_uq(q_latent).view(B, S, self.n_heads, self.d_nope).transpose(1, 2)
        q_rope = self.W_qr(q_latent).view(B, S, self.n_heads, self.d_rope).transpose(1, 2)
        q_rope = apply_rotary_emb(q_rope, rope_cos, rope_sin)

        # KV: compress -> expand
        kv_latent = self.kv_norm(self.W_dkv(x))
        k_nope = self.W_uk(kv_latent).view(B, S, self.n_heads, self.d_nope).transpose(1, 2)
        k_rope = self.W_kr(kv_latent).view(B, S, self.n_heads, self.d_rope).transpose(1, 2)
        k_rope = apply_rotary_emb(k_rope, rope_cos, rope_sin)
        v = self.W_uv(kv_latent).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # Reassemble full Q and K
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Causal attention (old)
        # attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # causal_mask = torch.triu(
        #     torch.full((S, S), float("-inf"), device=x.device), diagonal=1
        # )
        # attn_weights = attn_weights + causal_mask
        # attn_weights = F.softmax(attn_weights, dim=-1)

        # out = torch.matmul(attn_weights, v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
