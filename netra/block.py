import torch
import torch.nn as nn

from .config import ModelConfig
from .norm import RMSNorm
from .attention import MultiHeadLatentAttention, GatedLinearAttention
from .moe import MoELayer


def _uses_mla(config: ModelConfig, layer_idx: int) -> bool:
    if config.attention_type == "mla":
        return True
    if config.attention_type == "gla":
        return False
    # hybrid: even layers → MLA, odd layers → GLA
    return layer_idx % 2 == 0


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.use_mla = _uses_mla(config, layer_idx)

        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if self.use_mla:
            self.attn = MultiHeadLatentAttention(config)
        else:
            self.attn = GatedLinearAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.moe = MoELayer(config)

    def forward(self, x: torch.Tensor, rope_cos=None, rope_sin=None):
        if self.use_mla:
            x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        else:
            x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.ffn_norm(x))
        return x
