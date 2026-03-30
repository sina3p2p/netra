import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding
from .block import TransformerBlock


class Netra(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # RoPE is only needed when at least one layer uses MLA
        needs_rope = config.attention_type != "gla"
        self.rotary = RotaryEmbedding(config.d_rope, config.max_seq_len) if needs_rope else None

        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        B, S = input_ids.shape
        x = self.tok_emb(input_ids)

        rope_cos, rope_sin = None, None
        if self.rotary is not None:
            rope_cos, rope_sin = self.rotary(S)

        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss
