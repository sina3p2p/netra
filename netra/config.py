from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_head: int = 64
    d_kv_latent: int = 512
    d_q_latent: int = 768
    d_rope: int = 32
    ffn_hidden: int = 2048
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    dropout: float = 0.0

    # MoE
    n_experts: int = 8
    n_active_experts: int = 2
    has_shared_expert: bool = True
    bias_update_speed: float = 0.001

    @property
    def d_nope(self) -> int:
        return self.d_head - self.d_rope
