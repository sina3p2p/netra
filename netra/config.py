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

    # Attention: "mla", "gla", or "hybrid"
    attention_type: str = "hybrid"
    gla_every_n: int = 4  # hybrid only: 1 GLA per N layers (4 → 3:1 MLA:GLA)

    @property
    def d_nope(self) -> int:
        return self.d_head - self.d_rope

    # ── Presets ────────────────────────────────────────────────────────

    @classmethod
    def nano(cls, **kwargs):
        """~16M params · 4 layers · for quick sanity checks (minutes on 1 GPU)."""
        defaults = dict(
            vocab_size=32_000,
            d_model=256, n_layers=4, n_heads=4, d_head=64,
            d_kv_latent=128, d_q_latent=256, d_rope=32,
            ffn_hidden=512, max_seq_len=512,
            n_experts=4, n_active_experts=2,
            has_shared_expert=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def mini(cls, **kwargs):
        """~65M params · 8 layers (6M+2G) · architecture validation (hours on 1 GPU)."""
        defaults = dict(
            vocab_size=32_000,
            d_model=384, n_layers=8, n_heads=6, d_head=64,
            d_kv_latent=256, d_q_latent=384, d_rope=32,
            ffn_hidden=1024, max_seq_len=1024,
            n_experts=4, n_active_experts=2,
            has_shared_expert=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def small(cls, **kwargs):
        """~280M params · 12 layers (9M+3G) · hyperparameter tuning (days on 1 GPU)."""
        defaults = dict(
            vocab_size=32_000,
            d_model=640, n_layers=12, n_heads=10, d_head=64,
            d_kv_latent=448, d_q_latent=640, d_rope=32,
            ffn_hidden=1792, max_seq_len=2048,
            n_experts=6, n_active_experts=2,
            has_shared_expert=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def medium(cls, **kwargs):
        """~900M params · 16 layers (12M+4G) · serious pretraining (days on 8 GPUs)."""
        defaults = dict(
            vocab_size=32_000,
            d_model=1024, n_layers=16, n_heads=16, d_head=64,
            d_kv_latent=512, d_q_latent=1024, d_rope=32,
            ffn_hidden=2560, max_seq_len=2048,
            n_experts=8, n_active_experts=2,
            has_shared_expert=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def full(cls, **kwargs):
        """~2B+ params · 24 layers · full-scale training run."""
        return cls(**kwargs)
