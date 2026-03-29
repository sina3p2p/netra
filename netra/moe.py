import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLUFFN(nn.Module):
    """Feed-forward network with SwiGLU gating (DeepSeek / LLaMA style)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.w_down = nn.Linear(config.ffn_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts with auxiliary-loss-free load balancing (DeepSeek V3).

    Each expert carries a bias term used ONLY for top-k gating decisions
    (not for computing combination weights). The bias is adjusted at each
    forward pass: overloaded experts get their bias decreased, underloaded
    experts get it increased.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.n_active = config.n_active_experts
        self.bias_update_speed = config.bias_update_speed

        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(config.n_experts))

        self.experts = nn.ModuleList(
            [SwiGLUFFN(config) for _ in range(config.n_experts)]
        )
        self.shared_expert = SwiGLUFFN(config) if config.has_shared_expert else None

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        flat_x = x.view(-1, D)
        N = flat_x.shape[0]

        router_logits = self.gate(flat_x)

        # Combination weights from raw logits (no bias)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection uses logits + bias (bias steers routing, not weights)
        biased_logits = router_logits + self.expert_bias
        _, top_indices = torch.topk(biased_logits, self.n_active, dim=-1)

        top_weights = router_probs.gather(dim=-1, index=top_indices)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Dispatch tokens to experts
        out = torch.zeros_like(flat_x)
        for i in range(self.n_active):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i]

            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_out = self.experts[e](flat_x[mask])
                    out[mask] += weight[mask].unsqueeze(-1) * expert_out

        if self.shared_expert is not None:
            out = out + self.shared_expert(flat_x)

        out = out.view(B, S, D)

        with torch.no_grad():
            tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
            for i in range(self.n_active):
                tokens_per_expert.scatter_add_(
                    0, top_indices[:, i], torch.ones(N, device=x.device)
                )
            self._tokens_per_expert = tokens_per_expert

            if self.training:
                target_load = N * self.n_active / self.n_experts
                load_error = tokens_per_expert - target_load
                self.expert_bias -= self.bias_update_speed * load_error.sign()

        return out
