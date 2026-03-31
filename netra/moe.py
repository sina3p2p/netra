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

        # Flatten all (token, expert) assignments and sort by expert
        # for contiguous batched dispatch — no nested Python loops
        token_ids = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.n_active).reshape(-1)
        flat_expert_ids = top_indices.view(-1)
        flat_weights = top_weights.view(-1)

        sorted_expert_ids, sort_order = flat_expert_ids.sort()
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]

        # Single GPU→CPU sync per layer (unavoidable with Python expert loop)
        tokens_per_expert = torch.bincount(sorted_expert_ids, minlength=self.n_experts)
        split_sizes = tokens_per_expert.tolist()

        out = torch.zeros_like(flat_x)
        start = 0
        for e, count in enumerate(split_sizes):
            if count == 0:
                continue
            end = start + count
            tids = sorted_token_ids[start:end]
            w = sorted_weights[start:end].unsqueeze(-1)
            out.index_add_(0, tids, w * self.experts[e](flat_x[tids]))
            start = end

        if self.shared_expert is not None:
            out = out + self.shared_expert(flat_x)

        out = out.view(B, S, D)

        with torch.no_grad():
            self._tokens_per_expert = tokens_per_expert.float()

            if self.training:
                target_load = N * self.n_active / self.n_experts
                load_error = self._tokens_per_expert - target_load
                self.expert_bias -= self.bias_update_speed * load_error.sign()

        return out
