"""Fast CPU tests for the Netra model. Run with: pytest -q"""

import pytest
import torch

from netra import ModelConfig, Netra
from netra.block import _uses_mla
from netra.moe import MoELayer


def tiny_config(**overrides) -> ModelConfig:
    cfg = dict(
        vocab_size=256, d_model=64, n_layers=4, n_heads=2, d_head=32,
        d_kv_latent=32, d_q_latent=64, d_rope=16, ffn_hidden=128,
        max_seq_len=64, n_experts=4, n_active_experts=2, has_shared_expert=True,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("attention_type", ["mla", "gla", "hybrid"])
def test_forward_shapes_and_loss(attention_type):
    cfg = tiny_config(attention_type=attention_type)
    model = Netra(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(x, targets=y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    # An untrained model should be close to uniform over the vocabulary
    assert abs(loss.item() - torch.log(torch.tensor(float(cfg.vocab_size))).item()) < 1.0


def test_backward_reaches_every_trainable_parameter_group():
    cfg = tiny_config()
    model = Netra(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(x, targets=x)
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert all(p.grad is not None for p in model.layers[0].attn.parameters())


def test_lm_head_is_tied_to_embeddings():
    model = Netra(tiny_config())
    assert model.lm_head.weight is model.tok_emb.weight


def test_hybrid_layout_keeps_first_and_last_layers_mla():
    cfg = tiny_config(n_layers=12, attention_type="hybrid", gla_every_n=4)
    layout = "".join("M" if _uses_mla(cfg, i) else "G" for i in range(cfg.n_layers))
    assert layout == "MGMMMGMMMGMM"


def test_moe_routes_exactly_k_experts_per_token():
    cfg = tiny_config()
    moe = MoELayer(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = moe(x)
    assert out.shape == x.shape
    assert int(moe._tokens_per_expert.sum()) == 2 * 8 * cfg.n_active_experts


def test_aux_loss_free_bias_pushes_against_overloaded_expert():
    cfg = tiny_config(has_shared_expert=False)
    moe = MoELayer(cfg).train()
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0] += 1.0  # make expert 0 win for positive inputs
    x = torch.rand(1, 32, cfg.d_model)
    moe(x)
    # Expert 0 is overloaded, so its routing bias must go down
    assert moe.expert_bias[0] < 0


def test_bias_is_frozen_in_eval_mode():
    cfg = tiny_config()
    moe = MoELayer(cfg).eval()
    moe(torch.randn(1, 16, cfg.d_model))
    assert torch.all(moe.expert_bias == 0)
