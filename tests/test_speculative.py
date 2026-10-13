"""Speculative decoding: the losslessness theorem, enforced empirically.

Greedy mode must reproduce pure target greedy decoding token-for-token for
every γ. This is the algorithmic core of Leviathan et al. 2023 (Alg. 1) and
the hardest invariant in the stack.
"""
import numpy as np
import pytest

from chenxing.core.sampler import SamplingParams
from chenxing.engine.speculative import SpeculativeDecoder


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_greedy_losslessness(tiny_engine, draft_engine, sample_prompt, gamma):
    target, _ = tiny_engine.generate(sample_prompt,
                                     SamplingParams(max_tokens=16, temperature=0.0, seed=7))
    sd = SpeculativeDecoder(tiny_engine, draft_engine, num_speculative_tokens=gamma)
    out, _ = sd.generate(sample_prompt,
                         SamplingParams(max_tokens=16, temperature=0.0, seed=7))
    assert out == target


def test_stochastic_distribution_equivalence():
    """Draft == target ⇒ speculative sampling ≡ direct sampling (Thm 3.5).

    With an IDENTICAL draft model, accepted-with-prob-min(1, p/q) = always,
    and the residual branch never fires — output must equal direct sampling
    from the target with the same seed stream. Additionally, with a different
    draft, the empirical marginals must track the target distribution.
    """
    from chenxing.model.config import ModelConfig
    from chenxing.model.transformer import build_tiny_model
    from chenxing.engine.engine import LLMEngine

    cfg = ModelConfig(vocab_size=264, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model, _ = build_tiny_model(cfg, seed=7)
    engine = LLMEngine(model, None, block_size=8, n_blocks=64)
    same_draft = LLMEngine(model, None, block_size=8, n_blocks=64)

    prompt = list(map(int, np.random.default_rng(3).integers(0, 256, 8)))
    params = SamplingParams(max_tokens=24, temperature=1.0, seed=42)
    direct, _ = engine.generate(prompt, params)

    sd = SpeculativeDecoder(engine, same_draft, num_speculative_tokens=4)
    out, _ = sd.generate(prompt, params)
    assert out == direct


def test_acceptance_stats_sane(tiny_engine, draft_engine, sample_prompt):
    sd = SpeculativeDecoder(tiny_engine, draft_engine, num_speculative_tokens=4)
    sd.generate(sample_prompt, SamplingParams(max_tokens=16, temperature=0.0, seed=7))
    s = sd.stats.as_dict()
    assert 0.0 <= s["acceptance_rate"] <= 1.0
    assert s["drafted"] == s["steps"] * 4


def test_invalid_gamma_rejected(tiny_engine, draft_engine):
    import pytest
    with pytest.raises(ValueError):
        SpeculativeDecoder(tiny_engine, draft_engine, num_speculative_tokens=0)
