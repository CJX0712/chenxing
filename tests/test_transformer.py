"""Transformer correctness: prefill ≡ incremental, KV rollback idempotent."""
import numpy as np

from chenxing.core.sampler import SamplingParams


def test_incremental_equals_full_prefill(tiny_engine, sample_prompt):
    seq = tiny_engine.create_sequence(sample_prompt, SamplingParams(max_tokens=1))
    lg_full = tiny_engine.prefill(seq).copy()

    seq2 = tiny_engine.create_sequence(sample_prompt[:7], SamplingParams(max_tokens=1))
    tiny_engine.prefill(seq2)
    lg_inc = tiny_engine.forward_ids(seq2, sample_prompt[7:], allow_unconfirmed=True)
    assert np.abs(lg_full - lg_inc[-1]).max() < 1e-4


def test_rollback_and_rewrite_is_idempotent(tiny_engine, sample_prompt):
    """Rolling n_tokens back and re-forwarding must reproduce the same logits."""
    seq = tiny_engine.create_sequence(sample_prompt, SamplingParams(max_tokens=1))
    lg1 = tiny_engine.prefill(seq).copy()
    seq.n_tokens = 5
    lg2 = tiny_engine.forward_ids(seq, sample_prompt[5:], allow_unconfirmed=True)
    assert np.abs(lg1 - lg2[-1]).max() < 1e-4


def test_greedy_generation_deterministic(tiny_engine, sample_prompt, greedy):
    a, ra = tiny_engine.generate(sample_prompt, greedy)
    b, rb = tiny_engine.generate(sample_prompt, greedy)
    assert a == b and ra == rb


def test_generation_respects_max_tokens(tiny_engine, sample_prompt):
    out, reason = tiny_engine.generate(
        sample_prompt, SamplingParams(max_tokens=5, temperature=0.0))
    assert len(out) == 5 and reason == "length"


def test_prefix_cache_speeds_second_call(tiny_engine, sample_prompt):
    tiny_engine.generate(sample_prompt, SamplingParams(max_tokens=4, temperature=0.0))
    hits_before = tiny_engine.stats.prefix_cache_hits
    tiny_engine.generate(sample_prompt, SamplingParams(max_tokens=4, temperature=0.0))
    assert tiny_engine.stats.prefix_cache_hits > hits_before


def test_seed_reproducibility(tiny_engine, sample_prompt):
    p = SamplingParams(max_tokens=6, temperature=1.2, top_p=0.9, seed=123)
    a, _ = tiny_engine.generate(sample_prompt, p)
    b, _ = tiny_engine.generate(sample_prompt, p)
    assert a == b


def test_empty_prompt_rejected(tiny_engine):
    import pytest
    with pytest.raises(ValueError):
        tiny_engine.create_sequence([], SamplingParams())
