"""Evaluation metrics sanity."""
import numpy as np

from chenxing.core.sampler import SamplingParams
from chenxing.eval import generation_benchmark, perplexity


def test_perplexity_finite_and_positive(tiny_engine, sample_prompt):
    res = perplexity(tiny_engine, [sample_prompt, sample_prompt + [1, 2, 3]])
    assert res["n_predictions"] > 0
    assert np.isfinite(res["perplexity"]) and res["perplexity"] > 1.0


def test_benchmark_returns_rates(tiny_engine, sample_prompt):
    stats = generation_benchmark(tiny_engine, [sample_prompt], max_new_tokens=4)
    assert stats["prefill_tokens_per_s"] > 0
    assert stats["decode_tokens_per_s"] > 0
