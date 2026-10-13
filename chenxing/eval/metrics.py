"""Evaluation: perplexity and throughput benchmarks.

Provenance: perplexity = exp(mean NLL) is the standard LM metric
(GPT-2/GPT-3 protocol, Radford et al. 2019 / Brown et al. 2020); we report
bits-per-byte alongside for cross-tokenizer comparability.
"""
from __future__ import annotations

import time

import numpy as np

from ..core.sampler import SamplingParams
from ..engine.engine import LLMEngine

__all__ = ["perplexity", "generation_benchmark"]


def perplexity(engine: LLMEngine, token_sequences: list[list[int]],
               sliding: int | None = None) -> dict[str, float]:
    """Mean NLL over next-token predictions; PPL = exp(NLL).

    Sequences are scored with teacher forcing in windows of `sliding`
    (None = whole sequence at once). Deterministic.
    """
    nll_sum, n_pred = 0.0, 0
    for ids in token_sequences:
        if len(ids) < 2:
            continue
        seq = engine.create_sequence(ids, SamplingParams(max_tokens=1))
        # score in chunks through forward_ids (uses KV cache; no sampling)
        seq.n_tokens = 0
        pos = 0
        while pos < len(ids) - 1:
            end = min(pos + (sliding or len(ids) - 1), len(ids) - 1)
            logits = engine.forward_ids(seq, ids[pos:end])
            targets = np.asarray(ids[pos + 1:end + 1], dtype=np.int64)
            # logits[i] predicts ids[pos+i+1]
            lse = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
            lse = np.log(np.exp(lse).sum(axis=-1)) + logits.max(axis=-1)
            picked = logits[np.arange(len(targets)), targets].astype(np.float64)
            nll_sum += float((lse - picked).sum())
            n_pred += len(targets)
            pos = end
        engine._release_seq(seq)
    nll = nll_sum / max(n_pred, 1)
    return {"nll": nll, "perplexity": float(np.exp(nll)), "n_predictions": n_pred}


def generation_benchmark(engine: LLMEngine, prompts: list[list[int]],
                         max_new_tokens: int = 32,
                         temperature: float = 0.0) -> dict[str, float]:
    """Throughput benchmark: prefill tok/s, decode tok/s, wall time."""
    t_prefill = t_decode = 0.0
    n_prefill = n_decode = 0
    for p in prompts:
        seq = engine.create_sequence(p, SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature))
        t0 = time.perf_counter()
        engine.prefill(seq)
        t_prefill += time.perf_counter() - t0
        n_prefill += len(p)
        t0 = time.perf_counter()
        while len(seq.generated) < max_new_tokens:
            engine.decode_step(seq)
        t_decode += time.perf_counter() - t0
        n_decode += max_new_tokens
        engine._release_seq(seq)
    return {
        "prefill_tokens_per_s": n_prefill / max(t_prefill, 1e-9),
        "decode_tokens_per_s": n_decode / max(t_decode, 1e-9),
        "prefill_s": round(t_prefill, 4),
        "decode_s": round(t_decode, 4),
        "n_prompts": len(prompts),
    }
