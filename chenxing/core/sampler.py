"""Token sampling.

Provenance:

- Temperature/top-k/top-p (nucleus) : Holtzman et al., "The Curious Case of
  Neural Text Degeneration", ICLR 2020 (arXiv:1904.09751).
- min-p                            : Nguyen, "Min_p Sampling", 2024
  (arXiv:2407.01082) — dynamic floor relative to the top token.
- Typical (locally typical) sampling: Meister & Cotterell, 2023
  (arXiv:2202.00666).
- Repetition penalty               : CTRL paper style (Keskar et al. 2019) as
  adopted by HF transformers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Sampler", "SamplingParams"]


@dataclass
class SamplingParams:
    """Per-request sampling configuration (OpenAI-compatible superset)."""
    temperature: float = 1.0
    top_k: int = 0                 # 0 = off
    top_p: float = 1.0             # 1.0 = off
    min_p: float = 0.0
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    max_tokens: int = 64
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    n_keep: int = 0                # prompt length, for repetition penalty scope

    def __post_init__(self) -> None:
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")


class Sampler:
    """Stateless-per-call logits processor + sampler.

    Greedy (temperature == 0) picks argmax. Stochastic path applies, in the
    canonical order: repetition penalty → temperature → top-k → top-p →
    min-p → typical-p, then draws from the renormalized distribution.
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self._rng = np.random.default_rng()

    def _rng_for(self, seed: int | None) -> np.random.Generator:
        if seed is None:
            return self._rng
        return np.random.default_rng(seed)

    def apply_penalties(self, logits: np.ndarray, params: SamplingParams,
                        generated_ids: list[int]) -> np.ndarray:
        logits = logits.astype(np.float64, copy=True)
        if params.repetition_penalty != 1.0 and generated_ids:
            ctx = set(generated_ids[-max(64, params.n_keep):]) if params.n_keep else set(generated_ids)
            idx = np.fromiter((t for t in ctx if 0 <= t < self.vocab_size), dtype=np.int64)
            if idx.size:
                vals = logits[idx]
                logits[idx] = np.where(vals > 0, vals / params.repetition_penalty,
                                       vals * params.repetition_penalty)
        return logits

    def _truncate(self, probs: np.ndarray, params: SamplingParams) -> np.ndarray:
        if params.top_k > 0:
            k = min(params.top_k, probs.size)
            thresh = np.partition(probs, -k)[-k]
            probs = np.where(probs >= thresh, probs, 0.0)

        if params.top_p < 1.0:
            order = np.argsort(-probs)
            sorted_p = probs[order]
            cum = np.cumsum(sorted_p)
            cutoff = np.searchsorted(cum, params.top_p) + 1
            keep = order[:cutoff]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)

        if params.min_p > 0.0:
            top = probs.max()
            probs = np.where(probs >= params.min_p * top, probs, 0.0)

        if params.typical_p < 1.0:
            logp = np.log(np.maximum(probs, 1e-38))
            entropy = -(probs * logp).sum()
            shift = np.abs(logp + entropy)
            order = np.argsort(shift)
            cum = np.cumsum(probs[order])
            cutoff = np.searchsorted(cum, params.typical_p) + 1
            keep = order[:cutoff]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
        return probs

    def sample(self, logits: np.ndarray, params: SamplingParams,
               generated_ids: list[int] | None = None) -> int:
        """Draw one token id from `logits` (1-D array of vocab size)."""
        generated_ids = generated_ids or []
        logits = self.apply_penalties(logits, params, generated_ids)

        if params.temperature == 0.0:
            return int(np.argmax(logits))

        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if params.temperature != 1.0:
            probs = probs ** (1.0 / params.temperature)
            probs /= probs.sum()
        probs = self._truncate(probs, params)
        total = probs.sum()
        if total <= 0:
            return int(np.argmax(logits))
        probs /= total
        return int(self._rng_for(params.seed).choice(self.vocab_size, p=probs))
