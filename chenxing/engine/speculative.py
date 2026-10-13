"""Speculative decoding — distribution-preserving fast decoding.

Provenance:

- Algorithm & losslessness theorem : Leviathan et al., "Fast Inference from
  Transformers via Speculative Decoding", ICML 2023 (arXiv:2211.17192),
  Theorem 3.5 — and simultaneously Chen et al., "Accelerating Large Language
  Model Decoding with Speculative Sampling", 2023 (arXiv:2302.01318).
- Verification-by-single-forward   : the chain special case of SpecInfer's
  tree verification (Miao et al., ASPLOS 2024, arXiv:2305.09781).

Losslessness guarantee (enforced by tests):

  Greedy mode: a drafted token is accepted iff it equals the target's argmax
  at that position → output is token-identical to pure target greedy decoding.

  Stochastic mode: candidate x from draft distribution q is accepted with
  probability min(1, p(x)/q(x)); on rejection the replacement is drawn from
  the residual norm(max(0, p−q)). Output distribution == target distribution
  exactly (Leviathan Thm 3.5) — reproduced empirically by the
  distribution-equivalence test in tests/test_speculative.py.

Rollback semantics: rejection truncates by rolling back `n_tokens` metadata
only — PagedAttention turns KV rollback into O(1) bookkeeping; stale slots
are overwritten by the next forward pass.

Round invariant (per drafted chain):
  At loop entry, tseq.n_tokens == dseq.n_tokens == len(confirmed) − 1:
  every confirmed token's KV is committed except the last, whose next-token
  distribution the round derives by forwarding it first.
"""
from __future__ import annotations

import numpy as np

from ..core.sampler import SamplingParams
from .engine import LLMEngine
from .sequence import Sequence, SequenceStatus

__all__ = ["SpeculativeDecoder", "SpeculativeStats"]


class SpeculativeStats:
    def __init__(self) -> None:
        self.drafted = 0
        self.accepted = 0
        self.steps = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.drafted, 1)

    @property
    def mean_accepted_per_step(self) -> float:
        return self.accepted / max(self.steps, 1)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "mean_accepted_per_step": round(self.mean_accepted_per_step, 3),
        }


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x.astype(np.float64) - x.max())
    return e / e.sum()


class SpeculativeDecoder:
    """Draft γ tokens with a small model; verify with the target in one pass."""

    def __init__(self, engine: LLMEngine, draft_engine: LLMEngine,
                 *, num_speculative_tokens: int = 4) -> None:
        if num_speculative_tokens < 1:
            raise ValueError("need at least 1 speculative token")
        self.engine = engine
        self.draft = draft_engine
        self.k_draft = num_speculative_tokens
        self.stats = SpeculativeStats()

    # ------------------------------------------------------------------ #

    def _sync_to(self, seq: Sequence, eng: LLMEngine, n_target: int) -> None:
        """Bring a sequence's committed KV length to exactly `n_target`."""
        if seq.n_tokens > n_target:
            seq.n_tokens = n_target                      # metadata-only rollback
        elif seq.n_tokens < n_target:
            eng.forward_ids(seq, seq.total_tokens[seq.n_tokens:n_target],
                            allow_unconfirmed=True)

    def generate(self, prompt_tokens: list[int],
                 params: SamplingParams | None = None,
                 *, stop_ids: set[int] | None = None) -> tuple[list[int], str]:
        params = params or SamplingParams()
        eos = self.engine.cfg.eos_token_id
        stops = set(stop_ids or ()) | {eos}

        tseq = self.engine.create_sequence(prompt_tokens, params)
        dseq = self.draft.create_sequence(prompt_tokens, params)
        try:
            self.engine.prefill(tseq)
            self.draft.prefill(dseq)
            confirmed = list(prompt_tokens)
            # Round invariant: last token not yet forwarded.
            tseq.n_tokens -= 1
            dseq.n_tokens -= 1

            rng = np.random.default_rng(params.seed)
            greedy = params.temperature == 0.0

            while len(confirmed) - len(prompt_tokens) < params.max_tokens:
                n_start = len(confirmed) - 1

                # ---- 1. draft: propose γ tokens autoregressively ----
                self._sync_to(dseq, self.draft, n_start)
                q_rows: list[np.ndarray] = []
                proposed: list[int] = []
                logits_d = self.draft.forward_ids(dseq, [confirmed[n_start]],
                                                  allow_unconfirmed=True)
                q_rows.append(logits_d[-1])
                for i in range(self.k_draft):
                    tok = int(np.argmax(q_rows[-1])) if greedy else \
                        self.draft.sampler.sample(q_rows[-1], params, confirmed)
                    proposed.append(tok)
                    self.stats.drafted += 1
                    if i < self.k_draft - 1:
                        logits_d = self.draft.forward_ids(dseq, [tok],
                                                          allow_unconfirmed=True)
                        q_rows.append(logits_d[-1])

                # ---- 2. target: verify with one forward over the whole chain ----
                # We forward the FULL chain (seed token + all γ draft tokens) so
                # logits_t has γ+1 rows: rows [0..γ-1] verify proposed[0..γ-1],
                # and the final row logits_t[γ] is the target's distribution for
                # the token AFTER the chain — used as the bonus token when every
                # draft token is accepted (Leviathan 2023, Alg. 1, line 8).
                self._sync_to(tseq, self.engine, n_start)
                verify_in = [confirmed[n_start]] + proposed
                logits_t = self.engine.forward_ids(tseq, verify_in,
                                                   allow_unconfirmed=True)   # (γ+1, V)

                # ---- 3. verification (Leviathan 2023, Alg. 1) ----
                accepted_n = 0
                next_tok: int | None = None
                for i, tok in enumerate(proposed):
                    row = logits_t[i]
                    if greedy:
                        if tok == int(np.argmax(row)):
                            accepted_n += 1
                            continue
                        next_tok = int(np.argmax(row))
                        break
                    p = _softmax(row / params.temperature)
                    q = _softmax(q_rows[i] / params.temperature)
                    if rng.random() < min(1.0, p[tok] / max(q[tok], 1e-12)):
                        accepted_n += 1
                        continue
                    residual = np.maximum(p - q, 0.0)
                    total = residual.sum()
                    if total <= 0:
                        next_tok = int(np.argmax(row))
                    else:
                        next_tok = int(rng.choice(residual.size, p=residual / total))
                    break
                else:
                    last_row = logits_t[-1]
                    next_tok = int(np.argmax(last_row)) if greedy else \
                        self.engine.sampler.sample(last_row, params, confirmed)

                self.stats.accepted += accepted_n
                self.stats.steps += 1

                # ---- 4. commit: confirmed' = confirmed + accepted [+ next] ----
                confirmed.extend(proposed[:accepted_n])
                if next_tok is not None:
                    confirmed.append(next_tok)
                if accepted_n < self.k_draft:
                    # Partial acceptance: roll back draft+target beyond len+k.
                    target_len = len(confirmed) - 1
                    self._sync_to(dseq, self.draft, target_len)
                    self._sync_to(tseq, self.engine, target_len)

                # keep Sequence views consistent
                tseq.generated = confirmed[len(prompt_tokens):]
                dseq.generated = tseq.generated
                tseq.last_logits = logits_t[-1] if len(logits_t) else None

                if any(t in stops for t in confirmed[len(prompt_tokens):]):
                    break

            out = confirmed[len(prompt_tokens):][:params.max_tokens]
            reason = "eos" if any(t == eos for t in out) else "length"
            self.engine._publish_prefix(tseq)
            return out, reason
        finally:
            self.engine._release_seq(tseq)
            self.draft._release_seq(dseq)
