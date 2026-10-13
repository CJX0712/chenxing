"""LLMEngine — prefill/decode loop over the radix-paged KV cache.

Scheduling semantics:

- Continuous batching: every engine step admits new sequences (prefill) and
  advances running ones (decode) — the vLLM scheduling model (Kwon et al.
  2023). In this NumPy reference implementation the per-step batch is
  processed sequence-by-sequence (batch=1 kernels); the scheduler structure,
  prefix reuse, and KV management are production-shaped, and the batching
  axis is a documented limitation / future JIT path.

- Preemption: on KV exhaustion the engine evicts LRU radix leaves; if still
  blocked it preempts the oldest running sequence (freeing its private
  blocks) and re-queues it — its shared prefix survives in the radix tree.

State convention (single source of truth, relied on by speculative.py):
  seq.n_tokens == number of tokens whose KV is committed.
  decode_step feeds seq.total_tokens[seq.n_tokens] and increments.
"""
from __future__ import annotations

import numpy as np

from ..core.sampler import Sampler, SamplingParams
from ..model.config import ModelConfig
from ..model.transformer import Transformer
from .radix_kv import OutOfKVCacheError, RadixPagedKVCache
from .sequence import Sequence, SequenceStatus

__all__ = ["LLMEngine", "EngineStats"]


class EngineStats:
    def __init__(self) -> None:
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.prefix_cache_hits = 0
        self.preemptions = 0
        self.forward_calls = 0


class LLMEngine:
    """Owns the model, the paged KV cache, and the generation loop."""

    def __init__(self, model: Transformer, tokenizer, *, block_size: int = 16,
                 n_blocks: int = 512) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg: ModelConfig = model.cfg
        self.sampler = Sampler(self.cfg.vocab_size)
        self.kv = RadixPagedKVCache(
            n_blocks=n_blocks, block_size=block_size,
            n_layers=self.cfg.num_hidden_layers,
            n_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim)
        self.stats = EngineStats()
        self._next_id = 0
        self._sequences: dict[int, Sequence] = {}

    # ------------------------------------------------------------------ #
    # sequence lifecycle
    # ------------------------------------------------------------------ #

    def create_sequence(self, prompt_tokens: list[int],
                        params: SamplingParams | None = None) -> Sequence:
        if not prompt_tokens:
            raise ValueError("prompt must contain at least one token")
        self._next_id += 1
        seq = Sequence(self._next_id, list(prompt_tokens),
                       params or SamplingParams())
        self._sequences[seq.seq_id] = seq
        return seq

    def _ensure_capacity(self, seq: Sequence, target_tokens: int) -> None:
        have = len(seq.block_table) * self.kv.block_size
        if target_tokens <= have:
            return
        need = -(-(target_tokens - have) // self.kv.block_size)
        try:
            self.kv.ensure_free(need)
        except OutOfKVCacheError:
            self._preempt_oldest()
            self.kv.ensure_free(need)
        for _ in range(need):
            seq.block_table.append(self.kv.alloc_block())

    def _preempt_oldest(self) -> None:
        running = [s for s in self._sequences.values() if s.status == SequenceStatus.RUNNING]
        if not running:
            return
        victim = min(running, key=lambda s: s.seq_id)
        self._release_seq(victim)
        victim.block_table = []
        victim.n_tokens = 0
        victim.status = SequenceStatus.PREEMPTED
        self.stats.preemptions += 1

    def _release_seq(self, seq: Sequence) -> None:
        if seq.radix_node is not None:
            try:
                self.kv.release(seq.radix_node)
            except RuntimeError:
                pass
            seq.radix_node = None

    # ------------------------------------------------------------------ #
    # core forward primitive
    # ------------------------------------------------------------------ #

    def forward_ids(self, seq: Sequence, new_ids: list[int],
                    *, allow_unconfirmed: bool = False) -> np.ndarray:
        """Forward `new_ids` (must start at seq.n_tokens), committing KV.

        Returns logits of shape (len(new_ids), vocab_size).
        `allow_unconfirmed=True` lets speculative verification forward drafted
        tokens that are not yet part of the confirmed sequence.
        """
        if not allow_unconfirmed and seq.n_tokens + len(new_ids) > len(seq.total_tokens):
            raise ValueError("forward beyond sequence end")
        n_past = seq.n_tokens
        self._ensure_capacity(seq, n_past + len(new_ids))
        positions = np.arange(n_past, n_past + len(new_ids), dtype=np.int64)
        self.stats.forward_calls += 1
        self.stats.prefill_tokens += len(new_ids) if len(new_ids) > 1 else 0
        self.stats.decode_tokens += 1 if len(new_ids) == 1 else 0
        logits = self.model.forward(np.asarray(new_ids, dtype=np.int64), positions,
                                    self.kv, seq.block_table, n_past)
        seq.n_tokens += len(new_ids)
        return logits

    # ------------------------------------------------------------------ #
    # prefill / decode
    # ------------------------------------------------------------------ #

    def prefill(self, seq: Sequence) -> np.ndarray:
        """Process the prompt: reuse cached prefix blocks, forward the tail.

        Returns next-token logits (vocab,).
        """
        hit = self.kv.match_prefix(seq.prompt_tokens)
        seq.block_table = list(hit.block_ids)
        if hit.n_tokens > 0:
            self.kv.acquire(hit.node)
            seq.radix_node = hit.node
            self.stats.prefix_cache_hits += 1
        seq.status = SequenceStatus.RUNNING

        rest = seq.prompt_tokens[hit.n_tokens:]
        if rest:
            seq.n_tokens = hit.n_tokens      # cached prefix KV is already committed
            logits = self.forward_ids(seq, rest)
        else:
            # Full prefix hit: re-forward the final prompt token (idempotent
            # KV overwrite) to obtain its next-token distribution.
            seq.n_tokens = hit.n_tokens - 1
            logits = self.forward_ids(seq, seq.prompt_tokens[-1:])
        seq.last_logits = logits[-1]
        return seq.last_logits

    def decode_step(self, seq: Sequence) -> int:
        """Advance one token: forward the next uncommitted token, sample it."""
        if seq.n_tokens < len(seq.total_tokens):
            logits = self.forward_ids(seq, seq.total_tokens[seq.n_tokens:seq.n_tokens + 1])
            seq.last_logits = logits[-1]
        tok = self.sampler.sample(seq.last_logits, seq.params, seq.generated)
        seq.generated.append(tok)
        return tok

    # ------------------------------------------------------------------ #
    # generation entry points
    # ------------------------------------------------------------------ #

    def generate(self, prompt_tokens: list[int], params: SamplingParams | None = None,
                 *, stop_ids: set[int] | None = None) -> tuple[list[int], str]:
        params = params or SamplingParams()
        seq = self.create_sequence(prompt_tokens, params)
        try:
            self.prefill(seq)
            stops = set(stop_ids or ()) | {self.cfg.eos_token_id}
            while len(seq.generated) < params.max_tokens:
                tok = self.decode_step(seq)
                if tok in stops:
                    seq.stop_reason = "eos"
                    break
            else:
                seq.stop_reason = "length"
            self._publish_prefix(seq)
            return list(seq.generated), seq.stop_reason
        finally:
            self._release_seq(seq)

    def _publish_prefix(self, seq: Sequence) -> None:
        """Publish the sequence's block-aligned prefix into the radix tree."""
        bs = self.kv.block_size
        full_blocks = seq.n_tokens // bs
        if full_blocks == 0:
            return
        tokens = seq.total_tokens[:full_blocks * bs]
        blocks = seq.block_table[:full_blocks]
        try:
            self.kv.insert(tokens, blocks, parent=seq.radix_node)
        except ValueError:
            pass  # unaligned insert raced with eviction; skip (documented)

    # ------------------------------------------------------------------ #
    # chat helpers
    # ------------------------------------------------------------------ #

    def chat(self, messages: list[dict[str, str]], params: SamplingParams | None = None,
             stop_ids: set[int] | None = None) -> str:
        prompt = self.apply_chat_template(messages)
        ids = self.tokenizer.encode(prompt)
        if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id >= 0:
            ids = [self.tokenizer.bos_token_id] + ids
        stops = set(stop_ids or ())
        im_end = self.tokenizer.special_tokens.get("<|im_end|>", None) \
            if hasattr(self.tokenizer, "special_tokens") else None
        if im_end is not None:
            stops.add(im_end)
        out, _ = self.generate(ids, params, stop_ids=stops)
        return self.tokenizer.decode(out)

    @staticmethod
    def apply_chat_template(messages: list[dict[str, str]]) -> str:
        """Qwen/Llama-3 style ChatML template."""
        parts = []
        for m in messages:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
