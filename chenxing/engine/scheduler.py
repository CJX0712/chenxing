"""Continuous batching scheduler (vLLM-style, Kwon et al. SOSP 2023).

Admits waiting sequences while running ones decode — new requests join the
in-flight batch without waiting for it to drain. The scheduler owns priority
(FIFO), admission control (KV capacity), and preemption.

In this NumPy reference the per-step execution is sequential over sequences;
the scheduler itself (admission, preemption, bookkeeping) is what's tested
and is engine-agnostic.
"""
from __future__ import annotations

from collections import deque

from .engine import LLMEngine
from .radix_kv import OutOfKVCacheError
from .sequence import Sequence, SequenceStatus

__all__ = ["ContinuousBatchingScheduler"]


class ContinuousBatchingScheduler:
    """FIFO continuous batching over an LLMEngine."""

    def __init__(self, engine: LLMEngine, *, max_running: int = 8) -> None:
        self.engine = engine
        self.max_running = max_running
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self._finished: list[Sequence] = []

    def submit(self, seq: Sequence) -> None:
        self._waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self._waiting or self._running)

    def step(self) -> list[Sequence]:
        """One scheduler iteration: admit, then advance one token each.

        Returns sequences that finished during this step.
        """
        # admission
        while self._waiting and len(self._running) < self.max_running:
            seq = self._waiting.popleft()
            try:
                self.engine.prefill(seq)
                self._running.append(seq)
            except OutOfKVCacheError:
                self.engine.kv.evict(1)
                self._waiting.appendleft(seq)
                break

        # decode one token per running sequence
        eos = self.engine.cfg.eos_token_id
        finished_now: list[Sequence] = []
        still_running: list[Sequence] = []
        for seq in self._running:
            try:
                tok = self.engine.decode_step(seq)
                done = tok == eos or len(seq.generated) >= seq.params.max_tokens
                if done:
                    seq.stop_reason = "eos" if tok == eos else "length"
                    seq.status = SequenceStatus.FINISHED
                    self.engine._publish_prefix(seq)
                    self.engine._release_seq(seq)
                    finished_now.append(seq)
                else:
                    still_running.append(seq)
            except OutOfKVCacheError:
                # preempt the longest-running other sequence and retry once
                self.engine._preempt_oldest()
                still_running.append(seq)
        self._running = [s for s in still_running if s.status == SequenceStatus.RUNNING]
        self._finished.extend(finished_now)
        return finished_now

    def run_until_drained(self, on_token=None) -> list[Sequence]:
        """Drive the scheduler until all submitted sequences complete."""
        while self.has_work:
            done = self.step()
            if on_token is not None:
                for seq in done:
                    on_token(seq)
        return list(self._finished)
