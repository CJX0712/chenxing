"""Sequence state for the scheduling engine."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.sampler import SamplingParams
from .radix_kv import _RadixNode

__all__ = ["Sequence", "SequenceStatus"]


class SequenceStatus:
    WAITING = "waiting"          # queued, not yet prefilled
    RUNNING = "running"          # prefilled, decoding
    FINISHED = "finished"
    PREEMPTED = "preempted"      # evicted mid-decode; will be recomputed


@dataclass
class Sequence:
    seq_id: int
    prompt_tokens: list[int]
    params: SamplingParams
    status: str = SequenceStatus.WAITING
    generated: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)     # physical KV blocks
    n_tokens: int = 0                                         # processed prompt+gen
    radix_node: _RadixNode | None = None                      # node pinning prefix KV
    last_logits: np.ndarray | None = None
    output_text: str = ""
    stop_reason: str = ""

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def total_tokens(self) -> list[int]:
        return self.prompt_tokens + self.generated

    def is_finished(self, eos_id: int) -> bool:
        if self.generated and self.generated[-1] == eos_id:
            self.stop_reason = "eos"
            return True
        if len(self.generated) >= self.params.max_tokens:
            self.stop_reason = "length"
            return True
        return False
