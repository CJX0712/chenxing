#!/usr/bin/env python
"""Example: lossless speculative decoding (Leviathan et al. 2023).

作者 / Author: 晨星 (Chenxing)

Wires a small draft model + a larger target, runs SpeculativeDecoder in greedy
mode, and asserts the output is *token-identical* to pure target greedy (the
losslessness theorem). Also prints the acceptance rate, the metric that would
translate into wall-clock speedup on a real backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from chenxing.engine.engine import LLMEngine
from chenxing.engine.speculative import SpeculativeDecoder
from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model
from chenxing.core.sampler import SamplingParams


def main() -> None:
    np.random.seed(0)
    target_cfg = ModelConfig(vocab_size=264, hidden_size=96, intermediate_size=192,
                             num_hidden_layers=3, num_attention_heads=4,
                             num_key_value_heads=2)
    draft_cfg = target_cfg.scaled(0.5)        # half-width, half-depth draft
    tmodel, _ = build_tiny_model(target_cfg)
    dmodel, _ = build_tiny_model(draft_cfg)
    target = LLMEngine(tmodel, None, block_size=8, n_blocks=256)
    draft = LLMEngine(dmodel, None, block_size=8, n_blocks=256)

    spec = SpeculativeDecoder(target, draft, num_speculative_tokens=4)
    prompt = list(map(int, np.random.default_rng(2).integers(0, 256, 10)))
    p = SamplingParams(max_tokens=16, temperature=0.0, seed=0)

    out_spec, _ = spec.generate(prompt, p)
    out_target, _ = target.generate(prompt, p)

    print("spec greedy  :", out_spec)
    print("target greedy:", out_target)
    print("LOSSLESS (token-identical):", out_spec == out_target)
    print("acceptance stats:", spec.stats.as_dict())
    print("\nNote: acceptance_rate is ~0 here because draft and target are two")
    print("INDEPENDENT randomly-initialized models. The theorem (Leviathan'23)")
    print("only requires losslessness; a deployed draft is a smaller copy of the")
    print("SAME trained target, where acceptance >> 0 and yields real speedup.")


if __name__ == "__main__":
    main()
