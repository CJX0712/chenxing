#!/usr/bin/env python
"""Example: build the reference tiny model and run deterministic generation.

作者 / Author: 晨星 (Chenxing)

The bundled model is a *randomly initialized* architecture twin of Qwen2.5-0.5B
(see docs/LIMITATIONS.md) — it proves the engine/KV/scheduler are correct, not
that it speaks fluently. Swap in a real GGUF via examples/load_real_gguf.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.tokenizer import ByteTokenizer
from chenxing.model.transformer import build_tiny_model
from chenxing.core.sampler import SamplingParams


def main() -> None:
    np.random.seed(0)
    cfg = ModelConfig(vocab_size=264, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2)
    model, _ = build_tiny_model(cfg)
    tok = ByteTokenizer()
    engine = LLMEngine(model, tok, block_size=8, n_blocks=128)

    # token prompt (deterministic)
    prompt = list(map(int, np.random.default_rng(0).integers(0, 256, 10)))
    out, reason = engine.generate(prompt, SamplingParams(max_tokens=12, temperature=0.0, seed=0))
    print("prompt ids :", prompt)
    print("output ids :", out, f"({reason})")

    # same prompt again -> identical (greedy determinism invariant)
    out2, _ = engine.generate(prompt, SamplingParams(max_tokens=12, temperature=0.0, seed=0))
    print("deterministic:", out == out2)

    # decode with the byte-level fallback tokenizer
    print("decoded    :", repr(tok.decode(out)))

    # chat over the ChatML template
    text = engine.chat([{"role": "user", "content": "hello world"}],
                       SamplingParams(max_tokens=8, temperature=0.0, seed=1))
    print("chat       :", repr(text))

    print("kv stats   :", engine.kv.stats())


if __name__ == "__main__":
    main()
