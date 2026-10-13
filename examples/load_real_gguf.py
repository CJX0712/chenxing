#!/usr/bin/env python
"""Example: load a REAL GGUF checkpoint + tokenizer and chat.

作者 / Author: 晨星 (Chenxing)

The bundled demo models are untrained (docs/LIMITATIONS.md). To run real text,
export a GGUF from llama.cpp / HF and point this script at it:

    export CHENXING_GGUF=/path/to/model.Q4_K_M.gguf
    python examples/load_real_gguf.py

Supports F32/F16/Q8_0/Q2_K/Q4_K/Q6_K tensors (exact GGUF k-quant dequant).
If no GGUF is provided, the script explains what to do instead of crashing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.gguf import read_gguf
from chenxing.model.tokenizer import BPETokenizer
from chenxing.core.sampler import SamplingParams


def _config_from_gguf(g):
    # map common llama2/qwen GGUF metadata onto ModelConfig
    get = lambda k, d=None: g.get(k, d)
    return ModelConfig(
        vocab_size=int(get("general.alignment", 32)) and get("tokenizer.ggml.tokens")
                    and len(g.get("tokenizer.ggml.tokens")),
        hidden_size=int(get("llama.embedding_length", 896)),
        intermediate_size=int(get("llama.feed_forward_length", 4864)),
        num_hidden_layers=int(get("llama.block_count", 24)),
        num_attention_heads=int(get("llama.attention.head_count", 14)),
        num_key_value_heads=int(get("llama.attention.head_count_kv", 2)),
        head_dim=int(get("llama.attention.layer_norm_rms_epsilon") is not None and 64),
        rms_norm_eps=float(get("llama.attention.layer_norm_rms_epsilon", 1e-6)),
        rope_theta=float(get("llama.rope.freq_base", 1_000_000.0)),
        max_position_embeddings=int(get("llama.context_length", 32768)),
    )


def main() -> None:
    path = os.environ.get("CHENXING_GGUF") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not path or not Path(path).exists():
        print("No GGUF provided. This example needs a real checkpoint to produce text.")
        print("  1) Obtain a GGUF (e.g. Qwen2.5-0.5B-Instruct-Q4_K_M.gguf).")
        print("  2) export CHENXING_GGUF=/path/to/model.gguf")
        print("  3) python examples/load_real_gguf.py")
        print("\nThe bundled, untrained demo still runs end-to-end via:")
        print("  python examples/basic_generate.py")
        return

    g = read_gguf(path)
    print(f"loaded GGUF: {Path(path).name}  tensors={len(g.tensors)}  meta={len(g.metadata)}")

    cfg = _config_from_gguf(g)
    weights = {}
    for name in g.tensors:
        if name.startswith("token_embd") or name.startswith("blk.") or name in (
                "output_norm.weight", "output.weight"):
            data, _ = g.read_tensor(name)
            weights[name] = data
    # build transformer from loaded weights (weight naming follows GGUF convention)
    from chenxing.model.transformer import Transformer
    model = Transformer(cfg, weights)

    tok = None
    if g.get("tokenizer.ggml.tokens"):
        tok = BPETokenizer.from_gguf(g)

    engine = LLMEngine(model, tok, block_size=16, n_blocks=1024)
    text = engine.chat([{"role": "user", "content": "Hello, who are you?"}],
                       SamplingParams(max_tokens=64, temperature=0.7, seed=0))
    print("assistant:", text)


if __name__ == "__main__":
    main()
