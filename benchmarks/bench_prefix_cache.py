#!/usr/bin/env python
"""Benchmark: radix-paged prefix-cache reuse (shared-prefix acceleration).

作者 / Author: 晨星 (Chenxing)

Measures how many tokens are served from the radix tree instead of recomputed
when many requests share a long system prefix — the SGLang RadixAttention
use case, unified with vLLM PagedAttention here. Uses the public
`kv.match_prefix` API to read the reusable prefix length per request, and
prints the reuse rate plus a per-request table.

Usage:
  python benchmarks/bench_prefix_cache.py
  python benchmarks/bench_prefix_cache.py --shared 64 --variants 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model
from chenxing.core.sampler import SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", type=int, default=48, help="shared prefix length")
    ap.add_argument("--variants", type=int, default=8, help="number of distinct tails")
    ap.add_argument("--tail", type=int, default=12, help="tail length per variant")
    ap.add_argument("--new-tokens", type=int, default=8)
    args = ap.parse_args()

    cfg = ModelConfig(vocab_size=264, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2)
    model, _ = build_tiny_model(cfg)
    eng = LLMEngine(model, None, block_size=8, n_blocks=512)
    p = SamplingParams(max_tokens=args.new_tokens, temperature=0.0, seed=0)

    rng = np.random.default_rng(0)
    base = list(map(int, rng.integers(0, 256, args.shared)))
    total_prefill = 0
    total_reused = 0
    per_req = []
    for v in range(args.variants):
        tail = list(map(int, rng.integers(0, 256, args.tail)))
        prompt = base + tail
        # how many of these tokens are already in the shared radix tree?
        reused = eng.kv.match_prefix(prompt).n_tokens
        eng.generate(prompt, p)
        total_prefill += len(prompt)
        total_reused += reused
        per_req.append((v, len(prompt), reused))
        print(f"variant {v}: prompt={len(prompt)} tokens, reused_from_cache={reused}")

    reuse_rate = total_reused / max(total_prefill, 1)
    print(f"\nshared prefix length : {args.shared}")
    print(f"variants             : {args.variants}")
    print(f"total prefill tokens : {total_prefill}")
    print(f"tokens reused        : {total_reused}")
    print(f"prefix reuse rate    : {reuse_rate*100:.1f}%")
    print(f"kv stats             : {eng.kv.stats()}")

    out = Path(__file__).resolve().parent / "prefix_cache.json"
    out.write_text(json.dumps({
        "shared_prefix": args.shared, "variants": args.variants,
        "total_prefill": total_prefill, "reused_tokens": total_reused,
        "reuse_rate": round(reuse_rate, 4), "per_request": per_req,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
