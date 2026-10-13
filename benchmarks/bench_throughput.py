#!/usr/bin/env python
"""Benchmark: prefill/decode throughput across model sizes.

作者 / Author: 晨星 (Chenxing)

Pure-NumPy CPU reference numbers (not a CUDA backend — see docs/LIMITATIONS.md).
Sweeps hidden sizes and prompt lengths, prints a table, and writes
benchmarks/throughput.json. Deterministic (temperature=0).

Usage:
  python benchmarks/bench_throughput.py
  python benchmarks/bench_throughput.py --hidden 64 128 256 --prompt-len 16 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model
from chenxing.core.sampler import SamplingParams
from chenxing.eval import generation_benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--new-tokens", type=int, default=32)
    args = ap.parse_args()

    rows = []
    for h in args.hidden:
        for pl in args.prompt_len:
            cfg = ModelConfig(vocab_size=264, hidden_size=h, intermediate_size=h * 2,
                              num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2)
            model, _ = build_tiny_model(cfg)
            eng = LLMEngine(model, None, block_size=8, n_blocks=256)
            rng = np.random.default_rng(0)
            prompts = [list(map(int, rng.integers(0, 256, pl))) for _ in range(args.prompts)]
            t0 = time.perf_counter()
            stats = generation_benchmark(eng, prompts, max_new_tokens=args.new_tokens)
            wall = time.perf_counter() - t0
            stats["hidden"] = h
            stats["prompt_len"] = pl
            stats["wall_s"] = round(wall, 4)
            rows.append(stats)
            print(f"h={h:3d} pl={pl:2d}  prefill={stats['prefill_tokens_per_s']:9.1f} tok/s  "
                  f"decode={stats['decode_tokens_per_s']:8.1f} tok/s  wall={wall:.2f}s")

    out = Path(__file__).resolve().parent / "throughput.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
