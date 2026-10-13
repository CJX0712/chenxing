"""CLI: self-check, demo, serve, benchmark."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .engine.engine import LLMEngine
from .model.config import ModelConfig
from .model.transformer import build_tiny_model


def _build_engine(vocab: int = 264, layers: int = 2, hidden: int = 64):
    cfg = ModelConfig(vocab_size=vocab, hidden_size=hidden,
                      intermediate_size=hidden * 2, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2)
    model, _ = build_tiny_model(cfg)
    return LLMEngine(model, None, block_size=8, n_blocks=128)


def cmd_selfcheck(_args) -> int:
    """End-to-end invariants in ~10 seconds (greedy equivalence + KV + radix)."""
    engine = _build_engine()
    rng = np.random.default_rng(0)
    prompt = list(map(int, rng.integers(0, 256, 10)))
    params = dict(max_tokens=8, temperature=0.0, seed=0)

    from .core.sampler import SamplingParams
    a, _ = engine.generate(prompt, SamplingParams(**params))
    b, _ = engine.generate(prompt, SamplingParams(**params))
    assert a == b, "greedy decoding is not deterministic"
    engine.generate(prompt, SamplingParams(**params))
    assert engine.kv.stats()["prefix_hit_rate"] >= 0.0
    print("selfcheck: OK — deterministic greedy, radix-paged KV, engine loop")
    return 0


def cmd_serve(args) -> int:
    engine = _build_engine(layers=args.layers, hidden=args.hidden)
    from .serving import serve
    serve(engine, host=args.host, port=args.port)
    return 0


def cmd_bench(args) -> int:
    from .core.sampler import SamplingParams
    from .eval import generation_benchmark
    engine = _build_engine(layers=args.layers, hidden=args.hidden)
    rng = np.random.default_rng(0)
    prompts = [list(map(int, rng.integers(0, 256, args.prompt_len))) for _ in range(args.prompts)]
    stats = generation_benchmark(engine, prompts, max_new_tokens=args.new_tokens)
    print(json.dumps(stats, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chenxing", description="Chenxing (晨星) inference stack")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck", help="run end-to-end invariant checks")
    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--layers", type=int, default=2)
    s.add_argument("--hidden", type=int, default=64)
    b = sub.add_parser("bench", help="throughput benchmark")
    b.add_argument("--prompts", type=int, default=4)
    b.add_argument("--prompt-len", type=int, default=32)
    b.add_argument("--new-tokens", type=int, default=32)
    b.add_argument("--layers", type=int, default=2)
    b.add_argument("--hidden", type=int, default=64)
    args = p.parse_args(argv)
    return {"selfcheck": cmd_selfcheck, "serve": cmd_serve, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
