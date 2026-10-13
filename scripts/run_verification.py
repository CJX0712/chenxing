#!/usr/bin/env python
"""One-click reproducible verification for the Chenxing (晨星) stack.

作者 / Author: 晨星 (Chenxing)

What it does (no pytest required — runs against source via sys.path):
  1. Builds the reference tiny engine and checks greedy determinism.
  2. Checks the radix-paged KV cache records prefix hits on repeated prompts.
  3. Checks speculative decoding is token-identical to target greedy (Leviathan'23).
  4. Checks GGUF F32 round-trip (write -> read -> equals).
  5. If `pytest` is importable, also runs the full invariant suite.

Usage:
  python scripts/run_verification.py            # all checks
  python scripts/run_verification.py --no-pytest # skip the pytest suite
  python scripts/run_verification.py --report    # write verification_report.md

Exit code 0 == all checks passed, 1 == at least one failed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model
from chenxing.core.sampler import SamplingParams


def _engine(vocab=264, layers=2, hidden=64):
    cfg = ModelConfig(vocab_size=vocab, hidden_size=hidden,
                      intermediate_size=hidden * 2, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2)
    model, _ = build_tiny_model(cfg)
    return LLMEngine(model, None, block_size=8, n_blocks=128)


def check_deterministic_greedy():
    eng = _engine()
    rng = __import__("numpy").random.default_rng(0)
    prompt = list(map(int, rng.integers(0, 256, 10)))
    p = SamplingParams(max_tokens=8, temperature=0.0, seed=0)
    a, _ = eng.generate(prompt, p)
    b, _ = eng.generate(prompt, p)
    ok = a == b
    return ok, f"len(a)={len(a)} identical={ok}"


def check_radix_prefix_hits():
    eng = _engine()
    rng = __import__("numpy").random.default_rng(1)
    prompt = list(map(int, rng.integers(0, 256, 12)))
    p = SamplingParams(max_tokens=6, temperature=0.0, seed=0)
    eng.generate(prompt, p)            # first: cold (prefix published to radix)
    eng.generate(prompt, p)            # second: shared prefix should be reused
    stats = eng.kv.stats()
    rate = stats.get("prefix_hit_rate", 0.0)
    ok = isinstance(rate, (int, float)) and rate > 0.0
    return ok, f"prefix_hit_rate={rate:.3f} cached_blocks={stats.get('cached_blocks')}"


def check_speculative_greedy_lossless():
    np = __import__("numpy")
    target_cfg = ModelConfig(vocab_size=264, hidden_size=96,
                             intermediate_size=192, num_hidden_layers=3,
                             num_attention_heads=4, num_key_value_heads=2)
    draft_cfg = target_cfg.scaled(0.5)
    tmodel, _ = build_tiny_model(target_cfg)
    dmodel, _ = build_tiny_model(draft_cfg)
    target = LLMEngine(tmodel, None, block_size=8, n_blocks=256)
    draft = LLMEngine(dmodel, None, block_size=8, n_blocks=256)
    from chenxing.engine.speculative import SpeculativeDecoder
    spec = SpeculativeDecoder(target, draft, num_speculative_tokens=4)
    rng = np.random.default_rng(2)
    prompt = list(map(int, rng.integers(0, 256, 10)))
    p = SamplingParams(max_tokens=12, temperature=0.0, seed=0)
    out_spec, _ = spec.generate(prompt, p)
    out_target, _ = target.generate(prompt, p)
    ok = out_spec == out_target
    return ok, f"spec_len={len(out_spec)} target_len={len(out_target)} identical={ok}"


def check_gguf_roundtrip():
    np = __import__("numpy")
    w = (np.random.default_rng(3).standard_normal((20, 8)).astype(np.float32))
    from chenxing.model.gguf import write_gguf, read_gguf
    tmp = REPO / "assets" / "_roundtrip_check.gguf"
    tmp.parent.mkdir(exist_ok=True)
    write_gguf(tmp, {"architecture": "chenxing"}, {"blk.0.weight": w})
    g = read_gguf(tmp)
    back, _ = g.read_tensor("blk.0.weight")
    ok = np.allclose(back, w, atol=1e-5)
    tmp.unlink(missing_ok=True)
    return ok, f"max_err={float(np.abs(back - w).max()):.2e}"


CHECKS = [
    ("deterministic greedy", check_deterministic_greedy),
    ("radix prefix reuse", check_radix_prefix_hits),
    ("speculative greedy lossless", check_speculative_greedy_lossless),
    ("GGUF F32 round-trip", check_gguf_roundtrip),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pytest", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="write verification_report.md")
    args = ap.parse_args()

    print("=" * 64)
    print("Chenxing (晨星) · reproducible verification")
    print("=" * 64)
    results = []
    all_ok = True
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"EXCEPTION: {exc!r}"
        all_ok &= ok
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:32s} — {detail}")

    pytest_ok = None
    if not args.no_pytest:
        try:
            import pytest  # noqa: F401
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-p", "no:cacheprovider",
                 "-q"], cwd=str(REPO), capture_output=True, text=True)
            pytest_ok = r.returncode == 0
            all_ok &= pytest_ok
            last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip().splitlines()[-1]
            results.append(("pytest invariant suite", pytest_ok, last))
            print(f"[{'PASS' if pytest_ok else 'FAIL'}] {'pytest invariant suite':32s} — {last}")
        except ImportError:
            results.append(("pytest invariant suite", None,
                            "pytest not installed (run scripts/install.py); "
                            "inline checks above are sufficient"))
            print("[SKIP] pytest invariant suite — pytest not installed")

    print("-" * 64)
    print("OVERALL:", "PASS ✅" if all_ok else "FAIL ❌")
    print("-" * 64)

    if args.report:
        lines = ["# Chenxing verification report", "", f"Overall: **{'PASS' if all_ok else 'FAIL'}**", ""]
        for name, ok, detail in results:
            mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
            lines.append(f"- [{mark}] {name} — {detail}")
        (REPO / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")
        print("wrote verification_report.md")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
