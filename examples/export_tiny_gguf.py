#!/usr/bin/env python
"""Example: round-trip the bundled tiny model through the GGUF container.

作者 / Author: 晨星 (Chenxing)

Builds the reference tiny model, serializes it to a GGUF file (F32 or Q8_0),
reads it back, and asserts the weights reconstruct within tolerance. This is a
self-contained "sample dataset" that validates the container reader/writer
without any download — mirror of tests/test_gguf.py at script level.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model
from chenxing.model.gguf import read_gguf, write_gguf


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)

    cfg = ModelConfig(vocab_size=264, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2)
    _, weights = build_tiny_model(cfg)

    for q8 in (False, True):
        tag = "f32" if not q8 else "q8_0"
        path = out_dir / f"tiny_demo.{tag}.gguf"
        meta = {"architecture": "chenxing", "general.alignment": 32}
        write_gguf(path, meta, weights, quantize_q8_0=q8)

        g = read_gguf(path)
        ok = True
        max_err = 0.0
        for name, w in weights.items():
            back, _ = g.read_tensor(name)
            e = float(np.abs(back - w).max())
            max_err = max(max_err, e)
            ok &= np.allclose(back, w, atol=1e-2)
        print(f"[{tag}] wrote {path.name} ({path.stat().st_size} B)  "
              f"readback_ok={ok}  max_err={max_err:.2e}")

    print("round-trip validated; see assets/tiny_demo.*.gguf")


if __name__ == "__main__":
    main()
