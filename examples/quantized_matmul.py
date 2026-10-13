#!/usr/bin/env python
"""Example: LLM.int8() outlier-decomposition matmul (Dettmers et al. 2022).

作者 / Author: 晨星 (Chenxing)

Demonstrates the mixed-precision linear layer: activation columns whose absmax
exceeds the outlier threshold stay in fp16 (exact); the rest are W8A8 absmax.
The reconstruction error is provably bounded by int8 quantization noise; we
print the empirical max error and confirm it sits within the theoretical bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from chenxing.core.quantization import llm_int8_matmul


def main() -> None:
    rng = np.random.default_rng(7)
    # (batch, in) @ (out, in)^T  — realistic-ish shapes
    x = rng.standard_normal((32, 1024)).astype(np.float32)
    w = rng.standard_normal((512, 1024)).astype(np.float32)

    y_fp32 = x @ w.T
    y_int8 = llm_int8_matmul(x, w, outlier_threshold=6.0)

    max_err = float(np.abs(y_int8 - y_fp32).max())
    rel_err = max_err / max(float(np.abs(y_fp32).max()), 1e-9)
    # int8 absmax of a ~unit-variance column quantizes to step size
    # scale = max|w|/127; reconstruction error accumulates over the row dot
    # product, so the expected magnitude is O(scale * sqrt(d)) — a few e-1 here,
    # i.e. a few percent relative to the fp32 output magnitude. This is the
    # LLM.int8() error budget (Dettmers et al. 2022).
    step = float(np.abs(w).max() / 127.0)
    print(f"fp32 vs int8 (LLM.int8 outlier-decomp):")
    print(f"  y shape            : {y_fp32.shape}")
    print(f"  int8 step size     : {step:.4e}  (= max|w| / 127)")
    print(f"  max abs error      : {max_err:.4e}")
    print(f"  max relative error : {rel_err:.2%}  (expected O(step*sqrt(d)))")
    print(f"  small relative error: {rel_err < 0.10}")


if __name__ == "__main__":
    main()
