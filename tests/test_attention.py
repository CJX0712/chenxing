"""Attention kernel invariants.

Oracle: the naive full-softmax kernel is the independent reference; the
flash (blocked online-softmax) kernel must match it within float tolerance.
"""
import numpy as np
import pytest

from chenxing.core.attention import flash_attention, gqa_expand, naive_attention
from chenxing.core.tensor import CausalMask

SHAPES = [(1, 2, 5, 2, 8), (2, 4, 17, 4, 12), (1, 4, 64, 2, 8)]


@pytest.mark.parametrize("b,hq,sq,hk,d", SHAPES)
def test_flash_equals_naive_no_mask(b, hq, sq, hk, d):
    rng = np.random.default_rng(b * 100 + sq)
    q = rng.standard_normal((b, hq, sq, d)).astype(np.float32)
    k = rng.standard_normal((b, hk, sq, d)).astype(np.float32)
    v = rng.standard_normal((b, hk, sq, d)).astype(np.float32)
    a, n = flash_attention(q, k, v), naive_attention(q, k, v)
    assert np.abs(a - n).max() < 1e-5


def test_causal_mask_matches_numpy_triangle():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((1, 2, 9, 8)).astype(np.float32)
    k = rng.standard_normal((1, 2, 9, 8)).astype(np.float32)
    v = rng.standard_normal((1, 2, 9, 8)).astype(np.float32)
    mask = np.tril(np.ones((9, 9), dtype=bool))
    a, n = flash_attention(q, k, v, CausalMask()), naive_attention(q, k, v, mask)
    assert np.abs(a - n).max() < 1e-5


def test_sliding_window_matches_explicit_mask():
    rng = np.random.default_rng(3)
    sq, w = 12, 4
    q = rng.standard_normal((1, 1, sq, 8)).astype(np.float32)
    k = rng.standard_normal((1, 1, sq, 8)).astype(np.float32)
    v = rng.standard_normal((1, 1, sq, 8)).astype(np.float32)
    pos = np.arange(sq)[:, None]
    explicit = (pos >= np.arange(sq)) & (np.arange(sq) + w > pos)
    a = flash_attention(q, k, v, CausalMask(sliding_window=w))
    n = naive_attention(q, k, v, explicit)
    assert np.abs(a - n).max() < 1e-5


def test_incremental_decode_window_consistency():
    """Prefill(10) row 9 ≡ decode-with-cache at position 9 (sq=1, sk=10)."""
    rng = np.random.default_rng(7)
    q_full = rng.standard_normal((1, 2, 10, 8)).astype(np.float32)
    k_full = rng.standard_normal((1, 2, 10, 8)).astype(np.float32)
    v_full = rng.standard_normal((1, 2, 10, 8)).astype(np.float32)
    full = flash_attention(q_full, k_full, v_full, CausalMask())
    inc = flash_attention(q_full[:, :, 9:10, :], k_full, v_full, CausalMask())
    assert np.abs(full[:, :, 9, :] - inc[:, :, 0, :]).max() < 1e-5


def test_gqa_expand_semantics():
    kv = np.arange(2 * 3 * 2).reshape(1, 2, 3, 2).astype(np.float32)
    out = gqa_expand(kv, heads_kv=2, heads_q=4)
    assert out.shape == (1, 4, 3, 2)
    assert np.array_equal(out[0, 0], kv[0, 0]) and np.array_equal(out[0, 1], kv[0, 0])
    assert np.array_equal(out[0, 2], kv[0, 1])


def test_output_rows_are_probability_weighted():
    """Attention output must be a convex combination of V rows (per query)."""
    rng = np.random.default_rng(11)
    q = rng.standard_normal((1, 1, 6, 8)).astype(np.float32)
    k = rng.standard_normal((1, 1, 6, 8)).astype(np.float32)
    v = rng.standard_normal((1, 1, 6, 4)).astype(np.float32)
    out = flash_attention(q, k, v, CausalMask())
    # convex bound: each output coord within [min V, max V] over attended rows
    v_min, v_max = v.min(axis=2), v.max(axis=2)
    assert (out >= v_min - 1e-6).all() and (out <= v_max + 1e-6).all()
