"""Tensor primitive invariants: RMSNorm, RoPE, SwiGLU, YaRN."""
import numpy as np
import pytest

from chenxing.core.tensor import apply_rope, build_rope_cache, rms_norm, silu, swiglu


def test_rmsnorm_unit_rms():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    y = rms_norm(x, eps=1e-6)
    rms = np.sqrt((y ** 2).mean(axis=-1))
    assert np.abs(rms - 1.0).max() < 1e-5


def test_rmsnorm_affine_scales():
    x = np.ones((1, 16), dtype=np.float32)
    w = np.arange(16, dtype=np.float32)
    y = rms_norm(x, w, eps=1e-8)
    assert np.allclose(y, w / np.sqrt(1.0 + 1e-8), atol=1e-5)


def test_rope_preserves_norm_per_position():
    """RoPE is an orthogonal rotation — the token norm is invariant."""
    rng = np.random.default_rng(1)
    cos, sin = build_rope_cache(16, 32, base=10000.0)
    x = rng.standard_normal((2, 4, 16, 32)).astype(np.float32)
    r = apply_rope(x, cos, sin)
    assert np.abs(np.linalg.norm(x, axis=-1) - np.linalg.norm(r, axis=-1)).max() < 1e-4


def test_rope_relative_position_property():
    """dot(q_pos1, k_pos2) depends only on (pos1 - pos2) — the defining property."""
    cos, sin = build_rope_cache(24, 16, base=10000.0)
    rng = np.random.default_rng(2)
    q0 = rng.standard_normal((1, 1, 1, 16)).astype(np.float32)
    k0 = rng.standard_normal((1, 1, 1, 16)).astype(np.float32)
    dots = []
    for offset in range(0, 8):
        q = apply_rope(q0, cos, sin, positions=np.array([4 + offset]))
        k = apply_rope(k0, cos, sin, positions=np.array([4]))
        dots.append(float((q * k).sum()))
    # same relative offset (0) must give the same dot at different absolute pos
    q = apply_rope(q0, cos, sin, positions=np.array([12]))
    k = apply_rope(k0, cos, sin, positions=np.array([12]))
    assert abs(dots[0] - float((q * k).sum())) < 1e-4


def test_rope_interleaved_layout_differs_but_preserves_norm():
    cos, sin = build_rope_cache(8, 16)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 1, 8, 16)).astype(np.float32)
    a = apply_rope(x, cos, sin, interleaved=False)
    b = apply_rope(x, cos, sin, interleaved=True)
    assert np.abs(a - b).max() > 1e-6          # layouts are genuinely different
    assert np.abs(np.linalg.norm(a, axis=-1) - np.linalg.norm(b, axis=-1)).max() < 1e-4


def test_yarn_scaling_interpolates_low_frequencies():
    cos0, _ = build_rope_cache(256, 16, base=10000.0)
    cos1, _ = build_rope_cache(256, 16, base=10000.0, scaling_factor=4.0)
    # low-frequency dim (last) must change; high-frequency dim 0 must stay
    assert np.abs(cos0[:, -1] - cos1[:, -1]).max() > 1e-4
    assert np.abs(cos0[:, 0] - cos1[:, 0]).max() < 1e-6


def test_silu_and_swiglu_shapes():
    x = np.linspace(-3, 3, 7, dtype=np.float32)
    assert np.allclose(silu(x), x / (1 + np.exp(-x)), atol=1e-6)
    g = np.ones((2, 4), dtype=np.float32)
    u = np.full((2, 4), 2.0, dtype=np.float32)
    assert np.allclose(swiglu(g, u), silu(g) * 2.0)
