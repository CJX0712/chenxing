"""Core numerical primitives.

Provenance (zero-reinvention — every formula below is imported, not invented):

- RMSNorm                : Zhang & Sennrich, "Root Mean Square Layer Normalization",
                           NeurIPS 2019 (arXiv:1907.02039). Used by Llama/Qwen/Mistral.
- SwiGLU                 : Shazeer, "GLU Variants Improve Transformer", 2020
                           (arXiv:2002.05202). FFN(x) = W2( SiLU(W1 x) * W3 x ).
- RoPE                   : Su et al., "RoFormer: Enhanced Transformer with Rotary
                           Position Embedding", 2021 (arXiv:2104.09864).
- NTK-aware / YaRN drift : Peng et al., "YaRN: Efficient Context Window Extension",
                           2023 (arXiv:2309.00071) — frequency-dependent interpolation
                           ramp + temperature scaling.
- Online (flash) softmax : Milakov & Gimelshein, "Online normalizer calculation for
                           softmax", 2018 (arXiv:1805.02867); Dao et al.,
                           "FlashAttention", NeurIPS 2022 (arXiv:2205.14135).

All kernels are exact-math equivalent to their references; tests re-derive each
one from first principles and compare within float tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "rms_norm", "silu", "swiglu", "softmax", "log_softmax",
    "build_rope_cache", "apply_rope", "rope_cache_f32",
    "AttentionMask", "CausalMask",
]


def rms_norm(x: NDArray[np.floating], weight: NDArray[np.floating] | None = None,
             eps: float = 1e-5) -> NDArray[np.floating]:
    """RMSNorm (Zhang & Sennrich 2019): y = x / sqrt(mean(x^2) + eps) * weight.

    Invariant: output has zero mean-square ≈ 1 along the normalized axis before
    the (optional) affine gain.
    """
    dtype = x.dtype
    x32 = np.asarray(x, dtype=np.float32)
    inv = x32 * (1.0 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps))
    out = inv.astype(dtype, copy=False)
    if weight is not None:
        out = out * weight
    return out


def silu(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """SiLU / Swish: x * sigmoid(x) (Hendrycks & Gimpel 2016; Ramachandran 2017)."""
    x32 = np.asarray(x, dtype=np.float32)
    return (x32 / (1.0 + np.exp(-x32))).astype(x.dtype, copy=False)


def swiglu(gate: NDArray[np.floating], up: NDArray[np.floating]) -> NDArray[np.floating]:
    """SwiGLU (Shazeer 2020): silu(gate) * up."""
    return silu(gate) * up


def softmax(x: NDArray[np.floating], axis: int = -1) -> NDArray[np.floating]:
    """Stable softmax via the max-shift identity; sum along `axis` is exactly 1."""
    x32 = np.asarray(x, dtype=np.float64) if x.dtype == np.float64 else np.asarray(x, dtype=np.float32)
    m = np.max(x32, axis=axis, keepdims=True)
    e = np.exp(x32 - m)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype(x.dtype, copy=False)


def log_softmax(x: NDArray[np.floating], axis: int = -1) -> NDArray[np.floating]:
    """logsumexp-shifted log-softmax, overflow-proof (logsumexp trick)."""
    x32 = np.asarray(x, dtype=np.float64)
    m = np.max(x32, axis=axis, keepdims=True)
    shifted = x32 - m
    lse = m + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return (x32 - lse).astype(x.dtype, copy=False)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     *, scaling_factor: float = 1.0, original_max_pos: int = 4096,
                     beta_fast: float = 32.0, beta_slow: float = 1.0,
                     dtype: np.dtype | None = None) -> tuple[NDArray, NDArray]:
    """Precompute cos/sin tables for RoPE with optional YaRN-style extension.

    Plain RoPE (scaling_factor == 1): theta_i = base^(-2i/head_dim).

    YaRN (Peng et al. 2023) when scaling_factor > 1: low-frequency components
    are linearly interpolated, high-frequency ones are extrapolated as-is, with
    a smooth ramp on the "extrapolation vs interpolation" boundary and the
    temperature correction sqrt(1/t * ... ). We implement the standard ramp:

        r(i) = clamp((dim/2 - i) / (dim/2), 0, 1)
        gamma(i) = beta_fast_interp if r(i) > beta threshold ... 

    following the reference implementation used by HuggingFace transformers
    (`_compute_yarn_parameters`): find the rotary dim where wavelengths fall
    between the fix boundaries, apply interpolation only below it.
    """
    half = head_dim // 2
    inv_freq = base ** (-np.arange(0, half, dtype=np.float64) * 2.0 / head_dim)

    if scaling_factor != 1.0:
        # YaRN (Peng et al. 2023, eq. 12-15; HF `_compute_yarn_parameters`):
        # frequencies are split by a smooth ramp — high-frequency dims keep the
        # base frequency (extrapolation), low-frequency dims are linearly
        # interpolated by 1/scale, with the ramp boundary derived from the
        # wavelengths the model was actually trained on.
        low = head_dim * np.log(original_max_pos / (beta_fast * 2 * np.pi)) / (2 * np.log(base))
        high = head_dim * np.log(original_max_pos / (beta_slow * 2 * np.pi)) / (2 * np.log(base))
        low, high = max(float(np.floor(low)), 0.0), min(float(np.ceil(high)), half - 1.0)
        ramp = np.clip((np.arange(half, dtype=np.float64) - low) / max(high - low, 1e-9), 0.0, 1.0)
        # ramp=0 (high freq) → extrapolate as-is; ramp=1 (low freq) → interpolate.
        inv_freq = inv_freq * (1 - ramp) + (inv_freq / scaling_factor) * ramp

    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)                       # (seq, half)
    emb = np.concatenate([freqs, freqs], axis=-1)       # duplicate for rotate_half
    cos = np.cos(emb).astype(dtype or np.float32)
    sin = np.sin(emb).astype(dtype or np.float32)
    return cos, sin


def rope_cache_f32(seq_len: int, head_dim: int, base: float = 10000.0,
                   scaling_factor: float = 1.0) -> tuple[NDArray, NDArray]:
    return build_rope_cache(seq_len, head_dim, base, scaling_factor=scaling_factor, dtype=np.float32)


def apply_rope(x: NDArray, cos: NDArray, sin: NDArray, *, positions: NDArray | None = None,
               interleaved: bool = False) -> NDArray:
    """Apply rotary position embedding to `x` of shape (..., seq, head_dim).

    Two layouts are supported (both from ggml/HF, chosen per model family):
      - half-split (Llama/Qwen/Mistral default): x = [x1, x2] halves, rotate pair.
      - interleaved (GPT-NeoX-style adjacent pairs), used by some GGUF exports.

    `positions` (shape (seq,)) selects rows of the cache for incremental decode.
    """
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
    else:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]

    half = x.shape[-1] // 2
    if positions is not None:
        cos_ = cos[positions]                       # (T, head_dim)
        sin_ = sin[positions]
    else:
        # seq axis: axis 0 for (T, h, d), axis -2 for (b, h, T, d)
        seq_len = x.shape[0] if x.ndim == 3 else x.shape[-2]
        cos_ = cos[:seq_len]                        # (T, head_dim)
        sin_ = sin[:seq_len]

    # The cos/sin tables are full head_dim wide (build_rope_cache duplicates
    # `freqs` across the two halves). Broadcast the token axis against `x`:
    #   3D (T, h, head_dim)  -> (T, 1, head_dim)
    #   4D (b, h, T, head_dim) -> (1, 1, T, head_dim)
    if x.ndim == 3:
        cos_b = cos_[:, None, :]
        sin_b = sin_[:, None, :]
        cos_h = cos_[:, None, :half]
        sin_h = sin_[:, None, :half]
    elif x.ndim == 4:
        cos_b = cos_[None, None, :, :]
        sin_b = sin_[None, None, :, :]
        cos_h = cos_[None, None, :, :half]
        sin_h = sin_[None, None, :, :half]
    else:
        raise ValueError(f"apply_rope: unsupported input ndim {x.ndim} (need 3 or 4)")

    if interleaved:
        out = np.empty_like(x)
        out[..., 0::2] = x1 * cos_h - x2 * sin_h
        out[..., 1::2] = x1 * sin_h + x2 * cos_h
        return out
    # half-split (Llama/Qwen/Mistral): rotate the two halves jointly.
    rotated = np.concatenate([-x2, x1], axis=-1)   # (..., head_dim)
    return x * cos_b + rotated * sin_b


@dataclass(frozen=True)
class CausalMask:
    """Marker for causal (lower-triangular) attention with optional sliding window."""
    sliding_window: int | None = None


AttentionMask = CausalMask | NDArray | None
