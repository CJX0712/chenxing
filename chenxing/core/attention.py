"""Attention kernels.

Provenance:

- Scaled dot-product attention  : Vaswani et al., "Attention Is All You Need",
                                  NeurIPS 2017 (arXiv:1706.03762).
- Online (flash) softmax blocks : Milakov & Gimelshein 2018 (arXiv:1805.02867);
                                  Dao et al., FlashAttention, NeurIPS 2022
                                  (arXiv:2205.14135). We implement the numerically
                                  equivalent blocked online-softmax formulation:
                                  running max m, running sum l, rescaling output
                                  accumulator as blocks stream in.
- Grouped-Query Attention (GQA) : Ainslie et al., "GQA: Training Generalized
                                  Multi-Query Transformer Models", 2023
                                  (arXiv:2305.13245). K/V heads are shared across
                                  query heads via a repeat factor.
- Sliding-window attention      : Beltagy et al., Longformer 2020 (arXiv:2004.05150);
                                  Mistral (arXiv:2310.06825) rolling buffers.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .tensor import CausalMask

__all__ = ["flash_attention", "naive_attention", "gqa_expand"]


def gqa_expand(kv: NDArray, heads_kv: int, heads_q: int) -> NDArray:
    """Repeat K/V heads so each query head has a partner (Ainslie et al. 2023).

    kv: (b, kv_heads, s, d) → (b, q_heads, s, d), head h_q maps to kv head
    h_q // (heads_q // heads_kv). Zero-copy strided view via np.repeat on axis 1
    materialized lazily by matmul — we return a contiguous repeat for clarity.
    """
    if heads_kv == heads_q:
        return kv
    rep = heads_q // heads_kv
    return np.repeat(kv, rep, axis=1)


def flash_attention(
    q: NDArray,          # (b, hq, sq, d)
    k: NDArray,          # (b, hk, sk, d)  — may be fewer heads (GQA)
    v: NDArray,          # (b, hk, sk, dv)
    mask: CausalMask | NDArray | None = None,
    scale: float | None = None,
    *,
    block_q: int = 64,
    block_k: int = 64,
) -> NDArray:
    """Blocked online-softmax attention (numerically exact, memory-bounded).

    Streams over K/V in blocks of `block_k`, maintaining running statistics
    (m = max, l = sum-exp, acc = weighted V). Exact same math as the reference
    softmax; verified in tests against the naive kernel to ~1e-6.

    mask: CausalMask (optionally with sliding_window) or a boolean array of
    shape (sq, sk) where True = attend.
    """
    b, hq, sq, d = q.shape
    hk, sk = k.shape[1], k.shape[2]
    dv = v.shape[-1]
    if hq % hk != 0:
        raise ValueError(f"GQA requires hq % hk == 0, got hq={hq}, hk={hk}")
    if hk != hq:
        k = gqa_expand(k, hk, hq)
        v = gqa_expand(v, hk, hq)
    scale = (1.0 / np.sqrt(d)) if scale is None else scale

    out = np.empty((b, hq, sq, dv), dtype=np.float32)

    # Expand mask into (sq, sk) boolean attend-map once.
    if isinstance(mask, CausalMask):
        pos = np.arange(sq)[:, None] + (sk - sq)   # support incremental decode (sq <= sk)
        attend = pos >= np.arange(sk)[None, :]
        if mask.sliding_window is not None:
            attend &= (np.arange(sk)[None, :] + mask.sliding_window > pos)
        window: NDArray | None = attend
    elif mask is not None:
        window = np.asarray(mask, dtype=bool)
        if window.shape != (sq, sk):
            raise ValueError(f"mask shape {window.shape} != ({sq}, {sk})")
    else:
        window = None

    for i0 in range(0, sq, block_q):
        i1 = min(i0 + block_q, sq)
        qi = q[:, :, i0:i1, :].astype(np.float32)                     # (b,hq,bq,d)
        acc = np.zeros((b, hq, i1 - i0, dv), dtype=np.float32)
        m_run = np.full((b, hq, i1 - i0, 1), -np.inf, dtype=np.float32)
        l_run = np.zeros((b, hq, i1 - i0, 1), dtype=np.float32)

        for j0 in range(0, sk, block_k):
            j1 = min(j0 + block_k, sk)
            kj = k[:, :, j0:j1, :].astype(np.float32)                 # (b,h,bk,d)
            scores = np.einsum("bhqd,bhkd->bhqk", qi, kj) * scale     # (b,h,bq,bk)

            if window is not None:
                scores = np.where(window[i0:i1, j0:j1][None, None],
                                  scores, -np.inf)

            # ---- online softmax update (Milakov & Gimelshein 2018) ----
            m_blk = np.max(scores, axis=-1, keepdims=True)            # block max
            m_new = np.maximum(m_run, m_blk)
            # rows entirely masked in this block: keep old stats, exp(-inf - m) = 0
            p = np.exp(scores - m_new)
            l_blk = np.sum(p, axis=-1, keepdims=True)
            alpha = np.exp(m_run - m_new)
            acc = acc * alpha + np.einsum("bhqk,bhkd->bhqd", p, v[:, :, j0:j1, :])
            l_run = l_run * alpha + l_blk
            m_run = m_new

        out[:, :, i0:i1, :] = acc / np.maximum(l_run, 1e-38)
    return out


def naive_attention(q: NDArray, k: NDArray, v: NDArray,
                    mask: CausalMask | NDArray | None = None,
                    scale: float | None = None) -> NDArray:
    """Reference kernel: materialize full scores, one softmax, one matmul.

    O(sq·sk) memory — used by tests as the ground-truth oracle.
    """
    b, hq, sq, d = q.shape
    hk = k.shape[1]
    if hk != hq:
        k = gqa_expand(k, hk, hq)
        v = gqa_expand(v, hk, hq)
    scale = (1.0 / np.sqrt(d)) if scale is None else scale
    scores = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float32), k.astype(np.float32)) * scale

    if isinstance(mask, CausalMask):
        pos = np.arange(sq)[:, None] + (k.shape[2] - sq)
        attend = pos >= np.arange(k.shape[2])[None, :]
        if mask.sliding_window is not None:
            attend &= (np.arange(k.shape[2])[None, :] + mask.sliding_window > pos)
        scores = np.where(attend[None, None], scores, -np.inf)
    elif mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool)[None, None], scores, -np.inf)

    with np.errstate(invalid="ignore"):
        m = np.max(scores, axis=-1, keepdims=True)
        w = np.exp(scores - m)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = w / np.maximum(w.sum(axis=-1, keepdims=True), 1e-38)
    return np.einsum("bhqk,bhkd->bhqd", w, v.astype(np.float32))
