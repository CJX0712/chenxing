"""Quantization: integer formats + GGUF k-quant dequantization.

Provenance (all formats imported from their reference definitions):

- Absmax integer quantization : Jacob et al., "LLM.int8(): 8-bit Matrix
  Multiplication for Transformers at Scale", NeurIPS 2022 (arXiv:2208.07339).
- Outlier decomposition       : same paper — a few activation columns carry
  magnitudes ~20-100x the rest; keep those dimensions in fp16, quantize the rest.
- GGUF k-quants               : ggml-org/llama.cpp, ggml-quants.c (MIT). The
  dequantizers below are exact ports of dequantize_row_q2_K / q4_K / q6_K /
  q8_0 including get_scale_min_k4 bit-packing; tests validate against blocks
  hand-packed to the C byte layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "quantize_absmax", "dequantize_absmax", "QuantizedLinear", "llm_int8_matmul",
    "dequantize_gguf_row", "GGMLType", "QK_K",
]

QK_K = 256  # super-block size shared by all K-quants (llama.cpp ggml-common.h)


# --------------------------------------------------------------------------- #
# 1. Generic per-group absmax integer quantization (LLM.int8 baseline)
# --------------------------------------------------------------------------- #

def quantize_absmax(x: NDArray, bits: int = 4, group_size: int = 64
                    ) -> tuple[NDArray[np.int8], NDArray[np.float32]]:
    """Per-group symmetric absmax quantization (LLM.int8 reference scheme).

    scale_g = max|x_g| / (2^(b-1) - 1);  q = round(x / scale_g), clipped.
    """
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1]} not divisible by group_size {group_size}")
    qmax = 2 ** (bits - 1) - 1
    x32 = np.asarray(x, dtype=np.float32)
    g = x32.reshape(-1, group_size)
    amax = np.max(np.abs(g), axis=1, keepdims=True)
    scales = np.where(amax == 0, np.float32(1.0), amax / qmax).astype(np.float32)
    q = np.clip(np.rint(g / scales), -qmax, qmax).astype(np.int8)
    return q.reshape(x32.shape), scales.reshape(*x32.shape[:-1], x32.shape[-1] // group_size)


def dequantize_absmax(q: NDArray[np.int8], scales: NDArray[np.float32],
                      group_size: int = 64) -> NDArray[np.float32]:
    *lead, n = q.shape
    return (q.reshape(-1, group_size).astype(np.float32)
            * scales.reshape(-1, 1)).reshape(*lead, n)


def _pick_group(n: int, want: int) -> int:
    for g in (want, 64, 32, 16, 8, 4, 2, 1):
        if g <= n and n % g == 0:
            return g
    return 1


def llm_int8_matmul(x: NDArray, w: NDArray, *, outlier_threshold: float = 6.0,
                    group_size: int = 64) -> NDArray:
    """LLM.int8() mixed-precision matmul (Dettmers et al. 2022, Algs. 1-2).

    Activation columns whose absmax exceeds `outlier_threshold` are computed
    in fp16 (kept exact); the remaining columns go through W8A8 absmax
    quantization:  Y = Q_x·S_x @ (Q_w·S_w)ᵀ  with error bounded by the int8
    quantization noise (verified in tests).
    """
    x32 = np.asarray(x, dtype=np.float32)
    w32 = np.asarray(w, dtype=np.float32)
    amax = np.max(np.abs(x32), axis=0)
    outlier = amax > outlier_threshold

    y = np.zeros(x32.shape[:-1] + (w32.shape[0],), dtype=np.float32)
    if outlier.any():
        y += x32[..., outlier] @ w32[:, outlier].T
    norm_idx = np.flatnonzero(~outlier)
    if norm_idx.size:
        g = _pick_group(int(norm_idx.size), group_size)
        xn, sn = quantize_absmax(x32[..., norm_idx], bits=8, group_size=g)
        wn, sw = quantize_absmax(w32[:, norm_idx], bits=8, group_size=g)
        y += dequantize_absmax(xn, sn, g) @ dequantize_absmax(wn, sw, g).T
    return y


def _pack_nibbles(q: NDArray[np.int8]) -> NDArray[np.uint8]:
    """Pack int4 codes (shifted to unsigned nibbles) two-per-byte."""
    flat = (np.asarray(q, dtype=np.int32).reshape(-1) + 8).astype(np.uint8) & 0xF
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, np.uint8)])
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def _unpack_nibbles(p: NDArray[np.uint8], shape: tuple[int, ...]) -> NDArray[np.float32]:
    p = np.asarray(p, dtype=np.uint8).reshape(-1)
    lo = (p & 0xF).astype(np.int32) - 8
    hi = ((p >> 4) & 0xF).astype(np.int32) - 8
    out = np.empty(p.size * 2, dtype=np.int32)
    out[0::2] = lo
    out[1::2] = hi
    return out.astype(np.float32).reshape(shape)


@dataclass
class QuantizedLinear:
    """Linear layer stored at `bits` precision, dequantized once at use time.

    int4 codes are nibble-packed (two per byte) for true 4-bit storage;
    int8 stays one byte per code. Optimal for CPU/NumPy inference: decode
    cost paid once, matmul in fp32.
    """
    weight_q: NDArray            # int8 (bits=8) or packed uint8 (bits=4)
    scales: NDArray[np.float32]
    bias: NDArray | None
    bits: int
    group_size: int

    def __init__(self, weight: NDArray, bias: NDArray | None = None,
                 bits: int = 4, group_size: int = 64) -> None:
        w = np.atleast_2d(np.asarray(weight, dtype=np.float32))
        if w.shape[1] % group_size != 0:
            raise ValueError(f"in_features {w.shape[1]} not divisible by group_size {group_size}")
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        q, s = quantize_absmax(w, bits, group_size)
        if bits == 4:
            self.weight_q = _pack_nibbles(q).reshape(w.shape[0], w.shape[1] // 2)
        else:
            self.weight_q = q
        self.scales = s
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.bits, self.group_size = bits, group_size

    @property
    def bits_per_weight(self) -> float:
        n_weights = self.scales.shape[0] * self.scales.shape[1] * self.group_size
        return (self.weight_q.nbytes + self.scales.nbytes) * 8 / n_weights

    def dequantized(self) -> NDArray[np.float32]:
        if self.bits == 4:
            out_f = self.scales.shape[0]
            in_f = self.scales.shape[1] * self.group_size
            q = _unpack_nibbles(self.weight_q, (out_f, in_f))
        else:
            q = self.weight_q.astype(np.float32)
        return dequantize_absmax(q, self.scales, self.group_size)

    def __call__(self, x: NDArray) -> NDArray:
        y = x @ self.dequantized().T
        return y if self.bias is None else y + self.bias


# --------------------------------------------------------------------------- #
# 2. GGUF k-quant dequantization — exact ports of ggml-quants.c (MIT)
# --------------------------------------------------------------------------- #

class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

    #: type -> (block_elements, block_bytes), per ggml-common.h
    BLOCK_SIZES = {
        F32: (1, 4), F16: (1, 2), Q4_0: (32, 18), Q8_0: (32, 34),
        Q2_K: (256, 84), Q3_K: (256, 110), Q4_K: (256, 144),
        Q5_K: (256, 176), Q6_K: (256, 210), Q8_K: (256, 292),
    }


def _f16_to_f32(buf: NDArray[np.uint8]) -> NDArray[np.float32]:
    return np.frombuffer(buf.tobytes(), dtype="<f2").astype(np.float32)


def _get_scale_min_k4(j: int, q: NDArray[np.uint8]) -> tuple[int, int]:
    """Port of get_scale_min_k4 (ggml-quants.c): 16 x 6-bit values in 12 bytes."""
    if j < 4:
        return int(q[j]) & 63, int(q[j + 4]) & 63
    d = (int(q[j + 4]) & 0xF) | ((int(q[j - 4]) >> 6) << 4)
    m = (int(q[j + 4]) >> 4) | ((int(q[j]) >> 6) << 4)
    return d, m


def _dequant_q2_K(raw: NDArray[np.uint8], n_blocks: int) -> NDArray[np.float32]:
    """block_q2_K = scales[16] qs[64] d(f16) dmin(f16) → 84 B / 256 weights.

    Exact port of ggml-quants.c dequantize_row_q2_K: two super-groups of 128
    weights; within each, 4 shift planes (0,2,4,6) interleave q[0:16] and
    q[16:32], each 16-wide slice scaled by its own scales[] entry (low nibble
    = scale, high nibble = min, multiplied by the block d / dmin).
    """
    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for i in range(n_blocks):
        b = raw[i * 84:(i + 1) * 84]
        scales, qs = b[0:16], b[16:80]
        d = float(_f16_to_f32(b[80:82])[0])
        dmin = float(_f16_to_f32(b[82:84])[0])
        y = out[i]
        is_ = 0
        q = qs
        for n in range(0, QK_K, 128):           # 2 super-groups
            shift = 0
            for j in range(4):
                base = n + j * 32
                sc = int(scales[is_]); is_ += 1
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4)
                y[base:base + 16] = dl * ((q[0:16].astype(np.int32) >> shift) & 3) - ml
                sc = int(scales[is_]); is_ += 1
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4)
                y[base + 16:base + 32] = dl * ((q[16:32].astype(np.int32) >> shift) & 3) - ml
                shift += 2
            q = q[32:]
    return out


def _dequant_q4_K(raw: NDArray[np.uint8], n_blocks: int) -> NDArray[np.float32]:
    """block_q4_K = d(f16) dmin(f16) scales[12] qs[128] → 144 B / 256 weights."""
    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for i in range(n_blocks):
        b = raw[i * 144:(i + 1) * 144]
        d = float(_f16_to_f32(b[0:2])[0])
        dmin = float(_f16_to_f32(b[2:4])[0])
        scales, qs = b[4:16], b[16:144]
        y = out[i]
        qoff = iscale = 0
        for n in range(0, QK_K, 64):           # C: j += 64, 32 bytes per group
            sc, m = _get_scale_min_k4(iscale, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _get_scale_min_k4(iscale + 1, scales)
            d2, m2 = d * sc, dmin * m
            lo = (qs[qoff:qoff + 32] & 0xF).astype(np.float32)
            hi = (qs[qoff:qoff + 32].astype(np.int32) >> 4).astype(np.float32)
            y[n:n + 32] = d1 * lo - m1
            y[n + 32:n + 64] = d2 * hi - m2
            qoff += 32
            iscale += 2
    return out


def _dequant_q6_K(raw: NDArray[np.uint8], n_blocks: int) -> NDArray[np.float32]:
    """block_q6_K = ql[128] qh[64] scales[16] d(f16) → 210 B / 256 weights."""
    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for i in range(n_blocks):
        b = raw[i * 210:(i + 1) * 210]
        ql, qh = b[0:128], b[128:192]
        sc = b[192:208].astype(np.int8)
        d = float(_f16_to_f32(b[208:210])[0])
        y = out[i]
        qloff, qhoff, scoff = 0, 0, 0
        for n in range(0, QK_K, 128):
            for l in range(32):
                is_ = l // 16
                q1 = ((int(ql[qloff + l]) & 0xF) | (((int(qh[qhoff + l]) >> 0) & 3) << 4)) - 32
                q2 = ((int(ql[qloff + l + 32]) & 0xF) | (((int(qh[qhoff + l]) >> 2) & 3) << 4)) - 32
                q3 = ((int(ql[qloff + l]) >> 4) | (((int(qh[qhoff + l]) >> 4) & 3) << 4)) - 32
                q4 = ((int(ql[qloff + l + 32]) >> 4) | (((int(qh[qhoff + l]) >> 6) & 3) << 4)) - 32
                y[n + l] = d * sc[scoff + is_ + 0] * q1
                y[n + l + 32] = d * sc[scoff + is_ + 2] * q2
                y[n + l + 64] = d * sc[scoff + is_ + 4] * q3
                y[n + l + 96] = d * sc[scoff + is_ + 6] * q4
            qloff += 64
            qhoff += 32
            scoff += 8
    return out


def _dequant_q8_0(raw: NDArray[np.uint8], n_blocks: int) -> NDArray[np.float32]:
    """block_q8_0 = d(f16) qs[32] → 34 B / 32 weights. uint8 → int8 reinterpret."""
    out = np.empty((n_blocks * 32,), dtype=np.float32)
    for i in range(n_blocks):
        b = raw[i * 34:(i + 1) * 34]
        d = float(_f16_to_f32(b[0:2])[0])
        out[i * 32:(i + 1) * 32] = d * b[2:34].astype(np.int8).astype(np.float32)
    return out


_DEQUANTERS = {GGMLType.Q2_K: _dequant_q2_K, GGMLType.Q4_K: _dequant_q4_K,
               GGMLType.Q6_K: _dequant_q6_K, GGMLType.Q8_0: _dequant_q8_0}


def dequantize_gguf_row(raw: bytes | NDArray, ggml_type: int, n_elements: int) -> NDArray[np.float32]:
    """Decode one tensor row from its GGUF/ggml byte representation.

    Supported: F32, F16, Q8_0, Q2_K, Q4_K, Q6_K (the formats covering the
    dominant llama.cpp Q4_K_M / Q2_K / Q6_K model families).
    """
    buf = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, bytes) else raw
    if ggml_type == GGMLType.F32:
        return buf.view("<f4")[:n_elements].astype(np.float32).copy()
    if ggml_type == GGMLType.F16:
        return buf.view("<f2")[:n_elements].astype(np.float32).copy()
    fn = _DEQUANTERS.get(ggml_type)
    if fn is None:
        raise NotImplementedError(f"ggml type {ggml_type} not supported yet")
    blk_elems, blk_bytes = GGMLType.BLOCK_SIZES[ggml_type]
    return fn(buf, n_elements // blk_elems).reshape(-1)[:n_elements]
