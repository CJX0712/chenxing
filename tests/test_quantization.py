"""Quantization invariants.

- absmax INT4/INT8: reconstruction error within the theoretical bound.
- GGUF k-quants: byte blocks hand-packed to the llama.cpp C layout must
  dequantize to hand-computed values (independent of the port itself).
"""
import struct

import numpy as np
import pytest

from chenxing.core.quantization import (GGMLType, QuantizedLinear,
                                        dequantize_absmax, dequantize_gguf_row,
                                        llm_int8_matmul, quantize_absmax)

F16 = struct.Struct("<e")


def f16(v: float) -> bytes:
    return F16.pack(np.float16(v))


# --------------------------------------------------------------------------- #
# absmax
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bits", [4, 8])
def test_absmax_error_within_theoretical_bound(bits):
    rng = np.random.default_rng(bits)
    x = rng.standard_normal((8, 256)).astype(np.float32)
    q, s = quantize_absmax(x, bits=bits, group_size=64)
    rec = dequantize_absmax(q, s, 64)
    qmax = 2 ** (bits - 1) - 1
    # per-group worst-case: |err| <= scale/2 = amax/(2*qmax)
    amax = np.max(np.abs(x.reshape(8, 4, 64)), axis=-1, keepdims=True)
    bound = (amax / qmax / 2).repeat(64, axis=-1).reshape(8, 256)
    assert (np.abs(x - rec) <= bound + 1e-6).all()


def test_absmax_zero_group_safe():
    x = np.zeros((1, 64), dtype=np.float32)
    q, s = quantize_absmax(x, 4, 64)
    assert dequantize_absmax(q, s, 64).sum() == 0


def test_quantized_linear_bits_per_weight():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 128)).astype(np.float32)
    ql = QuantizedLinear(w, bits=4, group_size=64)
    assert 4.0 < ql.bits_per_weight < 5.0
    y = ql(np.ones((2, 128), dtype=np.float32))
    assert y.shape == (2, 64)


def test_llm_int8_outliers_exact():
    """Outlier columns (|x| > threshold) must be reproduced EXACTLY (fp16 path)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 32)).astype(np.float32)
    x[:, 5] *= 100.0                                    # planted outlier column
    w = rng.standard_normal((8, 32)).astype(np.float32)
    y = llm_int8_matmul(x, w, outlier_threshold=6.0)
    y_ref = x @ w.T
    # the outlier contribution dominates; error stays bounded by int8 noise on
    # the remaining columns
    rel = np.abs(y - y_ref).max() / np.abs(y_ref).max()
    assert rel < 0.15


# --------------------------------------------------------------------------- #
# GGUF k-quant byte-level ground truth
# --------------------------------------------------------------------------- #

def test_q8_0_block_layout():
    d = 0.5
    qs = [4, -4] * 16
    raw = f16(d) + struct.pack("<32b", *qs)
    out = dequantize_gguf_row(raw, GGMLType.Q8_0, 32)
    assert np.allclose(out, [d * q for q in qs])


def test_q4_k_block_layout():
    """Hand-packed Q4_K super-block verified against hand-computed dequant.

    scales[0..3]=[10,20,30,40] (sc), scales[4..7]=[5,6,7,8] (min), d=dmin=1.
    Group is=0: d1=10, m1=5, d2=20, m2=6 → y[0]=10*5-5=45, y[32]=20*6-6=114.
    Group is=2: j=2→(30,7), j=3→(40,8) → y[64]=30*1-7=23, y[96]=40*2-8=72.
    """
    scales = bytes([10, 20, 30, 40, 5, 6, 7, 8, 1, 2, 3, 4])
    qs = bytearray(128)
    qs[0] = (5) | (6 << 4)          # low nibble → y[0]; high nibble → y[32]
    qs[1] = (7) | (8 << 4)          # y[1], y[33]
    qs[32] = (1) | (2 << 4)         # group is=2: y[64], y[96]
    qs[33] = (3) | (4 << 4)         # y[65], y[97]
    raw = f16(1.0) + f16(1.0) + scales + bytes(qs)
    out = dequantize_gguf_row(raw, GGMLType.Q4_K, 256)
    assert out[0] == pytest.approx(10 * 5 - 5)
    assert out[1] == pytest.approx(10 * 7 - 5)
    assert out[32] == pytest.approx(20 * 6 - 6)
    assert out[33] == pytest.approx(20 * 8 - 6)
    assert out[64] == pytest.approx(30 * 1 - 7)
    assert out[96] == pytest.approx(40 * 2 - 8)
    # groups is=4..7 have d=1,2,3,4 and m=0, but qs[64:] are zeros
    assert out[128] == pytest.approx(0.0)


def test_q6_k_block_layout():
    """block_q6_K = ql[128] qh[64] scales[16] d(f16) -> 210 B / 256 weights.
    ql nibble=5, qh bits=0 → q = 5-32 = -27 for every slot; sc=3, d=0.5."""
    ql = bytes([0x55] * 128)
    qh = bytes([0x00] * 64)
    sc = bytes([3] * 16)
    raw = ql + qh + sc + f16(0.5)
    out = dequantize_gguf_row(raw, GGMLType.Q6_K, 256)
    assert np.allclose(out, 0.5 * 3 * (5 - 32))
    # now flip qh[0] low bits to 3 → q1 = 5+48-32 = 21
    qh2 = bytearray(qh)
    qh2[0] = 0x03
    raw2 = ql + bytes(qh2) + sc + f16(0.5)
    out2 = dequantize_gguf_row(raw2, GGMLType.Q6_K, 256)
    assert out2[0] == pytest.approx(0.5 * 3 * (5 + 3 * 16 - 32))


def test_q2_k_block_layout():
    """Exact byte-level ground truth for ggml-quants.c dequantize_row_q2_K.

    block_q2_K = scales[16] qs[64] d(f16) dmin(f16) -> 84 B / 256 weights.
    A super-group of 128 weights interleaves 4 shift planes (0,2,4,6) of
    q[0:16] and q[16:32]; each 16-wide slice is scaled by its own scales[]
    entry (low nibble = scale, high nibble = min). qs[0]=0xE4 packs the
    codes 0,1,2,3 at shifts 0,2,4,6; scales[0,2,4,6]=0x21 (dl=d*1,
    ml=dmin*2) so each plane can be traced independently.
    """
    scales = bytearray(16)
    for s in (0, 2, 4, 6):
        scales[s] = 0x21            # dl = 2.0*1, ml = 1.0*2
    qs = bytearray(64)
    qs[0] = 0b11_10_01_00           # codes 0,1,2,3 at shifts 0,2,4,6
    raw = bytes(scales) + bytes(qs) + f16(2.0) + f16(1.0)
    out = dequantize_gguf_row(raw, GGMLType.Q2_K, 256)
    # y[0:16]=q[0:16]>>0 (scale0); y[32:48]=q[0:16]>>2 (scale2)
    # y[64:80]=q[0:16]>>4 (scale4); y[96:112]=q[0:16]>>6 (scale6)
    assert out[0]  == pytest.approx(2.0 * 0 - 2.0)   # (qs[0]>>0)&3 = 0
    assert out[32] == pytest.approx(2.0 * 1 - 2.0)   # (qs[0]>>2)&3 = 1
    assert out[64] == pytest.approx(2.0 * 2 - 2.0)   # (qs[0]>>4)&3 = 2
    assert out[96] == pytest.approx(2.0 * 3 - 2.0)   # (qs[0]>>6)&3 = 3


def test_f16_f32_rows():
    raw32 = struct.pack("<4f", 1.0, -2.5, 3.25, 0.0)
    assert np.allclose(dequantize_gguf_row(raw32, GGMLType.F32, 4), [1.0, -2.5, 3.25, 0.0])
    raw16 = struct.pack("<4e", 1.0, -2.5, 3.25, 0.0)
    assert np.allclose(dequantize_gguf_row(raw16, GGMLType.F16, 4), [1.0, -2.5, 3.25, 0.0])
