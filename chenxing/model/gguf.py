"""GGUF container reader/writer (subset: everything needed to run real models).

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
llama.cpp GGUF (MIT). We implement the full metadata type system, tensor
directory, aligned data section, and reading of F32/F16/Q8_0/Q2_K/Q4_K/Q6_K
tensors via chenxing.core.quantization.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.quantization import GGMLType, dequantize_gguf_row

__all__ = ["GGUFFile", "GGUFTensorInfo", "GGUFType", "read_gguf", "write_gguf"]

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGUFType = {
    "u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5,
    "f32": 6, "bool": 7, "string": 8, "array": 9, "u64": 10, "i64": 11, "f64": 12,
}
_SCALAR_FMT = {  # type -> struct format
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}


@dataclass
class GGUFTensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int


@dataclass
class GGUFFile:
    version: int
    metadata: dict[str, object]
    tensors: dict[str, GGUFTensorInfo]
    _path: str
    _data_start: int
    _alignment: int

    def get(self, key: str, default: object = None) -> object:
        return self.metadata.get(key, default)

    def read_tensor(self, name: str) -> tuple[np.ndarray, int]:
        """Return (dequantized fp32 data reshaped to logical dims, ggml_type)."""
        info = self.tensors[name]
        with open(self._path, "rb") as f:
            f.seek(self._data_start + info.offset)
            n_elems = int(np.prod(info.dims)) if info.dims else 1
            from ..core.quantization import GGMLType as T
            _, blk_bytes = T.BLOCK_SIZES[info.ggml_type]
            blk_elems, _ = T.BLOCK_SIZES[info.ggml_type]
            size = (n_elems // blk_elems) * blk_bytes
            raw = f.read(size)
        data = dequantize_gguf_row(raw, info.ggml_type, n_elems)
        # GGUF stores dims in reverse (fastest-varying first)
        return data.reshape(list(reversed(info.dims))), info.ggml_type


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def _read_value(f):
    (t,) = struct.unpack("<I", f.read(4))
    if t == GGUFType["string"]:
        return _read_string(f)
    if t == GGUFType["array"]:
        (elem_t, n) = struct.unpack("<IQ", f.read(12))
        if elem_t == GGUFType["string"]:
            return [_read_string(f) for _ in range(n)]
        fmt = _SCALAR_FMT[elem_t]
        return list(struct.unpack(f"<{fmt[1:] * n}", f.read(struct.calcsize(fmt) * n))) if n else []
        # note: bulk unpack, returns python list
    fmt = _SCALAR_FMT[t]
    (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    return v


def read_gguf(path: str | Path) -> GGUFFile:
    p = Path(path)
    with open(p, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{p} is not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            metadata[key] = _read_value(f)
        tensors: dict[str, GGUFTensorInfo] = {}
        offset = 0
        alignment = int(metadata.get("general.alignment", 32))
        for _ in range(n_tensors):
            name = _read_string(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{'Q' * n_dims}", f.read(8 * n_dims)))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors[name] = GGUFTensorInfo(name, dims, ggml_type, off)
            from ..core.quantization import GGMLType as T
            blk_elems, blk_bytes = T.BLOCK_SIZES[ggml_type]
            offset = max(offset, off + (int(np.prod(dims)) // blk_elems) * blk_bytes)
        data_start = (f.tell() + alignment - 1) // alignment * alignment
        # tensor offsets are relative to the (aligned) data-section start;
        # the writer aligns identically, so no further adjustment is needed.
    return GGUFFile(version, metadata, tensors, str(p), data_start, alignment)


# --------------------------------------------------------------------------- #
# writer (F16/F32/Q8_0) — enough to serialize models for round-trip tests
# --------------------------------------------------------------------------- #

def _pack_value(v: object) -> bytes:
    if isinstance(v, str):
        b = v.encode("utf-8")
        return struct.pack("<I", GGUFType["string"]) + struct.pack("<Q", len(b)) + b
    if isinstance(v, bool):
        return struct.pack("<I", GGUFType["bool"]) + struct.pack("<?", v)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
        out = struct.pack("<I", GGUFType["array"]) + struct.pack("<IQ", GGUFType["string"], len(v))
        for s in v:
            b = s.encode("utf-8")
            out += struct.pack("<Q", len(b)) + b
        return out
    if isinstance(v, (list, tuple)):
        t = GGUFType["f32"] if any(isinstance(x, float) for x in v) else GGUFType["i32"]
        fmt = _SCALAR_FMT[t]
        return (struct.pack("<I", GGUFType["array"]) + struct.pack("<IQ", t, len(v))
                + struct.pack(f"<{fmt[1:] * len(v)}", *v))
    if isinstance(v, float):
        return struct.pack("<I", GGUFType["f32"]) + struct.pack("<f", v)
    if isinstance(v, int):
        return struct.pack("<I", GGUFType["i32"]) + struct.pack("<i", v)
    raise TypeError(f"cannot pack {type(v)}")


def _quantize_q8_0(w: np.ndarray) -> bytes:
    """Pack fp32 → block_q8_0 (34 B per 32 weights)."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    pad = (-w.size) % 32
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    blocks = w.reshape(-1, 32)
    d = np.abs(blocks).max(axis=1, keepdims=True) / 127.0
    d = np.where(d == 0, np.float32(1.0), d).astype("<f2")
    q = np.rint(blocks / d.astype(np.float32)).clip(-127, 127).astype(np.int8)
    out = bytearray()
    for i in range(blocks.shape[0]):
        out += d[i].tobytes() + q[i].tobytes()
    return bytes(out)


def write_gguf(path: str | Path, metadata: dict[str, object],
               tensors: dict[str, np.ndarray], quantize_q8_0: bool = False) -> None:
    """Write tensors (name -> fp32 array, 2-D (out, in)) as a GGUF file."""
    ttype = GGMLType.Q8_0 if quantize_q8_0 else GGMLType.F32
    with open(path, "wb") as f:
        f.write(GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata)))
        for k, v in metadata.items():
            f.write(struct.pack("<Q", len(k.encode()))) ; f.write(k.encode()) ; f.write(_pack_value(v))
        meta = dict(metadata)
        meta.setdefault("general.alignment", 32)
        alignment = int(meta["general.alignment"])
        offset, infos, blobs = 0, [], []
        for name, w in tensors.items():
            w = np.ascontiguousarray(np.asarray(w, dtype=np.float32))
            blob = _quantize_q8_0(w) if quantize_q8_0 else w.astype("<f4").tobytes()
            pad = (-len(blob)) % alignment
            blob += b"\x00" * pad
            # dims stored reversed (fastest-varying first)
            dims = [w.shape[1], w.shape[0]] if w.ndim == 2 else [w.shape[0]]
            infos.append((name, dims, ttype, offset))
            blobs.append(blob)
            offset += len(blob)
        data_start_hint = 0
        for name, dims, t, off in infos:
            nb = name.encode()
            f.write(struct.pack("<Q", len(nb)) + nb)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack(f"<{'Q' * len(dims)}", *dims))
            f.write(struct.pack("<IQ", t, off))
        cur = f.tell()
        aligned = (cur + alignment - 1) // alignment * alignment
        f.write(b"\x00" * (aligned - cur))
        for blob in blobs:
            f.write(blob)
    # rewrite offsets? offsets are relative to data section start which equals
    # `aligned`; we already stored absolute-relative offsets from data_start.
    return
