"""GGUF container round-trip and tensor decode."""
import numpy as np
import pytest

from chenxing.model.gguf import read_gguf, write_gguf


def test_roundtrip_f32(tmp_path):
    w = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
    path = tmp_path / "m.gguf"
    write_gguf(path, {"general.name": "t", "general.alignment": 32}, {"a.weight": w})
    f = read_gguf(path)
    assert f.get("general.name") == "t"
    got, ttype = f.read_tensor("a.weight")
    assert ttype == 0 and np.allclose(got, w)


def test_roundtrip_q8_0(tmp_path):
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((4, 64)) * 3).astype(np.float32)
    path = tmp_path / "q.gguf"
    write_gguf(path, {"general.alignment": 32}, {"b.weight": w}, quantize_q8_0=True)
    f = read_gguf(path)
    got, ttype = f.read_tensor("b.weight")
    assert ttype == 8
    err = np.abs(got - w).max() / np.abs(w).max()
    assert err < 0.02          # int8 absmax noise


def test_metadata_scalars_and_arrays(tmp_path):
    path = tmp_path / "md.gguf"
    write_gguf(path, {"n_ctx": 512, "f_alpha": 1.5, "tokens": ["a", "b"],
                      "flag": True}, {"w": np.zeros((2, 2), np.float32)})
    f = read_gguf(path)
    assert f.get("n_ctx") == 512 and f.get("f_alpha") == 1.5
    assert f.get("tokens") == ["a", "b"] and f.get("flag") is True


def test_rejects_non_gguf(tmp_path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError):
        read_gguf(path)
