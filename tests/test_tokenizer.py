"""Tokenizer: byte-level BPE + offline fallback."""
import numpy as np

from chenxing.model.tokenizer import BPETokenizer, ByteTokenizer


def test_byte_tokenizer_roundtrip():
    t = ByteTokenizer()
    ids = t.encode("hello 世界")
    assert t.decode(ids) == "hello 世界"


def test_bpe_greedy_merge():
    vocab = {"<|endoftext|>": 0, "h": 1, "e": 2, "l": 3, "o": 4, "hello": 5, "lo": 6}
    merges = [("l", "o"), ("he", "llo"), ("h", "e"), ("hell", "o")]
    t = BPETokenizer(vocab, merges, {"<|endoftext|>": 0})
    ids = t.encode("hello")
    assert t.decode(ids) == "hello"


def test_bpe_unknown_falls_back_to_chars():
    vocab = {"a": 1, "b": 2}
    t = BPETokenizer(vocab, [], {})
    ids = t.encode("ab")
    assert t.decode(ids) == "ab"


def test_bpe_roundtrip_ascii():
    """Full byte-level roundtrip for printable ASCII."""
    from chenxing.model.tokenizer import _bytes_to_unicode
    b2u = _bytes_to_unicode()
    vocab = {u: i for i, u in enumerate(b2u.values())}
    t = BPETokenizer(vocab, [], {})
    text = "The quick brown fox!"
    ids = t.encode(text)
    assert t.decode(ids) == text
