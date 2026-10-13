"""Tokenizer: byte-level BPE compatible with GPT-2 / Llama / Qwen tokenizers.

Provenance:

- Byte-level BPE          : Radford et al., GPT-2, 2019 — the standard
  byte-to-unicode trick (Ġ = space) used by all modern Llama/Qwen/Mistral GGUFs.
- BPE merge ranking       : Sennrich et al., "Neural Machine Translation of Rare
  Words with Subword Units", ACL 2016 (arXiv:1508.07909).

Loads vocabulary + merges either from a GGUF file (tokenizer.ggml.tokens /
tokenizer.ggml.merges) or a HuggingFace tokenizer.json; falls back to a
built-in byte-level identity tokenizer so the stack is always runnable
offline (documented limitation: fallback is character-per-byte, high perplexity).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["BPETokenizer", "ByteTokenizer"]


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte↔unicode table (printable bytes map to themselves)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class BPETokenizer:
    """Byte-level BPE. Works with vocab/merges from GGUF or tokenizer.json."""

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]],
                 special_tokens: dict[str, int] | None = None) -> None:
        self.vocab = dict(vocab)
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        self.merge_ranks: dict[tuple[str, str], int] = {m: r for r, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.special_tokens = special_tokens or {}
        self.eos_token_id = self.special_tokens.get("<|im_end|>",
                                                    self.special_tokens.get("</s>",
                                                    self.special_tokens.get("<|endoftext|>", -1)))
        self.bos_token_id = self.special_tokens.get("<|im_start|>",
                                                    self.special_tokens.get("<s>", -1))

    # ------------------------------------------------------------------ #
    @classmethod
    def from_gguf(cls, gguf) -> "BPETokenizer":
        vocab_list = gguf.get("tokenizer.ggml.tokens")
        merges_raw = gguf.get("tokenizer.ggml.merges") or []
        scores = gguf.get("tokenizer.ggml.scores")
        if not vocab_list:
            raise ValueError("GGUF has no tokenizer.ggml.tokens")
        vocab = {t: i for i, t in enumerate(vocab_list)}
        merges = []
        for m in merges_raw:
            parts = m.split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        special = {}
        for i, t in enumerate(vocab_list):
            if scores is not None and scores[i] < 0 or t.startswith("<|") and t.endswith("|>") \
               or t in ("<s>", "</s>", "<unk>", "<|endoftext|>"):
                special.setdefault(t, i)
        return cls(vocab, merges, special)

    @classmethod
    def from_tokenizer_json(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = data["model"]["vocab"]
        merges = []
        for m in data["model"]["merges"]:
            parts = m.split(" ")
            merges.append((parts[0], parts[1]) if len(parts) == 2 else (parts[0], parts[0]))
        special = {t: i for t, i in data.get("added_tokens", {}).items()} if isinstance(
            data.get("added_tokens"), dict) else {
            t["content"]: t["id"] for t in data.get("added_tokens", [])}
        return cls(vocab, merges, special)

    # ------------------------------------------------------------------ #
    def _bpe(self, token: str) -> list[str]:
        word = list(token)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.merge_ranks.get(p, 1 << 30))
            if best not in self.merge_ranks:
                break
            first, second = best
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        return word

    def encode(self, text: str) -> list[int]:
        """GPT-2 style: bytes → unicode chars → greedy BPE merges."""
        mapped = "".join(self.byte_encoder[b] for b in text.encode("utf-8"))
        ids: list[int] = []
        # simple whitespace-aware pre-split mirroring GPT-2's leading-space rule
        segments: list[str] = []
        buf = ""
        for ch in mapped:
            buf += ch
            if ch == " " and len(buf) > 1:
                segments.append(buf[:-1])
                buf = " "
        if buf:
            segments.append(buf)
        for seg in segments:
            for piece in self._bpe(seg):
                tid = self.vocab.get(piece)
                if tid is None:
                    for ch in piece:
                        tid2 = self.vocab.get(ch)
                        if tid2 is not None:
                            ids.append(tid2)
                else:
                    ids.append(tid)
        return ids

    def decode(self, ids: list[int] | np.ndarray) -> str:
        text = "".join(self.id_to_token.get(int(i), "") for i in ids)
        raw = bytes(self.byte_decoder.get(ch, ord("?")) for ch in text)
        return raw.decode("utf-8", errors="replace")


class ByteTokenizer:
    """Offline fallback: one id per byte + control specials.

    Guarantees the full stack (train tiny model → serve → agent) runs with zero
    downloads. Trade-off documented in docs/LIMITATIONS.md.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 256
        self.bos_token_id = 257

    def encode(self, text: str) -> list[int]:
        return [b for b in text.encode("utf-8")]

    def decode(self, ids: list[int] | np.ndarray) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i < 256:
                out.append(i)
        return out.decode("utf-8", errors="replace")
