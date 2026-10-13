"""Transformer forward pass (Llama/Qwen2-family decoder).

Provenance: architecture is the standard post-2023 open-weights decoder —
pre-norm RMSNorm (Zhang & Sennrich 2019), rotary embeddings (Su et al. 2021),
grouped-query attention (Ainslie et al. 2023), SwiGLU FFN (Shazeer 2020) —
i.e. the Llama-2/3 (arXiv:2307.09288) and Qwen2 (arXiv:2407.10671) family
topology. Weight naming follows llama.cpp GGUF conventions so real GGUF
checkpoints load without renaming:

    token_embd.weight, blk.{i}.attn_norm.weight, blk.{i}.attn_q/k/v/output.weight,
    blk.{i}.ffn_norm.weight, blk.{i}.ffn_gate/up/down.weight,
    output_norm.weight, output.weight (optional when tied)
"""
from __future__ import annotations

import numpy as np

from ..core.attention import flash_attention
from ..core.tensor import CausalMask, build_rope_cache, rms_norm, swiglu
from .config import ModelConfig

__all__ = ["Transformer", "build_tiny_model", "MODEL_TENSOR_PREFIX"]


def _w(weights: dict[str, np.ndarray], name: str) -> np.ndarray:
    w = weights.get(name)
    if w is None:
        raise KeyError(f"missing weight: {name}")
    return np.ascontiguousarray(w, dtype=np.float32)


class Transformer:
    """Stateless decoder; all state lives in the caller-managed paged KV cache."""

    def __init__(self, config: ModelConfig, weights: dict[str, np.ndarray]) -> None:
        self.cfg = config
        c = config
        self.tok_emb = _w(weights, "token_embd.weight")
        if self.tok_emb.shape != (c.vocab_size, c.hidden_size):
            raise ValueError(f"token_embd shape {self.tok_emb.shape} != {(c.vocab_size, c.hidden_size)}")
        self.layers = []
        for i in range(c.num_hidden_layers):
            p = f"blk.{i}."
            self.layers.append({
                "attn_norm": _w(weights, p + "attn_norm.weight"),
                "q": _w(weights, p + "attn_q.weight"),
                "k": _w(weights, p + "attn_k.weight"),
                "v": _w(weights, p + "attn_v.weight"),
                "o": _w(weights, p + "attn_output.weight"),
                "ffn_norm": _w(weights, p + "ffn_norm.weight"),
                "gate": _w(weights, p + "ffn_gate.weight"),
                "up": _w(weights, p + "ffn_up.weight"),
                "down": _w(weights, p + "ffn_down.weight"),
            })
        self.output_norm = _w(weights, "output_norm.weight")
        self.output = weights.get("output.weight")
        self.output = np.ascontiguousarray(self.output, dtype=np.float32) if self.output is not None else None
        cos, sin = build_rope_cache(
            c.max_position_embeddings, c.head_dim, c.rope_theta,
            scaling_factor=c.rope_scaling_factor,
            original_max_pos=c.rope_original_max_pos, dtype=np.float32)
        self._cos, self._sin = cos, sin

    # ------------------------------------------------------------------ #
    def forward(self, input_ids: np.ndarray, positions: np.ndarray,
                kv_cache, block_table: list[int], n_past: int) -> np.ndarray:
        """Run the decoder over `input_ids` with absolute `positions`.

        New K/V are written into the paged cache at slots [n_past, n_past+T);
        attention reads the full [0, n_past+T) window through `block_table`.
        Returns logits of shape (T, vocab_size).
        """
        c = self.cfg
        x = self.tok_emb[input_ids]                                   # (T, H)
        seq_len_total = n_past + len(input_ids)

        for li, layer in enumerate(self.layers):
            # ---- attention (weights are (out, in) per GGUF convention) ----
            h = rms_norm(x, layer["attn_norm"], c.rms_norm_eps)
            q = (h @ layer["q"].T).reshape(-1, c.num_attention_heads, c.head_dim)
            k = (h @ layer["k"].T).reshape(-1, c.num_key_value_heads, c.head_dim)
            v = (h @ layer["v"].T).reshape(-1, c.num_key_value_heads, c.head_dim)

            q = self._rope(q, positions)
            k = self._rope(k, positions)

            self._write_kv(kv_cache, block_table, n_past, li, k, v)
            k_all, v_all = self._gather(kv_cache, block_table, seq_len_total, li)

            # (T, h, d) → (1, h, T, d)
            q4 = np.ascontiguousarray(np.transpose(q, (1, 0, 2))[None])
            k4 = np.ascontiguousarray(np.transpose(k_all, (1, 0, 2))[None])
            v4 = np.ascontiguousarray(np.transpose(v_all, (1, 0, 2))[None])
            attn = flash_attention(q4, k4, v4, CausalMask())[0]        # (h, T, d)
            attn = attn.transpose(1, 0, 2).reshape(len(input_ids), -1)
            x = x + attn @ layer["o"].T

            # ---- FFN (SwiGLU) ----
            h = rms_norm(x, layer["ffn_norm"], c.rms_norm_eps)
            x = x + swiglu(h @ layer["gate"].T, h @ layer["up"].T) @ layer["down"].T

        x = rms_norm(x, self.output_norm, c.rms_norm_eps)
        out_w = self.output if self.output is not None else self.tok_emb
        return x @ out_w.T

    def _rope(self, t: np.ndarray, positions: np.ndarray) -> np.ndarray:
        from ..core.tensor import apply_rope
        return apply_rope(t, self._cos, self._sin, positions=positions)

    def _write_kv(self, kv_cache, block_table: list[int], n_past: int,
                  layer: int, k: np.ndarray, v: np.ndarray) -> None:
        """Write this layer's K/V for tokens [n_past, n_past+T) into the pool.

        Layout: pool is per-layer (layer-major) so each layer owns its own
        block chain view without copying. We store k/v stacked per layer by
        writing into per-layer pools provided by `kv_cache.layer_pools(layer)`.
        """
        kv_cache.write_layer(block_table, n_past, layer, k, v)

    def _gather(self, kv_cache, block_table: list[int], n_tokens: int,
                layer: int) -> tuple[np.ndarray, np.ndarray]:
        return kv_cache.read_layer(block_table, n_tokens, layer)


MODEL_TENSOR_PREFIX = "blk"


# --------------------------------------------------------------------------- #
# tiny model builder — deterministic random init for self-contained demos
# --------------------------------------------------------------------------- #

def build_tiny_model(cfg: ModelConfig, seed: int = 42) -> tuple[Transformer, dict[str, np.ndarray]]:
    """Deterministic (seeded) standard-architecture model.

    Used for end-to-end verification without any download. It is NOT a trained
    language model — see docs/LIMITATIONS.md. Structure matches Qwen2.5-0.5B
    so real checkpoints load through the same code path.
    """
    rng = np.random.default_rng(seed)
    c = cfg
    weights: dict[str, np.ndarray] = {}
    std = 1.0 / np.sqrt(c.hidden_size)

    def g(shape, scale=std):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    weights["token_embd.weight"] = g((c.vocab_size, c.hidden_size), std)
    for i in range(c.num_hidden_layers):
        p = f"blk.{i}."
        weights[p + "attn_norm.weight"] = np.ones(c.hidden_size, dtype=np.float32)
        weights[p + "attn_q.weight"] = g((c.num_attention_heads * c.head_dim, c.hidden_size))
        weights[p + "attn_k.weight"] = g((c.num_key_value_heads * c.head_dim, c.hidden_size))
        weights[p + "attn_v.weight"] = g((c.num_key_value_heads * c.head_dim, c.hidden_size))
        weights[p + "attn_output.weight"] = g((c.hidden_size, c.num_attention_heads * c.head_dim))
        weights[p + "ffn_norm.weight"] = np.ones(c.hidden_size, dtype=np.float32)
        weights[p + "ffn_gate.weight"] = g((c.intermediate_size, c.hidden_size))
        weights[p + "ffn_up.weight"] = g((c.intermediate_size, c.hidden_size))
        weights[p + "ffn_down.weight"] = g((c.hidden_size, c.intermediate_size),
                                           1.0 / np.sqrt(c.intermediate_size))
    weights["output_norm.weight"] = np.ones(c.hidden_size, dtype=np.float32)
    weights["output.weight"] = g((c.vocab_size, c.hidden_size), std)
    return Transformer(cfg, weights), weights
