"""Model configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Architecture hyperparameters (Llama/Qwen2-family layout).

    Defaults mirror Qwen2.5-0.5B so the bundled tiny demo model is
    structurally identical to a real downloadable checkpoint.
    """
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2          # GQA (Ainslie et al. 2023)
    head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    rope_scaling_factor: float = 1.0
    rope_original_max_pos: int = 4096
    tie_word_embeddings: bool = True
    eos_token_id: int = 151645
    bos_token_id: int = 151643

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            # allow explicit head_dim override (e.g. Qwen3 partial-RoPE styles)
            self.head_dim = self.hidden_size // self.num_attention_heads

    @property
    def num_gqa_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def scaled(self, scale: float) -> "ModelConfig":
        """A proportionally scaled-down variant (used to build draft/tiny models)."""
        import copy
        c = copy.deepcopy(self)
        c.hidden_size = max(16, int(self.hidden_size * scale) // 8 * 8)
        c.intermediate_size = max(16, int(self.intermediate_size * scale) // 8 * 8)
        c.num_hidden_layers = max(1, int(self.num_hidden_layers * scale))
        c.vocab_size = self.vocab_size
        c.head_dim = c.hidden_size // c.num_attention_heads
        return c
