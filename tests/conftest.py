"""Shared fixtures: tiny deterministic models + engines."""
import numpy as np
import pytest

from chenxing.core.sampler import SamplingParams
from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model

VOCAB = 264


@pytest.fixture(scope="session")
def tiny_cfg() -> ModelConfig:
    return ModelConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2)


@pytest.fixture(scope="session")
def tiny_engine(tiny_cfg) -> LLMEngine:
    model, _ = build_tiny_model(tiny_cfg, seed=42)
    return LLMEngine(model, None, block_size=8, n_blocks=64)


@pytest.fixture(scope="session")
def draft_engine(tiny_cfg) -> LLMEngine:
    model, _ = build_tiny_model(tiny_cfg.scaled(0.5), seed=99)
    return LLMEngine(model, None, block_size=8, n_blocks=64)


@pytest.fixture()
def sample_prompt() -> list[int]:
    return list(map(int, np.random.default_rng(1).integers(0, 256, 10)))


@pytest.fixture()
def greedy() -> SamplingParams:
    return SamplingParams(max_tokens=8, temperature=0.0, seed=0)
