"""Sampler invariants: determinism, truncation, penalties, greedy."""
import numpy as np

from chenxing.core.sampler import Sampler, SamplingParams


def test_greedy_is_argmax():
    s = Sampler(8)
    logits = np.array([0.1, 5.0, 0.2, 0.0, -1, 0, 0, 0])
    assert s.sample(logits, SamplingParams(temperature=0.0)) == 1


def test_temperature_zero_ignores_seed():
    s = Sampler(4)
    logits = np.array([1.0, 3.0, 2.0, 0.0])
    assert all(s.sample(logits, SamplingParams(temperature=0.0)) == 1 for _ in range(5))


def test_seed_reproducible():
    s = Sampler(16)
    p = SamplingParams(temperature=1.0, seed=99)
    rng = np.random.default_rng(0)
    a = [s.sample(rng.standard_normal(16), p) for _ in range(4)]
    rng = np.random.default_rng(0)
    b = [s.sample(rng.standard_normal(16), p) for _ in range(4)]
    assert a == b


def test_top_k_zeroes_tail():
    s = Sampler(8)
    logits = np.array([5.0, 4.9, 0.1, 0.0, 0, 0, 0, 0])
    p = SamplingParams(temperature=1.0, top_k=1, seed=0)
    assert all(s.sample(logits, p) in (0, 1) for _ in range(10))


def test_top_p_mass_cutoff():
    s = Sampler(4)
    # distribution: [0.9, 0.09, 0.009, 0.001]
    logits = np.log(np.array([0.9, 0.09, 0.009, 0.001]))
    p = SamplingParams(temperature=1.0, top_p=0.95, seed=0)
    probs = s._truncate(np.exp(logits - logits.max()), p)
    probs = probs / probs.sum()
    assert probs[2] == 0 and probs[3] == 0      # 0.909 cumulative → tail cut
    assert probs[0] + probs[1] > 0.99


def test_repetition_penalty_downweights_seen():
    s = Sampler(4)
    logits = np.array([4.0, 4.0, 0.0, 0.0])
    p = SamplingParams(temperature=0.0, repetition_penalty=10.0)
    penalized = s.apply_penalties(logits.copy(), p, generated_ids=[0])
    assert penalized[0] < penalized[1]


def test_min_p_relative_floor():
    s = Sampler(4)
    logits = np.log(np.array([0.8, 0.15, 0.04, 0.01]))
    p = SamplingParams(temperature=1.0, min_p=0.15, seed=0)   # floor = 0.15*0.8 = 0.12
    probs = s._truncate(np.exp(logits - logits.max()), p)
    assert probs[3] == 0 and probs[2] == 0        # 0.04, 0.01 < 0.12
    assert probs[0] > 0 and probs[1] > 0          # 0.8, 0.15 ≥ 0.12


def test_invalid_params_raise():
    import pytest
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1)
