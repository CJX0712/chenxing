"""RadixPagedKVCache invariants (I1–I4 from the module docstring)."""
import pytest

from chenxing.engine.radix_kv import (OutOfKVCacheError, RadixPagedKVCache)


def make_cache(n_blocks=8, block_size=4):
    return RadixPagedKVCache(n_blocks=n_blocks, block_size=block_size,
                             n_layers=1, n_kv_heads=2, head_dim=4)


def test_block_pool_bounds_and_free_accounting():
    c = make_cache(n_blocks=4)
    ids = [c.alloc_block() for _ in range(4)]
    assert len(set(ids)) == 4 and c.free_blocks == 0
    with pytest.raises(OutOfKVCacheError):
        c.alloc_block()
    c._free_block(ids[0])
    assert c.free_blocks == 1


def test_prefix_match_zero_copy_share():
    """I3: identical prefixes must share identical physical blocks."""
    c = make_cache()
    toks = [1, 2, 3, 4, 5, 6, 7, 8]
    node = c.insert(toks, [c.alloc_block(), c.alloc_block()])
    hit = c.match_prefix(toks)
    assert hit.n_tokens == 8
    assert hit.block_ids == node.blocks
    hit2 = c.match_prefix(toks[:4] + [99, 99, 99, 99])
    assert hit2.n_tokens == 4
    assert hit2.block_ids == node.blocks[:1]


def test_partial_prefix_does_not_overmatch():
    c = make_cache()
    c.insert([1, 2, 3, 4], [c.alloc_block()])
    hit = c.match_prefix([1, 2, 9, 9])
    assert hit.n_tokens == 0


def test_lru_eviction_returns_blocks():
    c = make_cache(n_blocks=4)
    n1 = c.insert([1, 2, 3, 4], [c.alloc_block()])
    n2 = c.insert([9, 9, 9, 9], [c.alloc_block()])
    assert c.free_blocks == 2
    c.evict(2)
    assert c.free_blocks == 4
    assert c.match_prefix([1, 2, 3, 4]).n_tokens == 0


def test_pinned_nodes_survive_eviction():
    c = make_cache(n_blocks=4)
    n1 = c.insert([1, 2, 3, 4], [c.alloc_block()])
    n2 = c.insert([9, 9, 9, 9], [c.alloc_block()])
    c.acquire(n1)
    freed = c.evict(2)
    assert freed == 1                                    # only n2 was evictable
    assert c.match_prefix([1, 2, 3, 4]).n_tokens == 4
    c.release(n1)
    c.evict(2)
    assert c.match_prefix([1, 2, 3, 4]).n_tokens == 0


def test_refcount_underflow_raises():
    c = make_cache()
    n = c.insert([1, 2, 3, 4], [c.alloc_block()])
    with pytest.raises(RuntimeError):
        c.release(n)


def test_acquire_pins_whole_chain():
    c = make_cache(block_size=2)
    mid = c.insert([1, 2], [c.alloc_block()])
    leaf = c.insert([3, 4], [c.alloc_block()], parent=mid)
    c.acquire(leaf)
    assert mid.refcnt == 1 and leaf.refcnt == 1
    c.release(leaf)
    assert mid.refcnt == 0 and leaf.refcnt == 0


def test_write_then_read_roundtrip():
    c = make_cache(n_blocks=4, block_size=2)
    bid = c.alloc_block()
    rng = np.random.default_rng(0)
    k = rng.standard_normal((2, 2, 4)).astype(np.float32)
    v = rng.standard_normal((2, 2, 4)).astype(np.float32)
    c.write_layer([bid], 0, 0, k, v)
    kr, vr = c.read_layer([bid], 2, 0)
    assert np.allclose(kr, k) and np.allclose(vr, v)


def test_kv_write_beyond_table_raises():
    c = make_cache(block_size=2)
    with pytest.raises(ValueError):
        c.write_layer([0], 0, 0, np.zeros((4, 2, 4)), np.zeros((4, 2, 4)))


def test_insert_requires_block_alignment():
    c = make_cache()
    with pytest.raises(ValueError):
        c.insert([1, 2, 3], [c.alloc_block()])


import numpy as np  # noqa: E402  (used above)
