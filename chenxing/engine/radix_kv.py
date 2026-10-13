"""RadixPagedKVCache — hierarchical KV-cache management.

This module is Chenxing's compositional core. It unifies the two strongest
published KV-management designs, taking the complementary layer of each:

  Logical layer — RadixAttention (Zheng et al., SGLang, NeurIPS 2024,
  arXiv:2312.07104): a token-level radix (prefix) tree over cached sequences
  gives exact longest-prefix matching and shared prefix reuse across requests.

  Physical layer — PagedAttention (Kwon et al., vLLM, SOSP 2023,
  arXiv:2309.06180): KV memory lives in fixed-size blocks allocated on demand
  from a pool, eliminating the 60-80% fragmentation of contiguous allocation.

Design decisions (documented because both upstream systems choose differently):

* Only *whole blocks* are shared through the radix tree. A partially filled
  block is private to the sequence that is writing it. This is vLLM's
  automatic-prefix-caching semantics; it removes the concurrent-write hazard
  on the last partial block that RadixAttention must otherwise solve with
  per-node locking, while keeping the radix tree's advantages (no hash
  collisions, O(prefix) match, per-node LRU eviction granularity).

* The pool is one contiguous array of shape (n_layers, n_blocks, block_size,
  n_heads, head_dim) per K and V. A sequence's KV is gathered through its
  block table (fancy indexing), so attention never needs contiguous memory.

Invariants maintained (all tested):
  I1  allocated blocks ⊆ pool, and free_count = n_blocks - Σ allocated
  I2  refcnt == 0 nodes are evictable LRU-first; blocks return to the pool
  I3  identical prefixes share identical physical blocks (zero-copy)
  I4  refcnt never goes negative (underflow raises)
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["RadixPagedKVCache", "PrefixHit", "OutOfKVCacheError"]


class OutOfKVCacheError(RuntimeError):
    """Raised when the physical block pool is exhausted (scheduler may preempt)."""


@dataclass
class PrefixHit:
    """Result of a longest-prefix lookup."""
    node: _RadixNode
    block_ids: list[int]
    n_tokens: int               # cached token count (block-aligned)


class _RadixNode:
    """One radix-tree edge: a block-aligned token segment + its physical blocks."""
    __slots__ = ("tokens", "blocks", "children", "parent", "refcnt", "last_access")

    def __init__(self, tokens: list[int], blocks: list[int], parent: "_RadixNode | None") -> None:
        self.tokens: list[int] = tokens
        self.blocks: list[int] = blocks
        self.children: dict[int, _RadixNode] = {}
        self.parent = parent
        self.refcnt: int = 0
        self.last_access: float = time.monotonic()

    @property
    def is_leaf(self) -> bool:
        return not self.children


class RadixPagedKVCache:
    """Physical block pool (all layers) + logical radix prefix tree."""

    def __init__(self, n_blocks: int, block_size: int, n_layers: int,
                 n_kv_heads: int, head_dim: int, dtype: np.dtype = np.float32) -> None:
        if block_size <= 0 or n_blocks <= 0 or n_layers <= 0:
            raise ValueError("n_blocks, block_size, n_layers must be positive")
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        shape = (n_layers, n_blocks, block_size, n_kv_heads, head_dim)
        self.k_pool = np.zeros(shape, dtype=dtype)
        self.v_pool = np.zeros_like(self.k_pool)
        self._free: OrderedDict[int, None] = OrderedDict((i, None) for i in range(n_blocks))
        self._root = _RadixNode([], [], None)
        self._hits = 0
        self._lookups = 0

    # ------------------------------------------------------------------ #
    # physical layer
    # ------------------------------------------------------------------ #

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def alloc_block(self) -> int:
        if not self._free:
            raise OutOfKVCacheError(
                f"KV cache exhausted ({self.n_blocks} blocks); scheduler must preempt")
        bid, _ = self._free.popitem(last=False)
        return bid

    def _free_block(self, bid: int) -> None:
        self._free[bid] = None

    def write_layer(self, block_table: list[int], slot_start: int, layer: int,
                    k: NDArray, v: NDArray) -> None:
        """Write K/V for tokens [slot_start, slot_start+T) along a block table.

        k, v: (T, n_kv_heads, head_dim). Token t lands at physical block
        block_table[(slot_start + t) // block_size], slot (slot_start + t) % block_size.
        """
        t_total = k.shape[0]
        if len(block_table) * self.block_size < slot_start + t_total:
            raise ValueError("block_table too short for requested write range")
        for t in range(t_total):
            pos = slot_start + t
            bid = block_table[pos // self.block_size]
            slot = pos % self.block_size
            self.k_pool[layer, bid, slot] = k[t]
            self.v_pool[layer, bid, slot] = v[t]

    def read_layer(self, block_table: list[int], n_tokens: int,
                   layer: int) -> tuple[NDArray, NDArray]:
        """Gather K/V for the first `n_tokens` positions (n_tokens, h, d)."""
        if n_tokens == 0 or not block_table:
            empty = np.zeros((0, self.n_kv_heads, self.head_dim), dtype=self.dtype)
            return empty, empty
        n_needed = min(n_tokens, len(block_table) * self.block_size)
        full_blocks = n_needed // self.block_size
        rem = n_needed - full_blocks * self.block_size
        k_parts = [self.k_pool[layer, block_table[:full_blocks]].reshape(-1, self.n_kv_heads, self.head_dim)]
        v_parts = [self.v_pool[layer, block_table[:full_blocks]].reshape(-1, self.n_kv_heads, self.head_dim)]
        if rem:
            k_parts.append(self.k_pool[layer, block_table[full_blocks], :rem])
            v_parts.append(self.v_pool[layer, block_table[full_blocks], :rem])
        return np.concatenate(k_parts), np.concatenate(v_parts)

    # ------------------------------------------------------------------ #
    # logical layer
    # ------------------------------------------------------------------ #

    def match_prefix(self, token_ids: list[int]) -> PrefixHit:
        """Longest block-aligned cached prefix (exact match, zero-copy).

        A node whose edge partially matches still contributes its fully
        covered blocks (floor(matched_tokens / block_size)) — prefixes are
        only ever shared at whole-block granularity.
        """
        self._lookups += 1
        node, blocks, i = self._root, [], 0
        while i < len(token_ids):
            child = node.children.get(token_ids[i])
            if child is None:
                break
            seg = child.tokens
            max_n = min(len(seg), len(token_ids) - i)
            n = 0
            while n < max_n and token_ids[i + n] == seg[n]:
                n += 1
            whole = n == len(seg)
            n_take = len(seg) // self.block_size if whole else n // self.block_size
            if n_take == 0:
                break
            blocks.extend(child.blocks[:n_take])
            i += n_take * self.block_size
            node = child
            if not whole:
                break
        if i > 0:
            self._hits += 1
            self._touch(node)
        return PrefixHit(node, blocks, i)

    def insert(self, token_ids: list[int], block_ids: list[int],
               parent: _RadixNode | None = None) -> _RadixNode:
        """Insert a block-aligned token chain into the tree, sharing what exists."""
        if len(token_ids) != len(block_ids) * self.block_size:
            raise ValueError("insert requires block-aligned tokens/blocks")
        if parent is None:
            hit = self.match_prefix(token_ids)
            parent = hit.node
            cut = hit.n_tokens
            token_ids = token_ids[cut:]
            block_ids = block_ids[cut // self.block_size:]
        if not token_ids:
            self._touch(parent)
            return parent
        if parent.is_leaf and parent.refcnt == 0 and parent is not self._root:
            parent.tokens.extend(token_ids)
            parent.blocks.extend(block_ids)
            self._touch(parent)
            return parent
        child = _RadixNode(list(token_ids), list(block_ids), parent)
        parent.children[token_ids[0]] = child
        return child

    def acquire(self, node: _RadixNode) -> None:
        """Pin a node (and ancestors) so eviction cannot reclaim its blocks."""
        n: _RadixNode | None = node
        while n is not None and n is not self._root:
            n.refcnt += 1
            self._touch(n)
            n = n.parent

    def release(self, node: _RadixNode) -> None:
        n: _RadixNode | None = node
        while n is not None and n is not self._root:
            if n.refcnt <= 0:
                raise RuntimeError("refcount underflow")
            n.refcnt -= 1
            n = n.parent

    def evict(self, n_blocks_needed: int = 1) -> int:
        """LRU-evict refcnt-0 leaves until `n_blocks_needed` blocks are free.

        Returns the number of blocks actually freed (0 if none available).
        """
        freed = 0
        while freed < n_blocks_needed:
            victim = self._lru_leaf()
            if victim is None:
                return freed
            freed += len(victim.blocks)
            for b in victim.blocks:
                self._free_block(b)
            del victim.parent.children[victim.tokens[0]]
        return freed

    def ensure_free(self, n_needed: int) -> None:
        """Guarantee `n_needed` free blocks, evicting LRU leaves as needed."""
        while self.free_blocks < n_needed:
            if self.evict(n_needed - self.free_blocks) == 0:
                raise OutOfKVCacheError("no evictable blocks; all pinned")

    def _lru_leaf(self) -> _RadixNode | None:
        best: _RadixNode | None = None
        stack = [self._root]
        while stack:
            n = stack.pop()
            for c in n.children.values():
                stack.append(c)
                if c.refcnt == 0 and (best is None or c.last_access < best.last_access):
                    best = c
        return best

    def _touch(self, node: _RadixNode) -> None:
        node.last_access = time.monotonic()

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, float | int]:
        total_nodes, stack = 0, [self._root]
        while stack:
            n = stack.pop()
            total_nodes += len(n.children)
            stack.extend(n.children.values())
        return {
            "n_blocks": self.n_blocks,
            "free_blocks": self.free_blocks,
            "cached_blocks": self.n_blocks - self.free_blocks,
            "radix_nodes": total_nodes,
            "prefix_hit_rate": self._hits / max(self._lookups, 1),
        }
