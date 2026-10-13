"""Typed state-graph agent orchestration.

Provenance: the graph-of-nodes execution model follows LangGraph
(LangChain Inc., 2024, github.com/langchain-ai/langgraph) — typed shared
state, nodes as pure state→state transforms, conditional edges for routing,
and a compiled runnable. We implement the core semantics in ~200 lines with
full type hints; the motivation is a minimal, auditable orchestration core
that any ReAct/planner/multi-agent pattern can be built on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

__all__ = ["StateGraph", "GraphCompileError", "END"]

END = "__end__"


class GraphCompileError(RuntimeError):
    pass


Reducer = Callable[[Any, Any], Any]


@dataclass
class _Node:
    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class _Edge:
    src: str
    dst: str                      # node name or END


@dataclass
class _CondEdge:
    src: str
    router: Callable[[dict[str, Any]], str]
    options: dict[str, str] = field(default_factory=dict)  # return value -> dst


class StateGraph:
    """Compile-and-run graph over a typed shared state dict.

    Nodes return *partial* state updates (LangGraph semantics); a node may
    also return the full dict. Unknown keys are merged, existing keys are
    overwritten unless a reducer is registered for them.
    """

    def __init__(self, state_schema: type[TypedDict] | None = None) -> None:
        self.state_schema = state_schema
        self._nodes: dict[str, _Node] = {}
        self._edges: list[_Edge] = []
        self._cond: list[_CondEdge] = []
        self._reducers: dict[str, Reducer] = {}
        self._entry: str | None = None

    # ------------------------------------------------------------------ #
    def add_node(self, name: str, fn: Callable[[dict], dict]) -> "StateGraph":
        if name in self._nodes:
            raise GraphCompileError(f"duplicate node {name!r}")
        self._nodes[name] = _Node(name, fn)
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        if src != "__start__" and src not in self._nodes:
            raise GraphCompileError(f"unknown source node {src!r}")
        if dst != END and dst not in self._nodes:
            raise GraphCompileError(f"unknown destination node {dst!r}")
        self._edges.append(_Edge(src, dst))
        if src == "__start__":
            self._entry = dst
        return self

    def add_conditional_edges(self, src: str, router: Callable[[dict], str],
                              options: dict[str, str]) -> "StateGraph":
        if src not in self._nodes:
            raise GraphCompileError(f"unknown source node {src!r}")
        for ret, dst in options.items():
            if dst != END and dst not in self._nodes:
                raise GraphCompileError(f"unknown option target {dst!r}")
        self._cond.append(_CondEdge(src, router, options))
        return self

    def set_reducer(self, key: str, reducer: Reducer) -> "StateGraph":
        """Reducer(key, existing, incoming) -> merged — e.g. list append."""
        self._reducers[key] = reducer
        return self

    # ------------------------------------------------------------------ #
    def compile(self, *, max_steps: int = 64) -> "CompiledGraph":
        if self._entry is None:
            raise GraphCompileError("no entry edge: add_edge('__start__', entry)")
        for n in self._nodes.values():
            has_out = any(e.src == n.name for e in self._edges) or \
                any(c.src == n.name for c in self._cond)
            if not has_out:
                raise GraphCompileError(f"node {n.name!r} has no outgoing edge")
        return CompiledGraph(self, max_steps=max_steps)


class CompiledGraph:
    def __init__(self, graph: StateGraph, *, max_steps: int) -> None:
        self.g = graph
        self.max_steps = max_steps

    def invoke(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = dict(state or {})
        current = self.g._entry
        steps = 0
        while current != END:
            if steps >= self.max_steps:
                raise GraphCompileError(f"exceeded max_steps={self.max_steps} (cycle?)")
            node = self.g._nodes[current]
            update = node.fn(state) or {}
            for k, v in update.items():
                red = self.g._reducers.get(k)
                if red is not None and k in state:
                    state[k] = red(state[k], v)
                else:
                    state[k] = v
            steps += 1
            nxt = self._route(current, state)
            current = nxt
        return state

    def _route(self, node_name: str, state: dict) -> str:
        for c in self.g._cond:
            if c.src == node_name:
                ret = c.router(state)
                if ret not in c.options:
                    raise GraphCompileError(
                        f"router at {node_name!r} returned {ret!r}; options {list(c.options)}")
                return c.options[ret]
        for e in self.g._edges:
            if e.src == node_name:
                return e.dst
        return END
