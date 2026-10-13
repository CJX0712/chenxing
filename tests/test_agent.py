"""Agent orchestration: graph semantics + tool registry + ReAct loop."""
import pytest

from chenxing.agent import END, GraphCompileError, StateGraph, ToolRegistry, react_loop


class S(dict):
    pass


def test_linear_graph_executes_in_order():
    g = StateGraph()
    order: list[str] = []
    g.add_node("a", lambda s: (order.append("a"), {"v": s.get("v", 0) + 1})[1])
    g.add_node("b", lambda s: (order.append("b"), {"v": s["v"] + 10})[1])
    g.add_edge("__start__", "a").add_edge("a", "b").add_edge("b", END)
    out = g.compile().invoke({"v": 0})
    assert out["v"] == 11 and order == ["a", "b"]


def test_conditional_routing():
    g = StateGraph()
    g.add_node("check", lambda s: {"n": s["n"]})
    g.add_node("big", lambda s: {"path": "big"})
    g.add_node("small", lambda s: {"path": "small"})
    g.add_edge("__start__", "check")
    g.add_conditional_edges("check", lambda s: "b" if s["n"] > 10 else "s",
                            {"b": "big", "s": "small"})
    g.add_edge("big", END).add_edge("small", END)
    assert g.compile().invoke({"n": 99})["path"] == "big"
    assert g.compile().invoke({"n": 1})["path"] == "small"


def test_reducer_merges_lists():
    g = StateGraph()
    g.add_node("a", lambda s: {"log": ["a"]})
    g.add_node("b", lambda s: {"log": ["b"]})
    g.set_reducer("log", lambda old, new: old + new)
    g.add_edge("__start__", "a").add_edge("a", "b").add_edge("b", END)
    assert g.compile().invoke({"log": []})["log"] == ["a", "b"]


def test_cycle_detected():
    g = StateGraph()
    g.add_node("a", lambda s: {})
    g.add_node("b", lambda s: {})
    g.add_edge("__start__", "a").add_edge("a", "b").add_edge("b", "a")
    with pytest.raises(GraphCompileError):
        g.compile().invoke({})


def test_unknown_edge_target_rejected():
    g = StateGraph()
    g.add_node("a", lambda s: {})
    g.add_edge("__start__", "a")
    with pytest.raises(GraphCompileError):
        g.add_edge("a", "nope")
    with pytest.raises(GraphCompileError):
        StateGraph().compile()


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #

reg = ToolRegistry()


@reg.register("add", "add two ints", {"a": int, "b": int})
def _add(a: int, b: int) -> int:
    return a + b


@reg.register("boom", "always fails", {"x": int})
def _boom(x: int) -> int:
    raise RuntimeError("kaboom")


def test_tool_call_success_and_error():
    ok = reg.call("add", {"a": 2, "b": 3})
    assert ok.result == 5 and ok.error is None
    bad = reg.call("boom", {"x": 1})
    assert bad.result is None and "kaboom" in bad.error
    unknown = reg.call("nope", {})
    assert "unknown tool" in unknown.error
    missing = reg.call("add", {"a": 1})
    assert "missing args" in missing.error


def test_json_schema_shape():
    schema = reg.tools["add"].json_schema()
    assert schema["parameters"]["properties"]["a"]["type"] == "integer"
    assert schema["parameters"]["required"] == ["a", "b"]


def test_react_loop_executes_tools():
    replies = iter([
        "Thought: need sum\nAction: add\nAction Input: {\"a\": 2, \"b\": 40}",
        "Final Answer: 42",
    ])
    answer, calls = react_loop("what is 2+40?", lambda msgs: next(replies), reg)
    assert answer == "42"
    assert len(calls) == 1 and calls[0].result == 42


def test_react_loop_max_iterations():
    replies = iter(["Action: add\nAction Input: {\"a\": 1, \"b\": 1}"] * 20)
    answer, calls = react_loop("loop forever", lambda msgs: next(replies), reg,
                               max_iterations=3)
    assert answer == "max_iterations_reached" and len(calls) == 3
