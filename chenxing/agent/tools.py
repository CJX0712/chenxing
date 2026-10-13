"""Tool registry + ReAct loop.

Provenance:

- ReAct (Reason+Act interleaving) : Yao et al., ICLR 2023 (arXiv:2210.03629).
- Tool/function-calling protocol  : OpenAI function-calling schema and its
  open adoption; tools declare name/description/JSON-schema params
  (json-schema.org draft 2020-12 subset).

The ReAct runner is engine-agnostic: it works with any `chat_fn(messages,
system) -> str`, so tests drive it with a scripted model instead of a
language model.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["Tool", "ToolRegistry", "ToolCall", "react_loop"]

_MISSING = object()


@dataclass
class Tool:
    name: str
    description: str
    params_schema: dict[str, type]           # param name -> python type
    fn: Callable[..., Any]

    def json_schema(self) -> dict[str, Any]:
        tmap = {str: "string", int: "integer", float: "number", bool: "boolean",
                list: "array", dict: "object"}
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": tmap.get(v, "string")}
                               for k, v in self.params_schema.items()},
                "required": list(self.params_schema),
            },
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, name: str, description: str, params_schema: dict[str, type]):
        def deco(fn: Callable[..., Any]) -> Tool:
            self.tools[name] = Tool(name, description, params_schema, fn)
            return fn
        return deco

    def call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        tool = self.tools.get(name)
        if tool is None:
            return ToolCall(name, arguments, error=f"unknown tool {name!r}")
        missing = [k for k in tool.params_schema if k not in arguments]
        if missing:
            return ToolCall(name, arguments, error=f"missing args {missing}")
        try:
            return ToolCall(name, arguments, result=tool.fn(**arguments))
        except Exception as exc:                       # tool boundary
            return ToolCall(name, arguments, error=f"{type(exc).__name__}: {exc}")

    def catalog(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())


# --------------------------------------------------------------------------- #
# ReAct loop
# --------------------------------------------------------------------------- #

_ACTION_RE = re.compile(r"Action:\s*(\w+)\s*Action\s*Input:\s*(\{.*?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL)


def react_loop(query: str, chat_fn: Callable[[list[dict[str, str]]], str],
               registry: ToolRegistry, *, max_iterations: int = 6) -> tuple[str, list[ToolCall]]:
    """Interleave reasoning and acting until `Final Answer:` or iteration cap.

    `chat_fn(messages) -> str` is any chat backend. The prompt protocol is the
    classic ReAct scratchpad (Thought / Action / Action Input / Observation).
    """
    messages: list[dict[str, str]] = [
        {"role": "system",
         "content": "You are a tool-using agent. Use exactly one tool per turn.\n"
                    f"Tools:\n{registry.catalog()}\n"
                    "Reply with either:\n"
                    "Thought: ...\nAction: <tool>\nAction Input: {\"arg\": ...}\n"
                    "or\nFinal Answer: <text>"},
        {"role": "user", "content": query},
    ]
    calls: list[ToolCall] = []
    for _ in range(max_iterations):
        reply = chat_fn(messages)
        final = _FINAL_RE.search(reply)
        if final:
            return final.group(1).strip(), calls
        action = _ACTION_RE.search(reply)
        if action is None:
            return reply.strip(), calls               # no protocol → treat as final
        name, raw_args = action.group(1), action.group(2)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
        call = registry.call(name, args)
        calls.append(call)
        observation = call.error if call.error else json.dumps(call.result, ensure_ascii=False)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    return "max_iterations_reached", calls
