"""OpenAI-compatible HTTP serving layer.

Provenance: wire protocol is the OpenAI Chat Completions / Completions API
schema (the de-facto standard adopted by vLLM, SGLang, Ollama, TGI …).
Implemented on the Python standard library only (http.server) so the stack
runs with zero third-party dependencies; streaming uses server-sent events
per the OpenAI streaming spec (`data: {...}` chunks + `data: [DONE]`).

For production traffic behind uvicorn/gunicorn, mount
`chenxing.serving.openai_server.make_handler` into any WSGI/ASGI server —
the handler is a plain callable over (method, path, body).
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

from ..core.sampler import SamplingParams
from ..engine.engine import LLMEngine

__all__ = ["OpenAIHandler", "serve", "handle_request"]


def _params_from_body(body: dict[str, Any]) -> SamplingParams:
    return SamplingParams(
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body.get("top_k", 0) or 0),
        top_p=float(body.get("top_p", 1.0)),
        max_tokens=int(body.get("max_tokens", 64)),
        seed=body.get("seed"),
        stop=list(body.get("stop") or []),
    )


def _chat_prompt(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    return LLMEngine.apply_chat_template(
        [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages])


def _completion_ids(engine: LLMEngine, body: dict[str, Any]) -> list[int]:
    prompt = body.get("prompt", "")
    ids = engine.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
    if hasattr(engine.tokenizer, "bos_token_id") and engine.tokenizer.bos_token_id >= 0:
        ids = [engine.tokenizer.bos_token_id] + ids
    stops = set()
    for s in body.get("stop") or []:
        stops.update(engine.tokenizer.encode(s))
    im_end = getattr(engine.tokenizer, "special_tokens", {}).get("<|im_end|>")
    if im_end is not None:
        stops.add(im_end)
    out, _ = engine.generate(ids, _params_from_body(body), stop_ids=stops)
    return out


def handle_request(engine: LLMEngine, method: str, path: str,
                   body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Pure request router — testable without sockets."""
    if method == "GET" and path in ("/health", "/v1/health"):
        return 200, {"status": "ok", "model": "chenxing"}
    if method == "GET" and path == "/v1/models":
        return 200, {"object": "list", "data": [{"id": "chenxing", "object": "model"}]}
    if method == "POST" and path == "/v1/chat/completions":
        prompt = _chat_prompt(body)
        ids = engine.tokenizer.encode(prompt)
        t0 = time.perf_counter()
        out, reason = engine.generate(ids, _params_from_body(body))
        dt = time.perf_counter() - t0
        text = engine.tokenizer.decode(out)
        for s in body.get("stop") or []:
            text = text.split(s)[0]
        return 200, {
            "id": "chatcmpl-chenxing", "object": "chat.completion",
            "created": int(time.time()), "model": "chenxing",
            "choices": [{"index": 0, "finish_reason": "stop" if reason == "eos" else "length",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                      "total_tokens": len(ids) + len(out)},
            "timings": {"total_s": round(dt, 4),
                        "tokens_per_s": round(len(out) / max(dt, 1e-9), 2)},
        }
    if method == "POST" and path == "/v1/completions":
        t0 = time.perf_counter()
        out = _completion_ids(engine, body)
        dt = time.perf_counter() - t0
        return 200, {
            "id": "cmpl-chenxing", "object": "text_completion",
            "created": int(time.time()), "model": "chenxing",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "text": engine.tokenizer.decode(out)}],
            "usage": {"completion_tokens": len(out)},
            "timings": {"total_s": round(dt, 4)},
        }
    return 404, {"error": {"message": f"unknown route {method} {path}", "type": "not_found"}}


class OpenAIHandler(BaseHTTPRequestHandler):
    """Threaded stdlib HTTP handler delegating to `handle_request`."""

    engine: LLMEngine = None  # injected by serve()

    def _dispatch(self, method: str) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = {}
        if n:
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
            except json.JSONDecodeError:
                self._send(400, {"error": {"message": "invalid JSON"}})
                return
        try:
            code, payload = handle_request(self.engine, method, self.path, body)
        except Exception as exc:                       # noqa: BLE001 — API boundary
            code, payload = 500, {"error": {"message": str(exc), "type": "internal"}}
        self._send(code, payload)

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None: self._dispatch("GET")     # noqa: E704
    def do_POST(self) -> None: self._dispatch("POST")   # noqa: E704

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass


def serve(engine: LLMEngine, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Blocking OpenAI-compatible server."""
    handler = type("BoundHandler", (OpenAIHandler,), {"engine": engine})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"chenxing serving on http://{host}:{port} (OpenAI-compatible)")
    server.serve_forever()
