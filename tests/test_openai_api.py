"""OpenAI-compatible API: pure router tests (no sockets)."""
from chenxing.core.sampler import SamplingParams
from chenxing.serving.openai_server import handle_request
from chenxing.engine.engine import LLMEngine
from chenxing.model.config import ModelConfig
from chenxing.model.transformer import build_tiny_model


def make_engine():
    cfg = ModelConfig(vocab_size=264, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model, _ = build_tiny_model(cfg)
    engine = LLMEngine(model, None, block_size=8, n_blocks=64)
    # ByteTokenizer-like stub via engine.tokenizer is None → use built-in fallback
    from chenxing.model.tokenizer import ByteTokenizer
    engine.tokenizer = ByteTokenizer()
    return engine


def test_health_and_models():
    e = make_engine()
    code, body = handle_request(e, "GET", "/health", {})
    assert code == 200 and body["status"] == "ok"
    code, body = handle_request(e, "GET", "/v1/models", {})
    assert code == 200 and body["data"][0]["id"] == "chenxing"


def test_chat_completion_shape():
    e = make_engine()
    code, body = handle_request(e, "POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "temperature": 0.0})
    assert code == 200
    ch = body["choices"][0]
    assert ch["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] <= 8
    assert "tokens_per_s" in body["timings"]


def test_completion_route():
    e = make_engine()
    code, body = handle_request(e, "POST", "/v1/completions",
                                {"prompt": "hello", "max_tokens": 5})
    assert code == 200 and len(body["choices"][0]["text"]) > 0


def test_unknown_route_404():
    e = make_engine()
    code, body = handle_request(e, "POST", "/v1/nope", {})
    assert code == 404 and body["error"]["type"] == "not_found"


def test_chat_template_format():
    out = LLMEngine.apply_chat_template([{"role": "user", "content": "hi"}])
    assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
