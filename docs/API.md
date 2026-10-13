# API 参考 · Chenxing (晨星)

> 作者 / Author: **晨星 (Chenxing)**

所有公开符号均通过 `chenxing` 顶层包导出。示例假设已安装并在 Python 中 `import chenxing`。

---

## 1. 顶层包 `chenxing`

```python
import chenxing
chenxing.__author__   # '晨星 (Chenxing)'
chenxing.__version__  # '1.0.0'
```

---

## 2. 模型配置 `chenxing.model.config`

### `ModelConfig`
解码器超参（默认等价于 Qwen2.5-0.5B，但可任意缩放）。

| 字段 | 默认 | 说明 |
|------|------|------|
| `vocab_size` | 151936 | 词表大小 |
| `hidden_size` | 896 | 隐藏维 |
| `num_hidden_layers` | 24 | 层数 |
| `num_attention_heads` | 14 | 查询头数 |
| `num_key_value_heads` | 2 | KV 头数（GQA） |
| `head_dim` | 64 | 每头维 |
| `intermediate_size` | 4864 | FFN 中间维 |
| `rms_norm_eps` | 1e-6 | RMSNorm ε |
| `rope_theta` | 1e6 | RoPE 基频 |
| `rope_scaling_factor` | 1.0 | YaRN 缩放（>1 启用） |
| `max_position_embeddings` | 4096 | 位置缓存长度 |
| `eos_token_id` | 151645 | 结束符 |

方法：
- `scaled(scale: float) -> ModelConfig`：等比缩放隐藏维 / 中间维 / 层数，用于生成草稿小模型。

---

## 3. 解码器 `chenxing.model.transformer`

### `Transformer(model_cfg, weights)`
从权重字典构建解码器。`weights` 使用 GGUF 布局（`(out, in)`），键如 `token_embd.weight`、`blk.{i}.attn_q.weight` 等。

### `build_tiny_model(cfg, seed=42) -> (Transformer, weights)`
**确定性**随机初始化一个标准拓扑小模型，用于零下载自检 / 演示。注意：它不是训练好的语言模型（见 LIMITATIONS）。

### `Transformer.forward(input_ids, positions, kv_cache, block_table, n_past) -> np.ndarray`
单步前向，返回 `(T, vocab)` 的 logits。所有状态存于调用方管理的分页 KV 缓存。

---

## 4. 推理引擎 `chenxing.engine.engine`

### `LLMEngine(model, tokenizer, *, block_size=16, n_blocks=512)`
拥有模型、分页 KV 缓存与生成循环。

| 方法 | 说明 |
|------|------|
| `create_sequence(prompt_tokens, params)` | 新建序列 |
| `prefill(seq)` | 处理 prompt，复用 radix 前缀；返回 next-token logits |
| `decode_step(seq) -> int` | 前进一步并采样一个 token |
| `generate(prompt_tokens, params) -> (list[int], str)` | 完整生成，返回 token 列表与停止原因 |
| `chat(messages, params) -> str` | ChatML 多轮对话 |
| `kv.stats()` | 返回 KV 缓存统计（含前缀命中率） |

---

## 5. 缓存 `chenxing.engine.radix_kv`

### `RadixPagedKVCache(n_blocks, block_size, n_layers, n_kv_heads, head_dim)`
RadixAttention（前缀树）+ PagedAttention（物理分页）融合缓存。

- `match_prefix(tokens) -> PrefixHit`：逻辑前缀精确命中
- `insert(tokens, blocks, parent)`：发布前缀到 radix 树
- `write_layer(block_table, n_past, layer, k, v)` / `read_layer(block_table, n_tokens, layer)`：按层读写 KV
- `ensure_free(n)` / `evict()`：容量管理
- `acquire(node)` / `release(node)`：引用计数

---

## 6. 投机解码 `chenxing.engine.speculative`

### `SpeculativeDecoder(engine, draft_engine, *, num_speculative_tokens=4)`
用草稿引擎提出 γ 个 token，目标引擎单次前向验证。

```python
sd = SpeculativeDecoder(engine, draft_engine, num_speculative_tokens=4)
out, reason = sd.generate(prompt_tokens, SamplingParams(max_tokens=24, temperature=0.0, seed=7))
```

不变量：
- 贪心（`temperature=0`）下 `out == engine.generate(...)`（逐位一致）。
- 随机下输出分布严格等于目标分布（Leviathan Thm 3.5）。

属性：`sd.stats`（`SpeculativeStats`：acceptance_rate / mean_accepted_per_step）。

---

## 7. 采样 `chenxing.core.sampler`

### `SamplingParams`
| 字段 | 默认 | 说明 |
|------|------|------|
| `temperature` | 1.0 | 0 = 贪心 |
| `top_k` | 0 | 0 = 关闭 |
| `top_p` | 1.0 | 1.0 = 关闭 |
| `min_p` | 0.0 | |
| `typical_p` | 1.0 | |
| `repetition_penalty` | 1.0 | |
| `max_tokens` | 64 | |
| `seed` | None | 可复现采样 |
| `stop` | `[]` | 停止串 |
| `n_keep` | 0 | 重复惩罚作用范围 |

### `Sampler(vocab_size)`
- `apply_penalties(logits, params, generated_ids)`
- `sample(logits, params, generated_ids) -> int`

---

## 8. 量化 `chenxing.core.quantization`

| 符号 | 说明 |
|------|------|
| `quantize_absmax(x, bits, group_size)` / `dequantize_absmax(q, s, group_size)` | 每组 absmax 整型量化 |
| `llm_int8_matmul(x, w, outlier_threshold=6.0)` | LLM.int8() 混合精度 matmul |
| `QuantizedLinear(weight, bits=4, group_size=64)` | 4/8 位线性层（int4 nibble 打包） |
| `dequantize_gguf_row(raw, ggml_type, n_elements)` | GGUF 行反量化（F32/F16/Q8_0/Q2_K/Q4_K/Q6_K） |
| `GGMLType` | GGUF 类型枚举与 `BLOCK_SIZES` |

---

## 9. 服务 `chenxing.serving.openai_server`

### `serve(engine, host='127.0.0.1', port=8000)`
启动 OpenAI 兼容 HTTP 服务（纯标准库）。

### `handle_request(engine, method, path, body) -> (status, dict)`
无框架路由入口，便于测试与嵌入。

路由：
- `GET /health` → `{"status":"ok"}`
- `GET /v1/models` → 模型列表
- `POST /v1/chat/completions` → ChatCompletion
- `POST /v1/completions` → Completion

---

## 10. 智能体 `chenxing.agent`

### `StateGraph`
- `add_node(name, fn)` / `add_edge(u, v)` / `add_conditional_edges(u, router, {label: node})`
- `set_reducer(state_key, fn)` / `compile() -> CompiledGraph`
- `CompiledGraph.invoke(inputs) -> outputs`

### `Tool` / `ToolRegistry`
- `Tool(name, description, fn)` / `registry.register(tool)`
- `registry.call(name, args)` / `registry.catalog()` / `registry.json_schema()`

### `react_loop(query, chat_fn, registry, max_iterations=5) -> str`
ReAct 推理-行动循环。

---

## 11. 评估 `chenxing.eval.metrics`

- `perplexity(logits_seq, targets) -> float`
- `generation_benchmark(engine, prompts, max_new_tokens) -> dict`（含 prefill/decode tokens-per-second）
