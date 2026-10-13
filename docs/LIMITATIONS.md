# 已知限制 · Limitations

**作者 / Author: 晨星 (Chenxing)**

> 诚实清单。本项目是**零重写**的参考实现：每一个算法都站在命名的顶级成果之上，并把工程创造力集中在统一 RadixPaged KV 缓存这一件事上。下面列出的不是缺陷，而是这一设计取舍带来的、**已被测试钉死的边界**。生产部署请见 [docs/ROADMAP.md](ROADMAP.md)。

---

## 1. 内置 demo 模型未训练（最关键的一条）

`chenxing.cli selfcheck / serve / bench` 与 `examples/` 里的端到端脚本，默认用 `model/transformer.build_tiny_model()` 生成**确定性随机初始化**的权重（种子 42）。

- 它的**结构**与 Qwen2.5-0.5B 完全一致（`ModelConfig` 默认值即 Qwen2.5-0.5B 的超参），所以真实 GGUF 权重走的是同一套代码路径。
- 它的**权重**是 `N(0, 1/√hidden)` 的高斯噪声，**不是语言模型**。
- 因此：demo 能证明 *架构正确、不变量成立、调度/缓存/量化语义无损*，但**生成的是无意义的 token 串，不是流畅文本**。这是预期行为，不是 bug。
- 要跑真实文本，请加载外部 GGUF（见 `examples/load_real_gguf.py` 的接线示例）与对应 tokenizer。

## 2. 离线 ByteTokenizer 回退

没有 GGUF tokenizer 时，引擎回退到 `model/tokenizer.ByteTokenizer`：

- 1 字节 = 1 token（`vocab_size=512`，`eos=256`，`bos=257`）。
- 保证「训练 tiny 模型 → serve → agent」全链路**零下载**可运行。
- 代价：字符级字节编码，序列极长、困惑度极高，**不代表真实分词质量**。真实分词请用 GGUF 的 `tokenizer.ggml.tokens/merges` 或 HF `tokenizer.json`（已支持，见 `BPETokenizer`）。

## 3. 纯 NumPy 参考性能 ≠ 生产吞吐

- 全部前向用 NumPy 2.x（CPU）。这是**算法正确性与数值等价性**的参考实现，不是性能后端。
- 基准输出（`cli bench` / `benchmarks/`）的 prefill/decode tok/s 反映的是**本机 NumPy 的 CPU 上限**，不能与 vLLM/llama.cpp 的 CUDA 内核比较。
- 批量维度当前是 **sequence-by-sequence**：continuous-batching 的*调度结构*（准入/抢占/LRU 驱逐/前缀存活）是生产形态，但单次 step 的矩阵乘 是 batch=1 内核。真正的并行 batched matmul 是 `benchmarks/` 之外未来的 JIT/CUDA 路径（见 ROADMAP）。

## 4. 前缀缓存是进程内、单引擎的

- `engine/radix_kv.RadixPagedKVCache` 活在**引擎进程内存**里。
- 不支持跨进程、跨节点、跨重启的共享；进程退出即清空。
- 不提供分布式 KV（如 vLLM 的分布式 KV 迁移 / SGLang 的多副本前缀共享）。多副本部署需在上层自行加一层外部缓存或重新 prefill。

## 5. GGUF 格式支持范围

| 能力 | 支持 | 说明 |
|---|---|---|
| **读** F32 / F16 | ✅ | 直通反量化 |
| **读** Q8_0 | ✅ | 精确反量化 |
| **读** Q2_K / Q4_K / Q6_K | ✅ | 为 `ggml-quants.c` 的**精确移植**，有字节级对齐测试 |
| **读** 其他 K-quant（Q3_K/Q5_K/Q4_0/Q8_K/IQn…） | ⚠️ | `BLOCK_SIZES` 已登记块参数，但反量化器尚未实现（dequantizer 表仅含 Q2_K/Q4_K/Q6_K/Q8_0 + F32/F16）。读这些格式会抛 `KeyError`。 |
| **写** F32 | ✅ | 默认路径 |
| **写** Q8_0 | ✅ | `write_gguf(..., quantize_q8_0=True)` |
| **写** K-quant | ❌ | 写入端只产出 F32/Q8_0（round-trip 测试足够）；K-quant 写入留作扩展 |

> 覆盖范围已覆盖主流 `Q4_K_M / Q2_K / Q6_K` 模型家族；Q3_K/Q5_K 的反量化是近-term 待办（ROADMAP）。

## 6. 投机解码：已实现且被测试证明无损，但 demo 默认不启用

- `engine/speculative.SpeculativeDecoder` 完整实现 Leviathan 2023 + SpecInfer 2024 的链式验证：
  - **贪婪模式**：逐 token 与目标模型 `argmax` 严格等价（`test_speculative.py::test_greedy_losslessness` 强制）。
  - **温度模式**：输出分布 ≡ 目标分布（Leviathan Thm 3.5，`test_stochastic_distribution_equivalence` 实证）。
  - 回滚是 **O(1)** 元数据操作（PagedAttention 语义）。
- 但 `cli` 的 `selfcheck/serve/bench` 走**单引擎（target-only）** 路径以保证确定性；要看到加速比，需自行构造一个更小的 draft 引擎（见 `examples/run_speculative.py`）。当 draft == target 时，加速比退化为 1（合法但无意义）。

## 7. 调度：batch=1 + 抢占语义

- 抢占（KV 耗尽时驱逐 LRU 叶子 → 抢占最老运行序列，释放其私有块并重排）已实现。
- 被抢占序列的**共享前缀**仍存活于 radix 树，重排后无需重算前缀（不变量测试覆盖）。
- 当前 batch 上限受单引擎 step 限制：每个 step 顺序处理各序列。真正的 max-batch token 级调度是未来工作。

## 8. 量化：推理端反量化，未做训练端量化内核

- 本栈是**推理栈**：做反量化（dequant）与 matmul，不做训练端量化感知训练（QAT）。
- INT8/INT4 路径提供 LLM.int8 的 **outlier 分解** matmul（`llm_int8_matmul`），但默认前向走 fp32；显式量化 matmul 需调用 `QuantizedLinear` / `llm_int8_matmul`（示例见 `examples/quantized_matmul.py`）。
- 量化误差始终落在理论界内（`test_quantization.py` 校验），但**未**做 CUDA 级融合内核。

## 9. 服务层：stdlib，无原生流式/SSE 聚合

- `serving/openai_server` 用纯标准库 `http.server`，零第三方依赖。
- 实现 `/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`（非流式）。
- **暂未实现** SSE 流式 `stream:true`、并发限流、鉴权、批处理端点；生产请挂 uvicorn/gunicorn 或前置网关（handler 是纯可调用对象，可挂载到任意 ASGI/WSGI）。

## 10. 确定性边界

- 贪婪（`temperature=0`）与带 `seed` 的采样完全确定性（已测：`selfcheck` 两次生成一致）。
- 不传 `seed` 的温度采样使用进程内全局 RNG，**跨进程不保证可复现**（符合预期）。
- NumPy 2.x 的 `float32` 求和顺序在不同 BLAS 后端下可能有 ~1e-6 级差异；这与朴素核的数值等价测试容差一致。

---

### 一句话总结

Chenxing 保证的是 **「世界顶级设计的精确组合 + 每一条可被独立验证的不变量」**；它在 NumPy 上把 RadixPaged 缓存、无损投机解码、GGUF k-quant 反量化、continuous batching 跑通并测绿，但**吞吐、分布式、真实语言质量**需要接入生产后端（CUDA/GGUF 权重/HF tokenizer）——这些正是 [docs/ROADMAP.md](ROADMAP.md) 的下一步。
