# 路线图 · Chenxing (晨星)

> 作者 / Author: **晨星 (Chenxing)**

本栈当前是可教学、可审计、可复现的**算法内核**（纯 NumPy）。路线图按「零重写」原则，优先复用工业级实现而非自研。

---

## 已完成（v1.0.0）

- [x] 标准解码器拓扑（RMSNorm + RoPE + GQA + SwiGLU）
- [x] FlashAttention 分块在线 softmax（数值等价验证）
- [x] RoPE + YaRN 长上下文扩展
- [x] RadixPagedKVCache（RadixAttention ⊕ PagedAttention 融合）
- [x] 连续批处理调度器
- [x] 投机解码（贪心无损 + 随机分布等价，已验证）
- [x] INT4/INT8 absmax 量化 + LLM.int8() 离群分解
- [x] GGUF Q2_K / Q4_K / Q6_K / Q8_0 精确反量化
- [x] OpenAI 兼容 HTTP 服务（纯标准库）
- [x] ReAct 智能体（StateGraph + 工具注册表）
- [x] 84 个测试全绿 + `selfcheck` 端到端自检
- [x] ByteTokenizer 离线兜底，零下载跑通全链路

---

## 近期（v1.1）

- [ ] **真实权重加载**：从 HuggingFace `transformers` / 官方 GGUF 导出 Qwen2.5-0.5B，跑通真实文本生成（零下载小模型仅用于自检）。
- [ ] **树状投机解码**：从 SpecInfer 链验证升级为树验证（`verify_tree`），提升高熵场景接受率。
- [ ] **前缀缓存持久化**：radix 树跨请求共享 + LRU 容量策略（当前为进程内）。
- [ ] **CPU 多线程 matmul**：将 `QuantizedLinear` / attention 的密集 matmul 接 `numpy` 线程池或 `llama.cpp` 绑定。

## 中期（v1.x）

- [ ] **GPU 后端适配**：将 `Transformer.forward` 的算子替换为 CUDA / Metal 内核，或桥接 vLLM 的 `ModelRunner`。
- [ ] **Medusa / EAGLE 草稿头**：相比独立草稿模型，草稿头与主干同权，延迟更低（复用对应论文）。
- [ ] **张量并行**：多卡推理（复用 Megatron-LM / vLLM 的切分方案）。
- [ ] **流式输出**：SSE 流式 `/v1/chat/completions`。

## 长期

- [ ] **多模态**：视觉编码器桥接（SigLIP / CLIP）接入同一 KV 缓存。
- [ ] **训练 / 微调**：LoRA / QLoRA 微调路径（复用 `peft` / `bitsandbytes`）。
- [ ] **可观测性**：Prometheus 指标 + 追踪，对齐生产推理服务。

---

## 设计纪律（不变）

1. 任何新需求先检索世界最优解并评估，再复用 / 组合 / 超越。
2. 一旦有更强现成实现，立即替换自写代码。
3. 每个数值核保留「独立实现交叉验证」不变量与测试。
4. 默认零下载可运行；真实模型作为可选增强。
