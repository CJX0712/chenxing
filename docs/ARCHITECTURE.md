# 架构总览 · Chenxing (晨星) 推理栈

> 作者 / Author: **晨星 (Chenxing)**
> 设计准则：**零重写**——每个组件都先检索并复用世界已有的最优解，再在其上组合 / 超越，绝不从零造轮子。

本文件描述 Chenxing 的整体架构、模块职责与数据流。所有模块的公式、算法、工程实现均来自公开论文或开源参考实现，已在 [PROVENANCE.md](PROVENANCE.md) 中逐项标注出处。

---

## 1. 设计目标

| 目标 | 本栈的做法 |
|------|-----------|
| 最优架构 | 直接采用 Llama-2/3 · Qwen2 家族解码器拓扑（RMSNorm + RoPE + GQA + SwiGLU） |
| 最优推理性能 | RadixAttention（前缀树）⊕ PagedAttention（物理分页）融合的 KV 缓存；投机解码；连续批处理调度 |
| 数值保真 | FlashAttention 分块在线 softmax；GGUF k-quant 精确反量化；LLM.int8() 离群分解 |
| 可复现验证 | 84 个测试全绿，每个数值核都有「独立实现交叉验证」不变量；CLI `selfcheck` 端到端自检 |
| 零依赖可运行 | 纯 NumPy 实现，附带 ByteTokenizer 离线兜底，无需下载权重即可跑通全链路 |

---

## 2. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI  (chenxing.cli)                      │
│   selfcheck │ serve │ bench    —— 一键自检 / OpenAI 服务 / 压测  │
└───────────────┬───────────────────────────┬──────────────────┘
                │                            │
      ┌─────────▼──────────┐       ┌─────────▼──────────┐
      │  serving (OpenAI)  │       │   agent (ReAct)    │
      │  /v1/chat/completions│     │  StateGraph + Tools│
      └─────────┬──────────┘       └─────────┬──────────┘
                │                            │
      ┌─────────▼────────────────────────────▼──────────┐
      │            engine.LLMEngine  (生成主循环)          │
      │   prefill / decode_step / generate / chat         │
      └─────────┬────────────────────────────┬──────────┘
                │                            │
   ┌────────────▼───────────┐    ┌───────────▼───────────┐
   │  RadixPagedKVCache       │    │  SpeculativeDecoder    │
   │  (Radix ⊕ Paged 融合)    │    │  (Leviathan 2023)      │
   └────────────┬───────────┘    └───────────┬───────────┘
                │                            │
      ┌─────────▼────────────────────────────▼──────────┐
      │            model.Transformer  (解码器前向)         │
      │   RMSNorm → RoPE → GQA → SwiGLU → 输出投影         │
      └─────────┬────────────────────────────┬──────────┘
                │                            │
   ┌────────────▼───────────┐    ┌───────────▼───────────┐
   │  core (数值原语)         │    │  model (权重/Tokenizer)│
   │  tensor / attention /    │    │  config / gguf /       │
   │  quantization / sampler   │    │  tokenizer            │
   └─────────────────────────┘    └───────────────────────┘
```

---

## 3. 模块职责

### 3.1 `core/` —— 数值原语（与具体模型解耦）

| 模块 | 职责 | 关键不变量（测试交叉验证）|
|------|------|--------------------------|
| `tensor.py` | RMSNorm / SiLU / SwiGLU / 稳定 softmax / RoPE 缓存与施加 | RoPE 是保正交旋转：`‖x‖` 逐位不变；相对位置性质 `dot(q_i,k_j)` 仅依赖 `i−j` |
| `attention.py` | 分块在线 softmax 的 FlashAttention；GQA 扩展；因果 / 滑窗掩码 | `flash_attention ≈ naive_attention`（误差 < 5e-8） |
| `quantization.py` | absmax INT4/INT8；LLM.int8() 离群分解；GGUF Q2_K/Q4_K/Q6_K/Q8_0 反量化 | INT4 重建误差 ≤ `amax/(2·qmax)`；k-quant 按 C 字节布局逐位反量化 |
| `sampler.py` | 温度 / top-k / top-p / min-p / typical-p / 重复惩罚；贪心 | 贪心 = argmax；随机采样分布 = 截断后归一化分布 |

### 3.2 `model/` —— 模型与权重

| 模块 | 职责 |
|------|------|
| `config.py` | `ModelConfig`：Qwen2.5-0.5B 默认超参；`scaled()` 生成草稿小模型 |
| `transformer.py` | 标准解码器前向；`build_tiny_model()` 确定性随机初始化用于零下载自检 |
| `gguf.py` | GGUF 容器读写（元数据类型系统 + 对齐数据段 + F32/F16/Q8_0 张量） |
| `tokenizer.py` | 字节级 BPE（GGUF/tokenizer.json）；`ByteTokenizer` 离线兜底 |

### 3.3 `engine/` —— 推理引擎

| 模块 | 职责 | 关键不变量 |
|------|------|-----------|
| `radix_kv.py` | **RadixPagedKVCache**：逻辑前缀树（RadixAttention）+ 物理分页块池（PagedAttention）融合 | 整块共享消除并发写竞争；O(1) 元数据回滚使投机解码无损 |
| `sequence.py` | 序列状态（token 列表 / 块表 / n_tokens 单一事实源） | `seq.n_tokens` = 已提交 KV 的 token 数 |
| `engine.py` | 预填充 / 解码 / 生成主循环；前缀复用；抢占式驱逐 | 增量解码 ≡ 全量预填充（误差 < 1e-5） |
| `speculative.py` | 投机解码：草稿 γ 个 token → 目标单次前向验证 | 贪心模式下 token 与目标贪心**逐位相同**；随机模式下输出分布**严格等于**目标分布（Leviathan Thm 3.5） |
| `scheduler.py` | 连续批处理调度器（vLLM 模型） | `submit / step / run_until_drained` |

### 3.4 `serving/` —— OpenAI 兼容服务

`openai_server.py`：纯标准库实现的 HTTP 路由，覆盖 `/health`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`，请求 / 响应体与 OpenAI SDK 对齐。

### 3.5 `agent/` —— 工具调用智能体

`graph.py`：受 LangGraph 启发的 `StateGraph`（节点 / 边 / 条件边 / reducer / `compile`）；`tools.py`：工具注册表 + ReAct 循环。

### 3.6 `eval/` —— 评估

`metrics.py`：困惑度（perplexity）、生成吞吐基准。

---

## 4. 数据流：一次投机解码请求

```
用户 prompt
   │
   ▼
engine.prefill(seq)            # 复用 radix 树中已有前缀 KV，仅前向尾部
   │
   ▼
speculative.generate:
   ├─ 草稿引擎：自回归提出 γ 个 token（每次前向消费 KV）
   ├─ 目标引擎：对 [种子, 草案[:-1]] 单次前向，得 γ 行 next-token 分布
   ├─ 验证（Leviathan Alg.1）：
   │     贪心：草案 token == 目标 argmax 则接受，否则以目标 argmax 替换
   │     随机：以 min(1, p/q) 接受；否则从残差 norm(max(0,p−q)) 抽取
   └─ 全接受时，额外一行分布给出「bonus token」
   │
   ▼
tokenizer.decode → 文本
```

---

## 5. 性能与验证状态

| 项 | 状态 |
|----|------|
| 单元测试 | **84 passed**（tensor / attention / quantization / radix / transformer / speculative / gguf / sampler / scheduler / agent / openai / tokenizer / eval） |
| 端到端自检 | `chenxing selfcheck` 秒级通过（确定性贪心 + radix KV + 引擎循环） |
| 无损性定理 | 投机解码贪心逐位一致（γ∈{1,2,4,8} 全验证）；随机分布等价（独立 RNG 流验证） |
| 吞吐基准 | `chenxing bench` 输出 prefill / decode tokens-per-second |

下一步路线见 [ROADMAP.md](ROADMAP.md)，已知边界见 [LIMITATIONS.md](LIMITATIONS.md)。
