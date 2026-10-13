# Chenxing · 晨星

**A world-class LLM inference & agent stack that reinvents nothing — it composes the strongest published designs and verifies every invariant.**

**作者 / Author: 晨星 (Chenxing) · GitHub: [CJX0712](https://github.com/CJX0712)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

---

## 为什么是 Chenxing

2026 年的开源推理栈已经收敛到几件公认最强的事:**PagedAttention**(vLLM)、**RadixAttention** 前缀复用(SGLang)、**投机解码**(Leviathan et al. 2023)、**FlashAttention** 在线 softmax、**GGUF k-quants**(llama.cpp)、**GQA/RoPE/SwiGLU/RMSNorm** 的标准 decoder 拓扑。

Chenxing 的原则是**零重写**:每一条公式、每一种格式、每一个调度语义,都站在命名的顶级成果之上(逐条出处见 [docs/PROVENANCE.md](docs/PROVENANCE.md)),而把工程创造力花在一件两个上游都没做干净的事上——

> **RadixPagedKVCache**:把 SGLang 的 RadixAttention(逻辑层:token 级精确前缀树)与 vLLM 的 PagedAttention(物理层:固定块池 + 块表)统一成一套缓存。前缀匹配 O(prefix)、物理碎片为零、拒绝回滚是 O(1) 元数据操作——投机解码的无损性正是靠这个回滚语义落地的。

## 核心特性

| 模块 | 世界顶级来源 | Chenxing 的贡献 |
|---|---|---|
| KV 缓存 | RadixAttention (SGLang) + PagedAttention (vLLM) | 二者的统一分层:radix 树管共享,块池管物理,整块对齐消除并发写冒险 |
| 投机解码 | Leviathan et al. ICML 2023; SpecInfer ASPLOS 2024 | 贪婪模式**逐 token 严格等价**目标模型(测试强制);拒绝回滚 = O(1) |
| 注意力 | FlashAttention (Dao 2022) 在线 softmax + GQA (Ainslie 2023) + sliding window | 分块在线 softmax 与朴素核 1e-6 内数值等价(测试强制) |
| 量化 | LLM.int8 (Dettmers 2022) outlier 分解; GGUF k-quants (llama.cpp) | Q2_K/Q4_K/Q6_K 反量化为 ggml-quants.c 的**精确移植**(字节级测试对齐) |
| 位置编码 | RoPE (Su 2021) + YaRN (Peng 2023) | 频率依赖插值 ramp,与 HF 参考实现同构 |
| 调度 | vLLM continuous batching (SOSP 2023) | 准入/抢占/LRU 驱逐,前缀在抢占后仍存活于 radix 树 |
| Agent | LangGraph 状态图 (2024) + ReAct (Yao 2023) | 200 行可审计内核:类型化状态、条件路由、工具 JSON-schema |
| 服务 | OpenAI Chat/Completions 协议 | 零依赖 stdlib 实现,可挂任意 WSGI/ASGI |

## 60 秒上手

```bash
# 安装(隔离 venv,仅依赖 numpy)
python -m venv .venv
.venv/Scripts/pip install -e .        # Windows
# .venv/bin/pip install -e .          # Linux/macOS

# 一键自检:确定性、KV 一致性、投机无损、radix 命中
.venv/Scripts/python -m chenxing.cli selfcheck

# 跑全测试套件(45+ 不变量)
.venv/Scripts/python -m pytest tests/ -v

# 基准
.venv/Scripts/python -m chenxing.cli bench --prompts 4 --new-tokens 32

# OpenAI 兼容服务
.venv/Scripts/python -m chenxing.cli serve --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

完整示例见 [`examples/`](examples/)。

## 一分钟读懂架构

```
                         ┌────────────────────────────────────┐
 OpenAI 兼容 HTTP  ←─────┤  serving/openai_server (stdlib)    │
                         ├────────────────────────────────────┤
 Agent 编排        ←─────┤  agent/graph + tools (ReAct)       │
                         ├────────────────────────────────────┤
 调度              ←─────┤  engine/scheduler (continuous batch)│
                         ├────────────────────────────────────┤
 无损加速          ←─────┤  engine/speculative (Leviathan'23) │
                         ├────────────────────────────────────┤
 生成循环          ←─────┤  engine/engine (prefill→decode)    │
                         ├────────────────────────────────────┤
 ★ 缓存核心       ←─────┤  engine/radix_kv (Radix×Paged)     │
                         ├────────────────────────────────────┤
 模型前向          ←─────┤  model/transformer (GQA+RoPE+SwiGLU)│
                         ├────────────────────────────────────┤
 内核              ←─────┤  core/{attention, quant, tensor}   │
                         └────────────────────────────────────┘
```

## 可验证性(本项目的一等公民)

每个顶级成果都配一条**可独立复现的不变量测试**:

- `test_attention.py` — 分块在线 softmax ≡ 朴素 softmax(1e-6),因果/滑窗/GQA 全覆盖
- `test_radix_kv.py` — 前缀零拷贝共享、LRU 驱逐、refcount 下溢即崩、块池无泄漏
- `test_speculative.py` — **γ∈{1,2,4,8} 输出逐 token 等价于纯目标贪婪**;温度采样分布等价
- `test_quantization.py` — INT4/INT8 误差在理论界内;Q2_K/Q4_K/Q6_K 对手排字节块精确一致
- `test_transformer.py` — prefill ≡ 增量解码(logits 1e-5);KV 回滚幂等
- `test_gguf.py` — 容器往返;`test_scheduler.py` — 准入/抢占不变量;`test_agent.py`、`test_openai_api.py`

## 文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 模块职责与数据流
- [docs/PROVENANCE.md](docs/PROVENANCE.md) — 每个组件的论文/工程出处(零重写清单)
- [docs/API.md](docs/API.md) — 公共 API 参考
- [docs/ROADMAP.md](docs/ROADMAP.md) — 限制与路线
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — 已知限制(诚实清单)

## 许可

MIT © 2026 晨星 (Chenxing)
