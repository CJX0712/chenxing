# 溯源清单 · 零重写原则（Provenance / Zero-Reinvention Manifest）

> 作者 / Author: **晨星 (Chenxing)**

**核心原则：本项目禁止从零造轮子。** 每一项数学、算法、工程实现都来自世界已有最优解，并标注确切出处。下表是「组件 → 原始论文 / 开源参考」的逐项映射，可作为第三方审计依据。

> 说明：本栈纯 NumPy 实现，目的是**可教学、可审计、可复现**地还原工业级推理引擎的算法内核；部署级性能应替换为 vLLM / llama.cpp / TensorRT-LLM 等（见 ROADMAP）。

---

## 1. 架构与拓扑

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| 解码器主拓扑（pre-norm RMSNorm + RoPE + GQA + SwiGLU） | Llama-2 / Llama-3 | Touvron et al., "Llama 2", 2023 (arXiv:2307.09288) |
| Qwen2 权重命名约定（GGUF `blk.{i}.*`） | Qwen2 | Bai et al., "Qwen2", 2024 (arXiv:2407.10671) |
| 连续批处理调度 | vLLM | Kwon et al., "vLLM", SOSP 2023 (arXiv:2309.06180) |

## 2. 注意力与位置编码

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| RMSNorm | Zhang & Sennrich, 2019 (arXiv:1907.02039) |
| RoPE 旋转位置编码 | Su et al., 2021 (arXiv:2104.09864) |
| NTK-aware / YaRN 长上下文扩展 | Peng et al., 2023 (arXiv:2309.00071)，实现对齐 HuggingFace `transformers` `_compute_yarn_parameters` |
| GQA 分组查询注意力 | Ainslie et al., 2023 (arXiv:2305.13245) |
| FlashAttention 分块在线 softmax | Dao et al., 2022 (arXiv:2205.14135)；在线归一化源自 Milakov & Gimelshein, 2018 (arXiv:1805.02867) |
| 滑窗注意力 | Beltagy et al., 2020 (arXiv:2004.05150) |

## 3. 量化

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| 每组 absmax 整型量化 | Jacob et al., "LLM.int8()", NeurIPS 2022 (arXiv:2208.07339) |
| 离群列 fp16 分解（混合精度 matmul） | 同上，Alg. 1–2 |
| GGUF k-quant 反量化（Q2_K / Q4_K / Q6_K / Q8_0） | **ggml-org/llama.cpp `ggml-quants.c`（MIT）** | 逐字节布局与 `get_scale_min_k4` 位运算为精确移植 |
| INT4 nibble 打包存储 | llama.cpp `ggml-quants.c`（`dequantize_row_q4_K` 风格） |

## 4. 投机解码

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| 无损性定理（Thm 3.5）与验证算法（Alg. 1） | Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023 (arXiv:2211.17192) |
| 残差分布 norm(max(0, p−q)) 采样 | Chen et al., "Accelerating LLM Decoding with Speculative Sampling", 2023 (arXiv:2302.01318) |
| 树验证的单次前向特例 | Miao et al., "SpecInfer", ASPLOS 2024 (arXiv:2305.09781) |

## 5. 采样

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| top-k / top-p (nucleus) | Holtzman et al., "The Curious Case of Neural Text Degeneration", ICLR 2020 (arXiv:1904.09751) |
| min-p 采样 | Nguyen, 2024 (arXiv:2407.01082) |
| typical (locally typical) 采样 | Meister & Cotterell, 2023 (arXiv:2202.00666) |
| 重复惩罚 | Keskar et al., 2019（CTRL，HF transformers 沿用） |

## 6. KV 缓存

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| RadixAttention（逻辑前缀树 + 精确 O(prefix) 命中 + 零拷贝前缀复用） | SGLang | Zheng et al., "SGLang", 2024 (arXiv:2312.07104) |
| PagedAttention（物理固定大小块池 + 块表 + ~0 碎片） | vLLM | Kwon et al., 2023 |
| **RadixPagedKVCache（本栈组合贡献）** | 将两者融合：整块共享消除并发写竞争，O(1) 元数据回滚使投机解码无损 | 组合创新，非新算法 |

## 7. Tokenizer

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| 字节级 BPE（字节↔unicode 映射，Ġ=空格） | Radford et al., GPT-2, 2019 |
| BPE 合并排序 | Sennrich et al., 2016 (arXiv:1508.07909) |

## 8. 智能体

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| ReAct（推理 + 行动交错） | Yao et al., 2023 (arXiv:2210.03629) |
| StateGraph 状态图编译 | LangGraph | LangChain AI, "LangGraph", 2024 |

## 9. 服务

| 组件 | 复用来源 | 出处 |
|------|----------|------|
| OpenAI 兼容 `/v1/chat/completions` 协议 | OpenAI API 规范（公共契约，非具体实现） |

---

## 验证方式（不变量 → 测试）

每个「复用」都附带一个可由独立实现交叉验证的数值不变量，位于 `tests/`：

- RoPE 保正交：`test_rope_preserves_norm_per_position`
- FlashAttention ≈ 朴素注意力：`test_flash_matches_naive`
- INT4 重建误差上界：`test_absmax_error_within_theoretical_bound`
- k-quant 字节级反量化：`test_q2_k_block_layout` / `test_q4_k_block_layout` / `test_q6_k_block_layout` / `test_q8_0_block_layout`
- 增量解码 ≡ 全量预填充：`test_incremental_equals_full_prefill`
- 投机解码无损（贪心 / 随机）：`test_greedy_losslessness` / `test_stochastic_distribution_equivalence`
- 前缀缓存加速：`test_prefix_cache_speeds_second_call`

> 若未来出现更强现成实现，本项目承诺**立即替换**自写代码（零重写原则）。
