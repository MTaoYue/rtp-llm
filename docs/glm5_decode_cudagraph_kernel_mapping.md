# GLM5 Decode CUDA Graph Kernel ↔ 代码映射

> 来源：`normal_profiler_wr0_1 (42).json`
> 对象：第 **7** 个 `cudaGraphLaunch`（`model_id=0`，`cuda_graph.forward(replayDecode)`）
> 日期：2026-07-09

---

## 1. 概览

| 项 | 值 |
|----|-----|
| CUDA Graph Launch 序号 | 第 7 / 24 |
| Marker | `py_model.forward(cuda_graph=1,prefill_cg=0,model_id=0)` |
| GPU kernel 数 | **3062** |
| GPU 总耗时（kernel sum） | **~31.88 ms** |
| 层数 | **78**（layer 0–77） |
| Sparse MLA（带 Indexer） | **21** 层 |
| Dense MLA（无 Indexer） | **57** 层 |
| Dense FFN | layer **0–2**（`first_k_dense_replace=3`） |
| MoE FFN | layer **3–77**（75 层） |

### 1.1 模型配置（由 GEMM 维度反推，与文档一致）

| 参数 | 值 | 证据 |
|------|-----|------|
| `hidden_size` | 6144 | GEMM `K=6144` / `N=6144` |
| `q_lora_rank` | 2048 | `q_b` 的 `K=2048` |
| `kv_lora_rank` | 512 | fused QKV `N=2048+512+64=2624` |
| `qk_rope_head_dim` | 64 | 同上 |
| `qk_nope_head_dim` | 192 | `q_b` `N=64×256=16384` |
| `num_attention_heads` | 64 | |
| `v_head_dim` | 256 | `o_proj` `K=64×256=16384` |
| `intermediate_size`（dense） | 12288 | up `N=2×12288=24576` |
| `moe_intermediate_size` / shared | 2048 | shared up `N=2×2048=4096` |
| Indexer heads × dim | 32 × 128 | `wq_b` `N=4096` |
| Indexer topk | 2048 | `FilteredTopKUnifiedKernel<..., 2048u>` |

### 1.2 层类型分布

| 类型 | Layer ID | 数量 |
|------|----------|------|
| Sparse MLA + Dense FFN | 0, 1, 2 | 3 |
| Dense MLA + MoE | 3–5, 7–9, …（非 4k） | 57 |
| Sparse MLA + MoE | 6, 10, 14, …, 74（每 4 层一个） | 18 |

Sparse 层规律：`layer % 4 == 2` 或 `layer < 3`（即 0,1,2 以及 6,10,…,74）。

### 1.3 耗时占比（第 7 个 Graph）

| 类别 | 时间 | 占比 | kernel 数 |
|------|------|------|-----------|
| MoE GEMM (`mega_moe`) | 12.38 ms | 38.8% | 150 |
| MLA Attention (sparse + combine) | 4.66 ms | 14.6% | 157 |
| Indexer | 2.95 ms | 9.3% | 105 |
| MLA `o_proj` | 2.15 ms | 6.8% | 78 |
| MoE Router | 1.53 ms | 4.8% | 375 |
| FP8 Quant | 1.13 ms | 3.5% | 453 |
| MLA Absorb BMM (`W_kc`/`W_vc`) | 0.90 ms | 2.8% | 156 |
| Norm | 0.88 ms | 2.8% | 334 |
| 其它（copy / q_b / fused_qkv / shared expert / …） | ~5.3 ms | ~16.6% | — |

---

## 2. 整网数据流

```
embedding
  → [Layer 0..77] ×
       input_layernorm (FusedAddRMSNorm)
       → MlaAttention
            ├─ fused_qkv_a_proj / q_a_ln / q_b_proj / kv_a_ln
            ├─ [sparse] Indexer → topk_indices
            └─ fmha_impl (RoPE + cache + absorb BMM + sparse MLA + o_proj)
       → post_attn_layernorm (FusedAddRMSNorm)
       → DenseMLP (L0–2)  或  Router + MegaMoE + SharedExpert (L3–77)
  → final_layernorm
  → (lm_head 不在本 graph / 或仅 memcpy)
```

主要代码入口：

| 模块 | 路径 |
|------|------|
| MLA Attention | `rtp_llm/models_py/modules/hybrid/mla_attention.py` |
| Indexer | `rtp_llm/models_py/modules/hybrid/indexer.py` |
| Sparse FMHA | `rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py` |
| Dense MLP | `rtp_llm/models_py/modules/hybrid/dense_mlp.py` |
| MegaMoE | `rtp_llm/models_py/modules/glm5_mega_moe/` |

---

## 3. Graph 头尾

| # | Kernel | 代码 / 含义 |
|---|--------|-------------|
| 0 | `embedding_lookup_kernel_vec` | Token embedding |
| 1 | `vectorized_elementwise` | 小 elementwise（位置/类型相关） |
| … | 78 × Transformer layer | 见下文 |
| 末尾 | `FusedAddRMSNorm` | Final RMSNorm |
| 末尾 | `memcpy32_post` | Graph 收尾拷贝 |

---

## 4. MlaAttention（每层都有）

对应 `MlaAttention.forward`（`mla_attention.py`）。

### 4.1 本 Profile 的融合状态

| 融合开关 | 状态 | 证据 |
|----------|------|------|
| F1（`q_a_layernorm` fused strided / fp8 dual） | **关闭** | 出现 `torch_copy` + `RMSNorm` + 独立 `fp8_quant` |
| F2（`kv_a_layernorm` fused strided） | **关闭** | 同上 |
| F1b 开启时预期 | 单 kernel `fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output` | 本次未出现 |

实际走的是 unfused 分支：

```python
# mla_attention.py
q_c = self.q_a_layernorm(q.contiguous())
q = self.q_b_proj(q_c)
compressed_kv = self.kv_a_layernorm(compressed_kv.contiguous())
```

### 4.2 Dense MLA 层 kernel 序列（无 Indexer）

以 Layer 3（kernel idx 129–147）为例：

| 序 | Profiler Kernel | 代码 | 说明 |
|----|-----------------|------|------|
| 0 | `FusedAddRMSNorm` | Layer `input_layernorm` | residual + RMSNorm |
| 1 | `per_token_group_quant_8bit(ue8m0)` | Linear 输入量化 | → `x_fp8, x_scale` |
| 2 | `DGEMM(N=2624, K=6144)` | `fused_qkv_a_proj` | `q_lora+kv_lora+rope` |
| — | （无 kernel） | `torch.split` | view，不落 GPU |
| 3 | `torch_copy` | `q.contiguous()` | F1 off |
| 4 | `RMSNorm` | `q_a_layernorm` | |
| 5 | `fp8_quant(ue8m0)` | `q_b_proj` 输入量化 | |
| 6 | `DGEMM(N=16384, K=2048)` | `q_b_proj` | `64×256` |
| 7 | `torch_copy` | `compressed_kv.contiguous()` | F2 off |
| 8 | `RMSNorm` | `kv_a_layernorm` | |
| 9–15 | 见 §5 FMHA | `fmha_impl.forward` | RoPE→cache→absorb→MLA→W_vc |
| 16 | `fp8_quant(ue8m0)` | `o_proj` 输入量化 | |
| 17 | `DGEMM(N=6144, K=16384)` | `o_proj` | |

### 4.3 Sparse MLA 层额外段

在 `kv_a_layernorm` 之后、`fmha_impl.forward` 之前插入 Indexer（§6），由：

```python
topk_indices = self._run_sparse_indexer(...)
attn_output = fmha_impl.forward(q_view, compressed_kv, k_pe, kv_cache, layer_idx, topk_indices)
```

---

## 5. FMHA / Absorb（`flashmla_sparse_impl.py`）

由 `MlaAttention` 调用 `fmha_impl.forward`，实现在 `FlashMlaSparseImpl`：

| Kernel | 代码方法 | 作用 |
|--------|----------|------|
| `BatchQKApplyRotaryPosIdsCosSinCache...` | `rope_impl.forward(q_pe, k_pe, ...)` | Q/K RoPE |
| `concat_and_cache_ds_mla_kernel` | `kv_cache_write_op.forward` | 写 MLA KV cache（FP8） |
| `_strided_slice_copy_kernel` | `_apply_input_bmm` → `strided_slice_copy_` | 把 `q_pe` 拷进 absorb buffer |
| `nvjet_sm103_tst_256x56_...` | `_apply_input_bmm` → `torch.bmm(q_nope, W_kc)` | Absorb：`q_nope @ W_kc` → `kv_lora_rank` |
| `_convert_req_index_to_global_index_kernel` | topk 索引转换 | req-local → global |
| `get_mla_metadata_kernel` | 调度 meta（全 graph 仅 1 次） | |
| `flash_fwd_splitkv_mla_fp8_sparse_kernel` | `fmha_impl.forward` | Sparse MLA 主计算 |
| `flash_fwd_mla_combine_kernel` | combine split-KV | |
| `nvjet_sm103_tst_128x56_...` | `_apply_output_bmm` → `torch.bmm(..., W_vc)` | `attn @ W_vc` → `v_head_dim` |

对应代码骨架：

```python
# flashmla_sparse_impl.py::forward
self.rope_impl.forward(q_pe, k_pe, ...)
self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, ...)
q_transformed = self._apply_input_bmm(q, layer_id)   # strided_slice + BMM W_kc
attn_output = self.fmha_impl.forward(q_transformed, kv_input, topk_indices, ...)
return self._apply_output_bmm(attn_output, layer_id)  # BMM W_vc
```

---

## 6. Indexer（仅 Sparse 层，21 次）

对应 `Indexer.forward` → `_get_q_k_bf16` + `_quantize_q_k` + `_get_logits_head_gate`（`indexer.py`）。

本 profile **未走** `_fused_forward_decode`（否则 Q 的 RoPE+Hadamard+FP8 会合成单个 Triton kernel，不会出现独立的 `nvjet` Hadamard GEMM）。`weights_proj` 也走 **非 F3** 路径（独立 `cutlass_sgemm` + `mul`×2）。

| 序 | Kernel | 代码 | 维度 / 说明 |
|----|--------|------|-------------|
| 1 | `fp8_quant` + `DGEMM(4096×2048)` | `wq_b(q_lora)` | indexer Q：`32×128` |
| 2 | `fp8_quant` + `DGEMM(128×6144)` | `wk(x)` | indexer K |
| 3 | `generalLayerNorm` | `k_norm` | |
| 4 | `_fused_rope_qk_had_k_kernel` | `fused_prefill_rope_hadamard_qk` | Q in-place RoPE + K(RoPE+Hadamard→bf16) |
| 5 | `nvjet(64x24)` | 同上：`q.view(-1,D) @ H_scaled` | **Q 的 Hadamard**（cuBLAS bf16 GEMM，128×128） |
| 6 | `fp8_quant(float)` | `quant_q_k` / `sgl_per_token_group_quant_fp8` | Q → FP8（float scale，非 ue8m0 packed） |
| 7 | `indexer_k_quant_and_cache` | `quant_k_only` / `quant_q_k` | 写 indexer K cache |
| 8 | `cutlass_80_simt_sgemm` | `weights_proj(x.float())` | **真正的 weights_proj**（fp32 GEMV） |
| 9 | `mul` × 2 | `weights * q_scale * scale` | `_get_logits_head_gate` |
| 10 | `torch_copy` / `arange` | topk 准备 | |
| 11 | `sm100_fp8_paged_mqa_logits` | `_get_topk_paged` 打分 | |
| 12 | `FilteredTopKUnifiedKernel<..., 2048>` | topk=2048 | → `topk_indices` |

> **更正**：`nvjet(64x24)` **不是** `weights_proj`。本 profile 走的是 `_get_q_k_bf16`（非 `_fused_forward_decode`），Q 的 Hadamard 用 cuBLAS GEMM 实现。

调用链：

```python
# indexer.py::forward  (F3 / fused-decode 未开时)
query, key = self._get_q_k_bf16(...)          # → apply_rope_and_rotate_q_k
q_fp8, q_scale = self._quantize_q_k(...)      # fp8_quant + indexer_k_cache
weights = self._get_logits_head_gate(...)     # cutlass_sgemm = weights_proj
return self._compute_topk(q_fp8, weights, ...)

# fused_prefill_rope_hadamard.py
_fused_rope_qk_had_k_kernel(...)              # Triton: RoPE(Q)+RoPE/Had(K)
q_out = (q.view(-1, D) @ H_scaled).view(...)  # ← nvjet(64x24)
```

---

## 7. Dense FFN（Layer 0–2）

对应 `DenseMLP.forward`（`dense_mlp.py`）。
`intermediate_size=12288`，SiGLU → up 输出 `2×12288=24576`。

本 profile **未开** `silu_and_mul + fp8_quant` 融合（仍是分离的 `SiLU*mul` + `fp8_quant`）。

| 序 | Kernel | 代码 | 说明 |
|----|--------|------|------|
| 0 | `FusedAddRMSNorm` | post-attn LN | |
| 1 | `fp8_quant(ue8m0)` | up 输入量化 | |
| 2 | `DGEMM(N=24576, K=6144)` | `up_proj`（gate+up 合并） | |
| 3 | `act_and_mul_kernel(silu)` | `FusedSiluAndMul` / `act_fn` | |
| 4 | `fp8_quant(ue8m0)` | down 输入量化 | |
| 5 | `DGEMM(N=6144, K=12288)` | `down_proj` | |

```python
# dense_mlp.py
up = self.up_proj(x_fp8, input_scales=x_scale)
activated = self.act_fn(up)          # SiLU*mul
output = self.down_proj(activated)
```

---

## 8. MoE FFN（Layer 3–77）

结构：

```
post_attn_layernorm
  → Router (gate GEMM + sigmoid + topk)
  → pack_mega_moe_inputs
  → deep_gemm::sm100_fp8_fp8_mega_moe   # routed experts
  → Shared Expert MLP (up / silu / down) # 本 profile 与 mega 分离
  → residual_add
```

代码主要在：

- Router：MoE router / gate 路径（`nvjet` splitK gate + `topk_with_k2` + `group_idx_and_topk`）
- Pack：`glm5_mega_moe/input_packer_triton.py` → `_pack_mega_moe_inputs_optimized_kernel`
- Routed：`glm5_mega_moe/mega_moe.py` / `mega_moe_fused.py` → `deep_gemm.fp8_fp4_mega_moe*`
- Shared Expert：等价 `DenseMLP`，`intermediate=2048`

### 8.1 Kernel 序列（以 Layer 3 为例）

| 序 | Kernel | 模块 | 说明 |
|----|--------|------|------|
| 0 | `FusedAddRMSNorm` | post-attn LN | |
| 1 | `nvjet(64x8_splitK)` + `cublasLt::splitKreduce` | Router gate GEMM | `hidden → n_experts` |
| 2 | `torch_copy` | | |
| 3 | `sigmoid` + `add_f32` | Router score | |
| 4 | `topk_with_k2` | Expert topk | |
| 5 | `group_idx_and_topk_idx` | 分组 / 本地 expert 映射 | |
| 6 | `_pack_mega_moe_inputs_optimized_kernel` | Input packer | BF16→FP8 + router 元数据 |
| 7 | `sm100_fp8_fp8_mega_moe_impl` | Routed MegaMoE | **单层最重 kernel**（~150–470 µs） |
| 8 | `fp8_quant` + `DGEMM(4096×6144)` | Shared expert up | `2×2048` |
| 9 | `SiLU*mul` | Shared expert act | |
| 10 | `fp8_quant` + `DGEMM(6144×2048)` | Shared expert down | |
| 11 | `residual_add(bf16)` | MoE 输出 + residual | |

> 注：`mega_moe_fused.forward_with_shared_expert` 可以把 shared 折进 mega kernel；本 profile 仍看到独立 shared GEMM，说明当前 capture 路径是 **routed mega + 分离 shared expert**。

---

## 9. 三类 Layer 完整对照表

### 9.1 Type A：Sparse MLA + Dense FFN（L0–2）

```
FusedAddRMSNorm                          # input LN
fp8_quant → DGEMM(2624×6144)             # fused_qkv_a_proj
torch_copy → RMSNorm                     # q_a_layernorm (F1 off)
fp8_quant → DGEMM(16384×2048)            # q_b_proj
torch_copy → RMSNorm                     # kv_a_layernorm (F2 off)
── Indexer (§6) ──
RoPE → cache_mla → strided_slice → BMM_Wkc
req2global → MLA_sparse → MLA_combine → BMM_Wvc
fp8_quant → DGEMM(6144×16384)            # o_proj
FusedAddRMSNorm                          # post-attn LN
fp8_quant → DGEMM(24576×6144)            # dense up
SiLU*mul → fp8_quant → DGEMM(6144×12288) # dense down
```

约 **42–43** kernels / layer。

### 9.2 Type B：Dense MLA + MoE（多数层）

```
FusedAddRMSNorm
fp8_quant → DGEMM(2624×6144)             # fused_qkv_a
torch_copy → RMSNorm → fp8 → DGEMM(16384×2048)  # q_a / q_b
torch_copy → RMSNorm                     # kv_a
RoPE → cache → strided_slice → BMM_Wkc
req2global → MLA_sparse → MLA_combine → BMM_Wvc
fp8_quant → DGEMM(6144×16384)            # o_proj
FusedAddRMSNorm
gate_GEMM → splitK_reduce → sigmoid → topk → pack
MoE_mega
fp8 → DGEMM(4096×6144) → SiLU → fp8 → DGEMM(6144×2048)  # shared
residual_add
```

约 **35** kernels / layer。

### 9.3 Type C：Sparse MLA + MoE（L6,10,…,74）

= Type B 的 MLA 段换成 Type A 的 Indexer+Sparse MLA 段。
约 **52** kernels / layer。

---

## 10. GEMM 维度速查

| Kernel 模板 | N × K | 算子 |
|-------------|-------|------|
| `DGEMM(2624×6144)` | 2624 × 6144 | `fused_qkv_a_proj` |
| `DGEMM(16384×2048)` | 16384 × 2048 | `q_b_proj` |
| `DGEMM(4096×2048)` | 4096 × 2048 | Indexer `wq_b` |
| `DGEMM(128×6144)` | 128 × 6144 | Indexer `wk` |
| `DGEMM(6144×16384)` | 6144 × 16384 | `o_proj` |
| `DGEMM(24576×6144)` | 24576 × 6144 | Dense FFN up |
| `DGEMM(6144×12288)` | 6144 × 12288 | Dense FFN down |
| `DGEMM(4096×6144)` | 4096 × 6144 | Shared expert up |
| `DGEMM(6144×2048)` | 6144 × 2048 | Shared expert down |
| `nvjet(256x56)` | BMM | Absorb `W_kc` |
| `nvjet(128x56)` | BMM | Absorb `W_vc` |
| `nvjet(64x8_splitK)` | — | MoE router gate |
| `sm100_fp8_fp8_mega_moe` | — | Routed experts |

---

## 11. 与代码融合开关的差异（优化机会）

本 CUDA Graph 捕获时多项融合 **未开启**，因此 kernel 数偏多：

| 位置 | 当前（profile） | 开启融合后预期 |
|------|-----------------|----------------|
| `q_a_layernorm` | `copy + RMSNorm + fp8_quant` | `fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output`（F1b） |
| `kv_a_layernorm` | `copy + RMSNorm` | `fused_strided_rmsnorm`（F2） |
| Indexer `weights_proj` | `sgemm + mul×2` | `fused_logits_head_gate`（F3） |
| DenseMLP / Shared silu | `SiLU*mul + fp8_quant` | `silu_and_mul_per_token_group_fp8_quant_*` |
| Shared + Routed MoE | 分离 | `fp8_fp4_mega_moe_fused`（含 shared） |

控制开关：`HWKernelConfig.enable_fuse_kernels` / 环境变量 `ENABLE_FUSE_KERNELS`（见 `fuse_config.py`）。

---

## 12. 关键文件索引

| 文件 | 角色 |
|------|------|
| `models_py/modules/hybrid/mla_attention.py` | MLA 主模块 |
| `models_py/modules/hybrid/indexer.py` | DSA Indexer |
| `models_py/modules/hybrid/dense_mlp.py` | Dense / Shared MLP |
| `models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py` | Sparse MLA + absorb BMM |
| `models_py/modules/factory/attention/cuda_mla_impl/mla_kv_cache_write_op.py` | KV cache 写入 |
| `models_py/triton_kernels/common/strided_slice_copy.py` | absorb 前 `q_pe` 拷贝 |
| `models_py/triton_kernels/common/fused_strided_rmsnorm.py` | F1/F2（本 profile 未用） |
| `models_py/modules/glm5_mega_moe/mega_moe.py` | Routed MegaMoE |
| `models_py/modules/glm5_mega_moe/mega_moe_fused.py` | MegaMoE + Shared 融合版 |
| `models_py/modules/glm5_mega_moe/input_packer_triton.py` | MoE input pack |
| `models_py/modules/factory/linear/impl/cuda/fp8_gemm_linear.py` | FP8/FP4 Linear → DeepGEMM |

---

## 附录 A：Sparse Layer 0 完整 kernel 列表（idx 2–44）

| idx | µs | Kernel |
|-----|-----|--------|
| 2 | 3.20 | FusedAddRMSNorm |
| 3 | 2.59 | fp8_quant(ue8m0) |
| 4 | 9.41 | DGEMM(2624×6144) fused_qkv_a |
| 5 | 3.10 | torch_copy |
| 6 | 2.34 | RMSNorm (q_a) |
| 7 | 2.08 | fp8_quant |
| 8 | 9.73 | DGEMM(16384×2048) q_b |
| 9 | 2.78 | torch_copy |
| 10 | 2.30 | RMSNorm (kv_a) |
| 11 | 2.11 | fp8_quant |
| 12 | 6.78 | DGEMM(4096×2048) indexer wq_b |
| 13 | 2.56 | fp8_quant |
| 14 | 7.97 | DGEMM(128×6144) indexer wk |
| 15 | 2.30 | generalLayerNorm |
| 16 | 3.10 | fused_rope_qk_had_k (RoPE Q + RoPE/Had K) |
| 17 | 2.66 | nvjet(64x24) = Q Hadamard GEMM (`q @ H_scaled`) |
| 18 | 2.02 | fp8_quant(float) (Q quant) |
| 19 | 2.85 | indexer_k_quant_and_cache |
| 20 | 3.52 | torch_copy |
| 21 | 43.74 | cutlass_sgemm = weights_proj |
| 22–23 | ~1.5 | mul × 2 |
| 24 | 3.68 | torch_copy |
| 25 | 1.02 | arange |
| 26 | 48.19 | paged_mqa_logits |
| 27 | 44.70 | FilteredTopK(2048) |
| 28 | 2.14 | RoPE |
| 29 | 2.59 | concat_and_cache_mla |
| 30 | 2.08 | strided_slice_copy |
| 31 | 6.40 | BMM W_kc |
| 32 | 2.05 | convert_req_index_to_global |
| 33 | 1.95 | get_mla_metadata |
| 34 | 52.51 | flash_fwd_splitkv_mla_fp8_sparse |
| 35 | 6.88 | mla_combine |
| 36 | 6.24 | BMM W_vc |
| 37 | 3.36 | fp8_quant |
| 38 | 27.71 | DGEMM(6144×16384) o_proj |
| 39 | 2.91 | FusedAddRMSNorm |
| 40 | 2.56 | fp8_quant |
| 41 | 27.43 | DGEMM(24576×6144) dense up |
| 42 | 4.03 | SiLU*mul |
| 43 | 3.23 | fp8_quant |
| 44 | 21.06 | DGEMM(6144×12288) dense down |

---

## 附录 B：如何复现解析

```python
import json
path = "normal_profiler_wr0_1 (42).json"
events = json.load(open(path))["traceEvents"]
launches = sorted(
    [e for e in events if e.get("name") == "cudaGraphLaunch"],
    key=lambda e: e["ts"],
)
corr = launches[6]["args"]["correlation"]  # 第 7 个
kernels = sorted(
    [e for e in events if e.get("cat") == "kernel"
     and e.get("args", {}).get("correlation") == corr],
    key=lambda e: e["ts"],
)
print(len(kernels), sum(e["dur"] for e in kernels) / 1000, "ms")
```
