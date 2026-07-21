# GLM5 推理代码路径导航

> 覆盖：模型注册 → Python 整网 → MLA/DSA → C++ 引擎交接 → Prefill/Decode 调度。
> 相关文档：[`glm5_decode_cudagraph_kernel_mapping.md`](./glm5_decode_cudagraph_kernel_mapping.md)、[`glm5_kv_cache_architecture.md`](./glm5_kv_cache_architecture.md)

---

## 1. 总览

GLM5 **没有独立模型类**。主模型注册为 `DeepSeekV2`，Python 执行体是 `GenericMoeModel`；MTP 走 `Glm5Mtp` / `GenericMoeMTPModel`。

```text
HTTP / RPC
  → NormalEngine 调度
  → NormalExecutor::process          # prefill + decode 共用
  → PyWrappedModel::forward
  → GenericMoeModel.forward
  → GenericMoeDecoderLayer
  → MlaAttention (+ Indexer if sparse)
  → fmha_impl (SparseMlaImpl / FlashInfer MLA)
  → DenseMLP / MegaMoE
```

---

## 2. 模型注册与入口

| 角色 | 路径 |
|------|------|
| 注册 `glm_5` → `DeepSeekV2` | `rtp_llm/models/deepseek_v2.py`（约 L988） |
| 注册 `glm_5_mtp` → `Glm5Mtp` | 同文件（约 L933–989） |
| 创建 Python 模型 | `DeepSeekV2._create_python_model()` → `GenericMoeModel` |
| MTP Python 模型 | `Glm5Mtp._create_python_model()` → `GenericMoeMTPModel` |
| Base 触发 new | `rtp_llm/models/base_model.py` → `_create_python_model()` |

```python
# deepseek_v2.py
register_model("glm_5", DeepSeekV2, ["GlmMoeDsaForCausalLM"])
register_model("glm_5_mtp", Glm5Mtp, ["GlmMoeDsaMtpForCausalLM"])
```

其它也会实例化 `GenericMoeModel` 的模型：`Qwen3Moe`、`DeepSeekVL2`、内源 `tbstars2_5` / `mixtbstars`。

---

## 3. Prefill / Decode 核心模块

整层数据流：

```text
embedding
  → [Layer 0..N]
       input_layernorm
       → MlaAttention
            ├─ Q/KV 投影 + norm
            ├─ [sparse] Indexer → topk_indices
            └─ fmha_impl (RoPE + cache + absorb BMM + attn)
       → post_attn_layernorm
       → DenseMLP 或 Router + MegaMoE + SharedExpert
  → final_layernorm → (lm_head 多在 C++ post layers)
```

| 模块 | 路径 | Prefill / Decode |
|------|------|------------------|
| 整网 forward | `models_py/model_desc/generic_moe.py` | 共用；FMHA 外置传入 |
| MLA Attention | `models_py/modules/hybrid/mla_attention.py` | 投影 + Indexer 调度 |
| Indexer（DSA 选 token） | `models_py/modules/hybrid/indexer.py` | `is_prefill` 分叉；decode 可走 fused path |
| Sparse MLA（DSA 算 attn） | `models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py` | Prefill：gather + `flash_mla_sparse_fwd`；Decode：`flash_mla_with_kvcache` |
| Sparse MLA + CP | `.../flashmla_sparse_cp_impl.py` | Prefill CP |
| Dense MLA Prefill | `.../flashinfer_mla_wrapper.py` → `MlaFlashInferPrefillImpl` | `is_prefill=True` |
| Dense MLA Decode | 同文件 → `MlaFlashInferDecodeImpl` | `is_prefill=False`（非 sparse） |
| MegaMoE | `models_py/modules/glm5_mega_moe/` | FFN，阶段共用 |
| FMHA 选型工厂 | `models_py/modules/factory/attention/attn_factory.py` | `get_mla_impl` |
| 实现注册 | `models_py/modules/factory/attention/__init__.py` | 填 `PREFILL_MLA_IMPS` / `DECODE_MLA_IMPS` |

Decode CUDA Graph 与 kernel ↔ 代码映射见：`docs/glm5_decode_cudagraph_kernel_mapping.md`。

---

## 4. MlaAttention / Indexer / Sparse MLA 关系

三者是 **一层 Attention 内的流水线**，不是并列的三种 attention。

| 组件 | 角色 | 输入 → 输出 |
|------|------|-------------|
| **MlaAttention** | 层外壳 | `hidden` → 投影 →（Indexer）→ FMHA → `o_proj` |
| **Indexer** | DSA「检索器」 | `hidden` + `q_lora` → **`topk_indices`**（如 top-2048） |
| **Sparse MLA**（`fmha_impl`） | DSA「注意力算子」 | `q` + KV cache + **`topk_indices`** → attn 输出 |

```text
hidden
  │
  ▼
┌──────────────── MlaAttention ────────────────┐
│  fused_qkv / q_b / kv_a_ln                   │
│           │                                  │
│           ├─(is_sparse)─► Indexer ──► topk   │
│           │                                  │
│           └──────────────► SparseMlaImpl     │
│                              ▲               │
│                              └── topk_indices│
│  o_proj                                      │
└──────────────────────────────────────────────┘
```

- `attn_config.is_sparse=True`：挂 `Indexer`，`fmha_impl` 常为 `SparseMlaImpl`，**必须**有 `topk_indices`。
- `is_sparse=False`：无 Indexer，走 dense FlashInfer MLA，看完整 KV。

**DSA = Indexer（选） + Sparse MLA（算）**，都挂在同一个 `MlaAttention` 上。

### GenericMoeModel → MlaAttention

不是直接 import 调函数，而是：

1. `GenericMoeModel.__init__` 建多层 `GenericMoeDecoderLayer`
2. `use_mla=True` 时，`GenericMoeDecoderLayer` 把 `self.self_attn = MlaAttention(...)`
3. `GenericMoeModel.forward` → `decoder_layer(...)` → `self.self_attn(..., fmha_impl=...)`

---

## 5. `fmha_impl` 如何传入 MlaAttention

`fmha_impl` **不是** `MlaAttention` 成员，而是 **每次 forward 在模型顶层创建、层层下传** 的对象。

```text
C++ PyWrappedModel / CudaGraphRunner
  │  prepare_fmha_impl(inputs) → fmha_impl
  │  forward(inputs, fmha_impl)
  ▼
GenericMoeModel.forward(inputs, fmha_impl)   # None 则内部再 prepare
  ▼
GenericMoeDecoderLayer.forward(..., fmha_impl)
  ▼
MlaAttention.forward(..., fmha_impl=fmha_impl)
  ▼
fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, layer_idx, topk_indices)
```

创建入口：`GptModelBase.prepare_fmha_impl` → `AttnImplFactory.get_fmha_impl` → MLA 时走 `get_mla_impl`。

选型依据：`inputs.attention_inputs`（`is_prefill`、`is_target_verify`、`is_sparse`、CP 等）。
**一次 forward 共用一个 `fmha_impl`，所有层复用。**

C++ 侧创建调用：

```cpp
// PyWrappedModel.cc — eager
held_attn_pyobj_ = py_model_.attr("prepare_fmha_impl")(py_model_inputs, false);
auto outputs = py_model_.attr("forward")(py_model_inputs, held_attn_pyobj_);

// CudaGraphRunner — capture/replay 绑同一 prepare_fmha_impl / forward
```

---

## 6. Python 建模 → C++ 引擎交接

| 步骤 | 位置 | 做什么 |
|------|------|--------|
| 1. new 模型 | `BaseModel` / `DeepSeekV2._create_python_model` | `self.py_model = GenericMoeModel(...)` |
| 2. 建引擎 | `model_factory.py` → `create_engine` → `LanguageCppEngine` | 包一层 `RtpLLMOp` |
| 3. start | `async_decoder_engine/rpc_engine.py` → `rtp_llm_op.start()` | 调 C++ `RtpLLMOp.init(model, ...)` |
| 4. 取 py_model | `cpp/pybind/.../RtpLLMOp.cc` `initModel` | `model.attr("py_model")` 写入 `EngineInitParams` |
| 5. 包 Executor | `NormalExecutor` 构造 | `model_ = new PyWrappedModel(..., py_model)` |
| 6. 每步推理 | `NormalExecutor::process` | `model_->forward(model_input)` |
| 7. 回调 Python | `PyWrappedModel::forward` | `prepare_fmha_impl` + `py_model.forward` |

分工：

| 侧 | 职责 |
|----|------|
| Python | 加载权重、建 `GenericMoeModel`、层算子（MLA/MoE） |
| C++ `NormalEngine` / `NormalExecutor` | 组 batch、KV cache、sampler、循环调度 |
| C++ `PyWrappedModel` | 把 `GptModelInputs` 转成对 `py_model` 的 pybind 调用 |

---

## 7. NormalExecutor：Prefill 与 Decode

**能跑两者。** `NormalExecutor` 是普通路径上 prefill + decode **共用** 的 executor。

一步内可以：

- 只有 context（prefill）
- 只有 decode
- 混合 batch（metrics 有 `*_when_has_context`）

体现：

```cpp
// NormalExecutor.cc — process()
model_forward(ctx_batch=%zu, gen_batch=%zu, ...)
  totalContextBatchSize()   // prefill
  totalDecodeBatchSize()    // decode
model_output = model_->forward(model_input);
```

阶段区分不在 Executor 拆成两个函数，而在：

1. `NormalBatchStreamProcessor::gatherModelInput` 组出带 `is_prefill` 的 `GptModelInputs`
2. `PyWrappedModel` / `AttnImplFactory` 按 `is_prefill` 选 FMHA / CUDA Graph

| 场景 | Executor |
|------|----------|
| 普通 serving（无 MTP） | `NormalExecutor`：prefill + decode |
| Speculative / MTP | `MtpExecutor`（也覆盖两阶段，多 draft/verify） |
| PD 分离 | Prefill / Decode 节点可各跑一份 engine；角色由部署配置决定 |

---

## 8. 快速跳转索引

```text
注册 / 权重
  rtp_llm/models/deepseek_v2.py

Python 整网
  rtp_llm/models_py/model_desc/generic_moe.py
  rtp_llm/models_py/model_desc/generic_moe_mtp.py
  rtp_llm/models_py/model_desc/module_base.py   # prepare_fmha_impl

Attention / DSA
  rtp_llm/models_py/modules/hybrid/mla_attention.py
  rtp_llm/models_py/modules/hybrid/indexer.py
  rtp_llm/models_py/modules/factory/attention/attn_factory.py
  rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashmla_sparse_impl.py
  rtp_llm/models_py/modules/factory/attention/cuda_mla_impl/flashinfer_mla_wrapper.py

MoE
  rtp_llm/models_py/modules/glm5_mega_moe/

引擎交接与调度
  rtp_llm/async_decoder_engine/rpc_engine.py
  rtp_llm/ops/rtp_llm/rtp_llm_op.py
  rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc
  rtp_llm/cpp/normal_engine/NormalExecutor.cc
  rtp_llm/cpp/models/PyWrappedModel.cc
  rtp_llm/cpp/cuda_graph/cuda_graph_runner.h|.cc
```
