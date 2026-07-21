# GLM-5 MLA：Prefill / Decode 与 Absorb 机制

> 相关文档：[`glm5_kv_cache_architecture.md`](./glm5_kv_cache_architecture.md)、[`glm5_inference_code_path.md`](./glm5_inference_code_path.md)

本文说明 GLM-5（复用 DeepSeekV2 MLA 路径）在 **prefill** 与 **decode** 下 attention 为何形态不同，以及 **absorb（权吸收）** 的含义、维度变化和性能取舍。

---

## 1. 结论先行

1. Prefill / Decode **共用同一套 MLA 配置**（head dim、`kv_lora_rank` 等不变）。
2. 差异在 **实现路径**：长 prefill 常把压缩 KV **展开**成多头 K/V 再算；decode 常把 \(W_K/W_V\) **吸收**进 Q 与输出，直接对 cache 里的 latent 算。
3. Absorb 会让进 attention 的 Q/K/V **变长**（对齐 latent），不是变短；**变短的是 KV cache 相对完整 MHA 的存储**。
4. Decode 的两次 BMM（`mla_kc` / `mla_vc`）只作用在**当前 token**；相对「每步把整段历史 KV 展开」，几乎总是更省。
5. GLM-5 开 DSA（sparse）时，prefill/decode **都倾向 absorb + top-k**，不必先按「prefill=MHA、decode=MQA」理解。

---

## 2. 共享的 MLA 维度（GLM-5）

| 参数 | 典型值 | 含义 |
|------|--------|------|
| `qk_nope_head_dim` | 128 | Q/K 非 RoPE 部分 |
| `qk_rope_head_dim` | 64 | Q/K RoPE 部分 |
| `q_head_dim` | **192** (=128+64) | 完整 Q（expand 路径） |
| `v_head_dim` | 128 | 最终 V / `o_proj` 输入维 |
| `kv_lora_rank` | **448** | 压缩 KV latent（DeepSeek 多为 512） |
| Cache 每 token | **512** (=448+64) | `compressed_kv \|\| k_pe` |

投影入口（`MlaAttention`）两边相同：得到 `q[T,H,192]`、`compressed_kv[T,448]`、`k_pe[T,64]`，再写入同一套 MLA paged cache。

```text
hidden
  → fused_qkv_a_proj → q_lora / compressed_kv / k_pe
  → q_a_ln + q_b_proj → q [T, H, 192]
  → kv_a_ln → compressed_kv [T, 448]
  → 写入 cache: [compressed_kv || k_pe]  # 每 token 512
  → FMHA（prefill expand / decode absorb / sparse …）
  → o_proj
```

---

## 3. 两条计算路径

### 3.1 Expand（长 Prefill 常用）— 先还原 K/V

```text
c [448] ── kv_b_proj ──► K_nope [H,128], V [H,128]
K = [K_nope || k_pe]     # [H, 192]
Q = [q_nope || q_pe]     # [H, 192]

attn = softmax(Q K^T / √192) V   → [H, 128]
```

- Kernel：FlashInfer `BatchPrefillWithRaggedKVCacheWrapper`（ragged / 类 MHA）
- Attention 内：**QK=192，V=128**
- 需要显式 `kv_b_proj`（把 latent 展开成 per-head K/V）

### 3.2 Absorb（Decode 默认；短 Prefill 也可）— 不还原历史 KV

利用线性与结合律，把「还原 K/V」挪到 Q 侧和输出侧：

```text
① q_nope [128] × W_kc(=mla_kc) → q' [448]
   Q = [q' || q_pe]              → [512]

② Attention 直接对 cache：
   K_cache = [c || k_pe]         # [512]
   score   = Q · K_cache^T / √192
   o'      = softmax(score) · c  # V 用 c，dim=448

③ o' [448] × W_vc(=mla_vc) → out [128]
```

- BF16：FlashInfer `BatchMLAPagedAttentionWrapper`
- FP8：`flash_mla_with_kvcache(..., head_dim_v=kv_lora_rank)`
- **不对历史做 `kv_b_proj`**；cache 原样读

### 3.3 Prefill 何时也会 Absorb

`MlaFlashInferPrefillImpl` 在短序列且（有 prefix reuse 或 CUDA Graph）时，可挂 `absorb_fmha = MlaFlashInferDecodeOp`：

- 条件大意：`q_len < absorb_opt_len`（默认 1024）且允许 absorb
- 计算形态与 decode 相同

### 3.4 DSA Sparse（GLM-5 主模型常见）

`is_sparse=True` 时，Indexer 选出 top-k，再跑 Sparse MLA。
**Prefill / Decode 都走 absorb**（`W_kc` → sparse attn → `W_vc`），差异主要在 kernel / Indexer，而不是 expand vs absorb。

---

## 4. Absorb 通俗解释

### 4.1 在干什么

MLA 不直接存完整 K/V，只存一段压缩「密码」\(c\)。要用时本来要：

```text
c × 配方_K → 真正的 K
c × 配方_V → 真正的 V
```

**Absorb = 别每次把历史密码都还原成 K/V，改成等价算法：**

1. 用「还原 K 的配方」**改写当前问题 Q**（Q 变长）
2. 改写后的 Q 直接和密码 \(c\) 比对
3. 对密码做完加权汇总后，再用「还原 V 的配方」变回 128 维输出

数学结果（忽略精度）与「先还原再算」一致。

### 4.2 为什么「能」吸

- \(K = c\,W_K\)、\(V = c\,W_V\) 都是**线性**变换
- 点积 / 加权和满足结合律：
  - \(q(cW_K)^\top = (q W_K^\top)\,c^\top\) → \(W_K\) 吸进 Q（`mla_kc`）
  - \(A(cW_V) = (Ac)\,W_V\) → \(W_V\) 吸到输出（`mla_vc`）
- Softmax 仍在中间，吸的是它**两侧**的线性部分
- **RoPE 段不能吸**（\(q_{pe}\cdot k_{pe}\) 没有「\(c\mapsto K\)」这种固定线性层）

### 4.3 吸收是变长，不是变短

| | Expand | Absorb |
|--|--------|--------|
| Q 进 attn | 192 | **512** |
| K 进 attn | 192 | **512**（即 cache） |
| V 在 attn 内 | 128 | **448**（即 \(c\)） |
| 层输出（attn 后） | 128 | 128（经 `mla_vc`） |

- Absorb：把 NoPE 的 Q 从 **128 拉长到 448**，以对齐 \(c\)
- MLA「压缩」指的是 **cache 比完整多头 K/V 小**，不是 absorb 把 attention dim 压短

---

## 5. 两次 BMM 与 \(W_K\) / \(W_V\)

权重都来自 checkpoint 的 `kv_b_proj`，加载时切成：

| 名称 | 角色 | Decode 用法 |
|------|------|-------------|
| `mla_kc` | \(W_K\) 侧（吸进 Q） | attn **前**：`q_nope × mla_kc` → 448，再拼 `q_pe` → 512 |
| `mla_vc` | \(W_V\) 侧（吸到输出） | attn **后**：`o' × mla_vc` → 128 |

注意：Decode **没有对 KV 做 up**。路径是：

```text
Q × W_K(吸收) → Attention(对 c) → × W_V(吸收) → 128
```

不是「Q 和 KV 都 up 到 512」。

代码参考：

- `flashinfer_mla.py`：`MlaFlashInferDecodeOp.forward`（`torch.bmm` + MLA wrapper / `flash_mla_with_kvcache`）
- `deepseek_v2.py`：`W.mla_kc` / `W.mla_vc` 从 `kv_b_proj.weight` 经 `transpose_slice_k/v` 得到
- Sparse：`flashmla_sparse_impl.py` 的 `_apply_input_bmm` / `_apply_output_bmm`

---

## 6. 「Decode≈MQA，Prefill≈MHA」对不对？

**半对，需加限定。**

| 说法 | 更准确的含义 |
|------|----------------|
| Decode absorb ≈ MQA | 多头 Q，**一份** shared latent KV（`num_heads_k=1`） |
| Prefill expand ≈ MHA | `kv_b_proj` 后每头有自己的 K/V，kernel 按多头 FMHA 跑 |

要点：

1. 根上始终是 **MLA**，不是两套模型。
2. Prefill expand 的多头 K/V 仍由**同一份** \(c\) 线性投出，不是经典 MHA「每头从 hidden 独立投 KV」；`k_pe` 仍共享。
3. DSA 打开时，prefill/decode **都更像 MQA + top-k**。
4. 短 prefill absorb 短路与 decode 同形。

---

## 7. 性能：Decode 为何选 Absorb

### 7.1 成本落在谁身上

| 路径 | 额外线性代数 | 随什么变贵 |
|------|----------------|------------|
| Absorb | 两次 BMM，只打在**当前** Q / 输出 | 主要跟 **T**（decode 常 ≈1） |
| 每步 Expand KV | `kv_b_proj` 打在**整段历史** | 跟 **S（上下文长度）** 线性涨 |

数量级（示意）：

```text
Absorb BMM：  O(T · H · 128 · 448)      # T≈1，很小
Expand 全历史：O(S · H · 448 · (128+128)) # S 一长就很贵
```

### 7.2 取舍

- Absorb 让 attention head dim **变大**（512/448 vs 192/128），单次 attn kernel 更重一点。
- Decode 瓶颈通常是 **反复读长 KV**；保持压缩 cache、避免每步展开历史，通常更划算。
- 若展开后还缓存多头 K/V，显存回到 MHA 量级，MLA 省 cache 的意义削弱。

### 7.3 何时 Expand 更好

**长 Prefill**：T、S 都大，在小 dim（192/128）上跑成熟 FlashAttention，往往优于全程 512 维 absorb。
因此引擎策略是：**decode / 短序列 → absorb；长 prefill → expand**（DSA 则另论，两侧多 absorb）。

---

## 8. 端到端对照表

| 项目 | Prefill（长，dense expand） | Decode（absorb） |
|------|----------------------------|------------------|
| 是否 `kv_b_proj` 展开历史 | 是 | 否 |
| Q 进 attn | 192 | 512（128→448 后拼 rope） |
| K | 展开后每头 192 | cache 512（共享） |
| V（attn 内） | 128 | 448（\(c\)） |
| 额外 BMM | 无 absorb BMM | `mla_kc` + `mla_vc` |
| Kernel 形态 | 类 MHA ragged prefill | 类 MQA paged MLA |
| Softmax scale | \(1/\sqrt{192}\) | 同（完整 Q dim） |
| 最终到 `o_proj` | `[T,H,128]` | 同 |

---

## 9. 关键代码索引

| 文件 | 内容 |
|------|------|
| `rtp_llm/models_py/modules/hybrid/mla_attention.py` | 投影 + Indexer 调度入口 |
| `.../cuda_mla_impl/flashinfer_mla_wrapper.py` | Prefill/Decode Impl 选型与 absorb 短路 |
| `.../cuda_mla_impl/flashinfer_mla.py` | Expand prefill op / Absorb decode op |
| `.../cuda_mla_impl/flashmla_sparse_impl.py` | DSA Sparse MLA + 两次 BMM |
| `rtp_llm/models/deepseek_v2.py` | MLA 参数解析、`mla_kc`/`mla_vc` 权重切片、`glm_5` 注册 |

---

## 10. 一句话记忆

**MLA 把 KV 压成 \(c\) 存进 cache；Prefill 长序列常把 \(c\) 展开成小 head 做 FMHA；Decode 把 \(W_K/W_V\) 吸到当前 Q 和输出上，Q 变长、历史不展开——两次小 BMM 换整段 KV up 的大开销。**
