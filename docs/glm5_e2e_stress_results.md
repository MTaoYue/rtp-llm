# GLM-5-FP4MoE-FP8 端到端压测结果

- 模型：`/home/zw193905/models/GLM-5-FP4MoE-FP8`
- Speculative：`sp_type=eagle`，`sp_model_type=glm_5_mtp`，`gen_num_per_cycle=3`
- 压测脚本：`internal_source/.claude/skills/single-node-stress-test/stress_test.py`
- 参数：`--input-lengths 1024,4096 --output-lengths 100,500 --concurrency-steps 1,2,4,8 --duration 60 --timeout 600`
- 记录日期：2026-07-20

---

## 第一次：Decode DP2/EP2（卡不足时的降级拓扑）

### 拓扑

| 项 | 配置 |
|---|---|
| Prefill | TP4/EP4，`CUDA_VISIBLE_DEVICES=4,5,6,7`，`concurrency_limit=64`，cuda graph off |
| Decode | **TP1/DP2/EP2**，`CUDA_VISIBLE_DEVICES=1,2`，`concurrency_limit=4`，cuda graph on，`decode_capture_config=1,2,4` |
| 计划拓扑 | Prefill TP4/EP4 + Decode DP4/EP4（8 卡） |
| 降级原因 | Prefill TP4/EP4 on GPU4-7 + Decode DP2/EP2 on GPU1-2 (GPU0 orphaned ~260GiB; EP must divide 256) |
| Speculative | eagle + glm_5_mtp + gen_num_per_cycle 3 |

原始 JSON：`/tmp/glm5_e2e_stress.json`（另有 `/tmp/glm5_e2e_stress.round1.json`）

### input=1024, output=100

| Conc | QPS | Succ% | RT avg | RT p50 | RT p99 | TTFT avg | TTFT p50 | TTFT p99 | TPOT avg |
|-----:|----:|------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|
| 1 | 0.42 | 100.0% | 1383.4ms | 1344.2ms | 2149.2ms | 293.6ms | 284.5ms | 375.4ms | 10.90ms |
| 2 | 0.83 | 100.0% | 1377.3ms | 1357.2ms | 2024.5ms | 329.5ms | 308.1ms | 865.3ms | 10.48ms |
| 4 | 1.50 | 100.0% | 1664.3ms | 1434.4ms | 5819.6ms | 553.2ms | 349.7ms | 4689.0ms | 11.11ms |
| 8 | 2.73 | 100.0% | 1923.7ms | 1634.1ms | 6920.1ms | 689.8ms | 405.9ms | 5542.7ms | 12.34ms |

### input=1024, output=500

| Conc | QPS | Succ% | RT avg | RT p50 | RT p99 | TTFT avg | TTFT p50 | TTFT p99 | TPOT avg |
|-----:|----:|------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|
| 1 | 0.17 | 100.0% | 5097.5ms | 5345.0ms | 5570.3ms | 333.3ms | 329.6ms | 399.8ms | 10.12ms |
| 2 | 0.33 | 100.0% | 5103.1ms | 5471.2ms | 5793.5ms | 287.5ms | 257.7ms | 524.5ms | 10.52ms |
| 4 | 0.58 | 100.0% | 6143.3ms | 6257.3ms | 6890.5ms | 342.0ms | 272.3ms | 841.4ms | 11.83ms |
| 8 | 0.97 | 100.0% | 7461.9ms | 7426.1ms | 10696.7ms | 489.9ms | 314.1ms | 1533.9ms | 14.28ms |

### input=4096, output=100 — server unstable on 4096; aborted

| Conc | QPS | Succ% | RT avg | RT p50 | RT p99 | TTFT avg | TTFT p50 | TTFT p99 | TPOT avg |
|-----:|----:|------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|
| 1 | 627.87 | 0.0% | 3528.4ms | 1996.4ms | 6749.3ms | 747.3ms | 751.6ms | 761.8ms | 27.81ms |

**本次最好 TTFT P50（成功率≥99%）**：257.7ms（input=1024, output=500, conc=2；avg=287.5ms）

---

## 第二次：完整 8 卡（Prefill TP4/EP4 + Decode DP4/EP4）

### 拓扑

| 项 | 配置 |
|---|---|
| Prefill | TP4/EP4，`CUDA_VISIBLE_DEVICES=4,5,6,7`，`concurrency_limit=64`，cuda graph off |
| Decode | **TP1/DP4/EP4**，`CUDA_VISIBLE_DEVICES=0,1,2,3`，`concurrency_limit=8`，cuda graph on，`decode_capture_config=1,2,4,8` |
| Speculative | eagle + glm_5_mtp + gen_num_per_cycle 3 |
| 对齐 | 与 smoke `mla_mtp_mega_moe_cudagraph_pd_full_ckpt` 一致 |

原始 JSON：`/tmp/glm5_e2e_stress_8gpu.json`

### input=1024, output=100

| Conc | QPS | Succ% | RT avg | RT p50 | RT p99 | TTFT avg | TTFT p50 | TTFT p99 | TPOT avg |
|-----:|----:|------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|
| 1 | 0.40 | 100.0% | 1447.4ms | 1217.3ms | 4133.4ms | 268.6ms | 262.7ms | 456.0ms | 11.79ms |
| 2 | 0.88 | 100.0% | 1274.7ms | 1232.1ms | 2467.1ms | 312.0ms | 265.9ms | 1542.9ms | 9.63ms |
| 4 | 1.68 | 100.0% | 1355.8ms | 1275.5ms | 3233.2ms | 336.2ms | 256.4ms | 2024.1ms | 10.20ms |
| 8 | 3.10 | 100.0% | 1564.6ms | 1386.4ms | 3617.7ms | 480.2ms | 290.4ms | 2456.6ms | 10.84ms |

### input=1024, output=500

| Conc | QPS | Succ% | RT avg | RT p50 | RT p99 | TTFT avg | TTFT p50 | TTFT p99 | TPOT avg |
|-----:|----:|------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|
| 1 | 0.18 | 100.0% | 4822.0ms | 4841.6ms | 5162.2ms | 246.8ms | 243.1ms | 285.7ms | 9.25ms |
| 2 | 0.33 | 100.0% | 5131.1ms | 5140.6ms | 5265.2ms | 286.1ms | 263.0ms | 518.0ms | 9.69ms |
| 4 | 0.65 | 100.0% | 5485.5ms | 5518.8ms | 6110.0ms | 349.2ms | 276.5ms | 751.6ms | 10.45ms |
| 8 | 1.20 | 100.0% | 6032.5ms | 5911.7ms | 7375.8ms | 446.1ms | 262.1ms | 1540.6ms | 11.30ms |

### input=4096, output=100 — stopped_early（服务不稳定，后续 combo 中止）

| Conc | QPS | Succ% | RT avg | RT p50 | RT p99 | TTFT avg | TTFT p50 | TTFT p99 | TPOT avg |
|-----:|----:|------:|-------:|-------:|-------:|---------:|---------:|---------:|---------:|
| 1 | 4.55 | 6.2% | 1882.0ms | 1879.3ms | 1997.5ms | 724.4ms | 720.3ms | 782.4ms | 11.58ms |

**本次最好 TTFT P50（成功率≥99%）**：243.1ms（input=1024, output=500, conc=1；avg=246.8ms）

---

## 两次对比摘要（input=1024）

| 指标 | 第一次 DP2/EP2 | 第二次 DP4/EP4 |
|---|---|---|
| 1024/100 conc=1 QPS | 0.42 | 0.40 |
| 1024/100 conc=1 TTFT p50 | 284.5ms | 262.7ms |
| 1024/100 conc=1 TPOT avg | 10.90ms | 11.79ms |
| 1024/100 conc=8 QPS | 2.73 | 3.10 |
| 1024/100 conc=8 TTFT p50 | 405.9ms | 290.4ms |
| 1024/100 conc=8 TPOT avg | 12.34ms | 10.84ms |
| 1024/500 conc=1 QPS | 0.17 | 0.18 |
| 1024/500 conc=1 TTFT p50 | 329.6ms | 243.1ms |
| 1024/500 conc=1 TPOT avg | 10.12ms | 9.25ms |
| 1024/500 conc=8 QPS | 0.97 | 1.20 |
| 1024/500 conc=8 TTFT p50 | 314.1ms | 262.1ms |
| 1024/500 conc=8 TPOT avg | 14.28ms | 11.30ms |

## 说明

- **Conc**：并发请求数；**QPS**：每秒完成请求数；**RT**：端到端延迟；**TTFT**：首 token 时延；**TPOT**：每输出 token 平均时延。
- **input=4096** 两次均出现大量 HTTP 500 / 服务中断，结果不可作为性能基线。
- 第一次 Decode `concurrency_limit=4`，conc=8 会排队；第二次 `concurrency_limit=8`，与计划一致。
