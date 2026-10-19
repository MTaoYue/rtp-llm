# E1PD4 four-video mixed performance smoke

当前 launcher（2026-10-10）不设置 `RTP_TRTLLM_DECODE_BATCH_INVARIANT`，也不将它写入预设环境元数据。未外部传入服务时，算子按默认值 `0` 关闭该开关。下文启用该开关的结果属于历史验收记录。

当前配置（2026-10-09 单开关合并）：只使用 `RTP_BF16_GATE_KERNEL` 控制共享专家标量门控、Router 和 GDN BA 三处修复。普通服务设为 `1` 开启，`0` 或未设置关闭；本混合预设默认开启，显式环境变量 `0` 可关闭并传到 worker/服务。开关在模块构造时读取，修改后需要重建模块/重启服务。

```bash
export RTP_BF16_GATE_KERNEL=1
```

旧 `RTP_BF16_SCALAR_LINEAR`、`RTP_BF16_ROUTER_LINEAR`、`RTP_BF16_GDN_LINEAR`、`RTP_BF16_SCALAR_LINEAR_MAX_ROWS` 不再读取。Decode 所有 M 使用固定归约；小 Prefill 原有范围固定保留（标量 M≤256，Router/GDN 65≤M≤256），大 Prefill 和普通 Decode Graph 行为不变。下面独立开关和旧测试配置属于历史记录。

最新验收（2026-10-09 18:49）：单 kernel 门控优化完成，C512 含排空 1153/1153 成功，全部回答与旧 fixed64 基线逐字一致。M128 Router/GDN kernel 时间分别减少约28%/33%；小 M 有性能代价，端到端吞吐本轮基本持平。配置 chunk=0、Decode Graph=1，详见文末 single-kernel gate optimization。下文保留历史诊断记录。

This mixed preset explicitly injects the following optimization environment:

```ini
RTP_QWEN35_GDN_PREFILL_BACKEND=flashinfer
RTP_QWEN35_FUSED_CONV_QKV_NORM=1
RTP_QWEN35_FUSED_GATED_RMSNORM_FP8=1
RTP_QWEN35_GDN_DECODE_BACKEND=flashinfer
RTP_QWEN35_DECODE_FUSION=1
```

The preset forwards these values through Bazel and applies them to service
configuration after the base profile, so baseline values cannot override them.
Historical validation results below retain the settings used in those runs.

Before removing three unused environment entries on 2026-10-10, validation
of the eight-entry configuration passed all 77 CPU cases. The live Encoder and
PD environments matched the preset, but GPU
preflight failed: videos 1 and 2 returned HTTP 200 with finish_reason=stop;
video 3 returned HTTP 200 with finish_reason=length and 8192 completion tokens,
including consecutive word repetition and missing requested speaker annotations.
The harness stopped before video 4 and the concurrent preflight. All owned
services were cleaned up and GPUs 0,4,5,6,7 returned to idle. The five settings
listed above remain configured as requested; the failure has not been isolated
to a specific option. Evidence is under
`build_logs/e1pd4-qwen35-optimizations-preflight-20261010-01/verification.json`.

Manual Qwen3.5-397B-A17B-FP8 test using four video requests.
Encoder GPU 0; PD Fusion GPUs 4,5,6,7; N7; C512 across four measured client
processes. Warm up for 240 seconds, measure two 300-second windows, then drain.

Inside `tuoyu.ty`, from `/ssd/7/tuoyu.ty/workspace/RTP-LLM-epd-fusion`:

```bash
/opt/conda310/bin/python github-opensource/rtp_llm/test/smoke/qwen35_e1pd4_mixed_video_perf_smoke.py \
  --output "$PWD/build_logs/e1pd4-mixed-$(date +%Y%m%d-%H%M%S)" --execute
```

Omit `--execute` for a side-effect-free command preview. The target is
`//rtp_llm/test/smoke:qwen35_e1pd4_mixed_video_perf_smoke`; use the launcher
for the precheck, CUDA13/SM10x build, dependency checks and passive GPU lock.
The CPU target is `//rtp_llm/test/smoke:qwen35_mixed_video_test`.
Relocation options are inherited from the fixed-video preset, including
`--data-dir`, `--model-dir`, GPU placement, cache and runtime dependency paths.
The default fixture directory is `qwen35_mixed_video_data` next to the launcher.

MegaMoE chunking is disabled by default in this mixed preset. The launcher
forwards `RTP_MEGAMOE_CHUNK_TOKENS=0` through Bazel to the worker and both
services and records it in the preset snapshot. Pass
`--mega-moe-chunk-tokens=8192` to enable chunking explicitly. The fixed-video
reference preset is unchanged. Keep grid sync diagnostics off for performance
runs.

## Workload and compatibility

Encoder uses explicit service arguments `--mm_video_total_min_pixels 2500000`
and `--mm_video_total_max_pixels 36864000`, with matching environment values.
Input expectations use the same budgets. As of 2026-10-10, all four request
JSONs have exactly `preprocess_config={"fps":6,"min_frames":4,"max_frames":90}`.
No request includes pixel limits or resized dimensions, so the existing Encoder
pixel budgets determine resized shapes. This update changes request fields,
not the RTP video processing configuration.

Request-update validation on 2026-10-10 passed 77 CPU tests, fixture checksums,
exact prompt comparison against the local templates, and input accounting
using the current Qwen35Renderer. This update did not rerun GPU load testing.

| Request | Frames | Resized H x W | Prompt tokens | Video tokens |
| --- | ---: | --- | ---: | ---: |
| 1 | 46 | 640 x 1184 | 21379 | 17204 |
| 2 | 18 | 1280 x 704 | 8729 | 7992 |
| 3 | 62 | 480 x 1184 | 18607 | 17455 |
| 4 | 81 | 448 x 992 | 22356 | 18133 |

Evidence: `build_logs/e1pd4-template-request-update-20261010-01/verification.json`.

With the previous request templates, before enabling the optimization settings
above, the Encoder pixel-budget validation on 2026-10-10 passed 77 CPU cases and
all eight GPU preflight requests
(four serial, four concurrent) with HTTP 200 and finish_reason=stop.
The live Encoder process and parsed VitConfig both recorded 2500000/36864000.
A default-sizing control changed video 1 from 704x1280 to 640x1184, while all
four original requests retained 704x1280 because of their explicit dimensions.
Evidence is under `build_logs/e1pd4-encoder-pixels-preflight-20261010-01/`;
`verification.json` includes configuration, policy checks and cleanup evidence.

The manifest binds request IDs 1–4 to checked request bytes and video SHA256.
Each worker chooses uniformly among all four requests after every completion.
Templates retain model `qwen35_moe`, temperature 0, max_tokens 8192, stream false,
and content order. Request 2 and 3 prompts are copied from `request_02.json` and
`request_03.json` in
`/Users/taoyue/Documents/work/qwen3.5/rtp-llm-video-loadtest/混合视频240并发压测脚本/templates`.
Request 2 asks for visual-event JSON with `scenes`, `visual_events`, and
`on_screen_text`; request 3 asks for five Markdown sections describing visible
subjects and interactions. Neither asks for audio or speaker analysis. Request
1 and 4 prompts retain their original text. All four request preprocessing
objects are unified to FPS 6 and a frame range of 4–90 per the latest user
instruction. Only the runtime video URL becomes an absolute local path.
No thinking setting is injected. The four uploads are
identical to the videos in the supplied local dataset and original URL manifest.

Input expectations use the current Qwen35 renderer and shared sampling/grid
logic, not historical token counts. Sequence and both token-budget limits are
65536 for this preset. Other server settings and dependency hash checks match
the fixed-video C512 preset. This is a different workload and capacity, so its
throughput is not an equal-work comparison with the fixed-video reference.
The old preset's default 32768 limits and SSE requests remain unchanged.

## Validation and evidence

First send every request sequentially, then send the four requests concurrently
across the four PD ranks. Check nonempty content, finish_reason=stop, current
prompt/video token counts, pd_sep=false, reuse_len=0 and Encoder execution
logs before C512 starts. HTTP errors, truncation and timeouts fail the test.
The measured client stops issuing requests on failure and cleans up its own
services after outstanding requests drain or time out.

Results retain full nonstream response bodies, fixture IDs, source snapshots,
request/asset hashes, effective requests, current expected inputs, service
configurations, GPU observations and server logs. `steady-summary.json` adds
per-fixture window QPS, output-token throughput, E2E latency, counts and queue
inventory. Warmup and drain completions are separate. TTFT/TPOT are absent for
nonstream responses. Both windows must contain successful completions without
errors, and every fixture must succeed during measurement. There is no QPS
parity threshold; actual in-flight counts accompany configured concurrency.

## Validation on 2026-10-08

The new nine-case CPU target and both existing harness/preset CPU targets passed.
The first GPU launch exposed and fixed a renderer field-access mismatch before
starting services. The next run computed current prompt lengths 24599, 8977,
30102 and 40642 and brought up both services. Request 1 in the serial preflight
then failed after 84.89 seconds: DeepGEMM FP8 grid sync timeout in
`comm/barrier_fp8.cuh:115`, followed by HTTP 500 / 8205_CONNECTION_RESET_BY_PEER.
The four-concurrent preflight and C512 measurement did not run. All owned services
exited and selected GPUs returned to idle. No performance result is claimed.

Evidence, relative to the internal worktree root:
`build_logs/e1pd4-mixed-c512-20261008-1343/` (result, original response and backend
logs); CPU logs are in `build_logs/e1pd4-mixed-cpu-20261008/` and
`build_logs/e1pd4-mixed-cpu-v2-20261008/`. The failure report from that run retains
its original `model_initialization` phase label; the final harness now records
`mixed_video_preflight` before sending these requests. Kernel/JIT diagnosis is
outside this fixture/client addition; the pinned FP8 configuration was retained.

## Bounded synchronization diagnosis

Add `--preflight-only` to stop after the serial and four-concurrent correctness
checks, without warmup or performance windows. Add `--grid-sync-diag` to enable
DeepGEMM's per-CTA stage dump and bounded host metadata snapshots. Diagnostics
require `--preflight-only` and must not be used as performance measurements.
Use a fresh writable directory with `--dg-jit-cache` for diagnostic compilation.
Default performance behavior, request bodies and the frozen FP8 package stay
unchanged. The process log holds `DG_FP8_SYNC_DIAG` records if the device times
out; host launch metadata is under `smoke/mega-moe-snapshots`.

The October 8 grid-sync investigation identified mismatched collective order:
coordinated Prefill steps inserted fake Decode streams on idle PD ranks. Those
ranks replayed Decode Graph directly into MegaMoE, while the real Prefill rank
waited for the chunk-plan `all_reduce(MAX)` before entering the layer stack.
Diagnostic dumps show ranks 0/2/3 at dispatch site 2 with NVLink signal 3/4;
rank 1's Python stack stops at `MegaMoeChunker.max_tokens`, `count.item()`.
`NormalEngine` now creates a fake Prefill stream for a coordinated global Prefill
step. Decode steps retain their previous fake Decode behavior. Evidence and
verification live in `build_logs/e1pd4-grid-sync-debug-20261008/`.

Fix verification completed with diagnostics disabled: 69 CPU cases, the focused
engine regression, all four serial requests, and all four concurrent requests
passed. Every video returned HTTP 200 with `finish_reason=stop`, matching prompt
and video token counts, PD Fusion/cache checks and Encoder execution evidence.
The first video's serial response took 72.35 seconds and produced 2614 tokens.
All owned services were cleaned up and selected GPUs returned to idle. This
bounded debug run did not execute C512 performance windows.

## Chunk disabled C512 result on 2026-10-08

The full launcher was executed with `--mega-moe-chunk-tokens=0`, diagnostics
disabled and a fresh JIT cache. All 20 focused CPU cases and all eight GPU
preflight requests passed. Live process environments recorded chunk size 0;
MegaMoE initialization logged `max_tokens_per_rank=65536`.

C512 reached 512 actual requests in flight, but failed during warmup at 125.62
seconds. The client stopped issuing requests and drained the initial 512:
416 succeeded, 80 returned Encoder RPC `907_MM_REMOTE_RPC_FAILED / Deadline
Exceeded`, two failed RDMA reads with only 5–7 ms left in the request deadline,
and 14 video-3 responses reached 8192 tokens with `finish_reason=length`.
The last response completed at 433.95 seconds. No DeepGEMM grid-sync timeout
or CUDA out-of-memory was found in either service log. Both performance
windows are invalid because the load aborted before warmup ended.

All owned services exited and GPUs were idle. Results, complete responses,
service logs, environment evidence and source snapshots are under
`build_logs/e1pd4-mixed-nochunk-c512-20261008-1501/`; `verification.json`
separates the aborted attempt from valid steady-state performance.

## Omit the request output limit

Add `--omit-max-tokens --mega-moe-chunk-tokens=0` for the requested follow-up.
Original fixtures and hashes remain intact. Runtime requests remove only
`max_tokens`, in addition to replacing the video URL. No `max_completion_tokens`,
`extra_configs` or thinking override is injected. The backend currently defaults
to 32000 new tokens and still applies the 65536 sequence capacity; omission does
not mean unlimited generation. Length truncation remains a validation failure.
Encoder timeouts, warmup, windows and failure handling remain unchanged.

The follow-up at `build_logs/e1pd4-mixed-nochunk-nomax-c512-20261008-1529/`
passed 72 CPU cases and eight GPU preflight requests. All runtime payloads omit
`max_tokens` and `max_completion_tokens`, while chunking stays disabled. Both
300-second windows completed without errors: 261 successes / 0.870 QPS /
2747.77 output tokens per second, then 237 / 0.790 / 2527.94.

The complete run nevertheless failed during drain: of 1117 requests, 1069
succeeded, 43 hit the unchanged 1200-second client read timeout, and five ended
at the backend default 32000 tokens with `finish_reason=length`. Three valid
responses exceeded 8192 tokens (maximum 9541). Removing the request field does
not disable the backend default; this run is not an unlimited-output test.
No DeepGEMM grid-sync timeout or CUDA OOM was found in either service log.
The final response completed at 1916.24 seconds; owned services exited and the
selected GPUs returned to idle. The window metrics are observed results from
a run that failed overall acceptance. Full evidence is in `verification.json`.

## Lower-concurrency quality comparison

The mixed preset accepts `--steady-concurrency` levels 4, 8, 16, 32, 64, 128,
256 and 512 (default), including comma-separated sweeps such as
`--steady-concurrency=8,32,64,128,512`. Each level runs its own warmup, windows
and drain against the same service instance. Other capacity settings, including the per-rank service capacity of 256, four client processes,
240-second warmup and two 300-second windows, stay unchanged. The fixed-video
reference still permits only C512. Combine with `--omit-max-tokens` and
`--mega-moe-chunk-tokens=0` to compare with the previous mixed C512 run.
Inspect saved responses for repetition and analysis-section inflation; HTTP
200 plus `finish_reason=stop` alone is not a quality-equivalence check.


The C256 comparison on October 8 passed 23 focused CPU cases and all eight
GPU preflight requests. Preflight outputs were byte-identical to the preceding
C512 preflight; runtime request bodies were also identical. Actual in-flight
load reached 256 and stayed at 256 at both measurement-window boundaries.
The two windows completed 285 and 281 requests without protocol errors,
observing 0.950 / 0.937 QPS and 3131.25 / 2963.26 output tokens per second.
These are observed window metrics, not overall acceptance.

The complete 954-request run failed: 925 normally stopped responses, 17 client
read timeouts at 1200 seconds, and 12 responses truncated at the backend default
32000 tokens. Eight normally stopped responses exceeded 8192 tokens (maximum
13769). The longest contained approximately 13106 tokens before `</think>`
and only 661 afterwards. All 12 truncations lacked `</think>`; video-3 samples
repeat `bas` over 31000 times. Thus C256 still reproduces both long analysis
and repeated generation. Normal HTTP completion is not semantic acceptance.

All failures occurred in drain, which ended at 1548.05 seconds from load start.
No DeepGEMM grid-sync timeout or CUDA OOM appeared. Owned services exited and
GPUs 0/4/5/6/7 returned to idle. Complete responses, logs, source snapshots,
per-video/window statistics and quality analysis are under
`build_logs/e1pd4-mixed-nochunk-nomax-c256-20261008-1633/`.
The exact cause of concurrency-dependent output divergence remains unresolved.

## Decode accuracy and nonstream completion

During the historical validation runs, the mixed preset enabled
`RTP_TRTLLM_DECODE_BATCH_INVARIANT=1`. Each Decode attention request used the
same TRTLLM reduction geometry as a one-request call. Request concurrency,
MegaMoE batching, FP8 query/KV/weights, BF16 SSM
state, and normal Decode Graph remain enabled. This opt-in setting can cost
throughput through additional attention launches; measure that cost with the
saved windows. The fixed-video preset and the default production attention
path do not enable it. It guarantees the tested attention operator's batch
invariance, not determinism of every operator in the complete model.

An independent test with the actual 32 query heads / 2 KV heads / 256 head
dimension and a 30153-token context found up to 3.29% relative L2 variation
in the original FP8 Decode attention output as batch size changed. The
opt-in path matched the one-request output bit for bit for batches 1–128,
including CUDA Graph replay and poisoned unused value tails (38 comparisons).
The permanent regression additionally compares all rows of mixed-length
batches with independent serial calls.

BF16-query/FP8-KV Decode now clears unused values in each request's final
partial cache page. The native FP8 query conversion already did this; its
BF16 bypass did not. Recycled NaN values could otherwise contaminate valid
attention output through zero-probability multiplications. Only unused V
suffixes are cleared; active KV data and other pages are preserved, including
when lengths and page mappings change during Graph replay.

The nonstream renderer now decodes a complete, valid UTF-8 token span once.
Previously it decoded every increasing token prefix before joining the
results, creating quadratic CPU work. Partial contexts and malformed UTF-8
retain the existing fallback; streaming token handling is unchanged. With
the checkpoint tokenizer, the 8192-token replay took 8.74 seconds before and
0.00177 seconds after this change; the 32000-token complete decode took
0.00877 seconds. All four saved serial response texts matched byte for byte.
These numbers measure decoding only, not model generation or request latency.

Mixed-response validation checks each original prompt's required structure
and detects long repeated lines / repeated words. A response above 8192 tokens
is not rejected merely for its length. A normal long thinking section is
reported separately from looping generation; `finish_reason=length` remains
a failure. No sampling or thinking settings are changed by these fixes.

Validation before the full model sweep: 22 attention regression tests,
74 smoke/preset CPU tests and 30 frontend parsing tests passed. The full
C8/C32/C64/C128/C512 sweep is recorded under
`build_logs/e1pd4-output-degeneration-debug-20261008-2219/native-fp8-invariant-sweep/`.
Its final result is required before claiming end-to-end resolution.

## Follow-up result on 2026-10-09

The per-request Decode attention candidate is not an end-to-end resolution.
C8 completed all 154 requests successfully, including drain (maximum 4388
output tokens). C32 completed 316 HTTP-200/stop responses, but one video-1
response repeated the same four-line timestamp block 42 times and reached
8211 tokens. Semantic revalidation therefore yields 315 successes and one
failure. The old single-line repetition check missed this case; validation now
also rejects repeated multi-line blocks. All 75 smoke/preset CPU tests passed.

C32's measured windows observed 0.337/0.347 QPS and 1144.98/1196.58 output
tokens per second, but the run fails overall semantic acceptance. C64 was
interrupted after the late C32 failure was discovered; C128 and C512 were not
measured. Full evidence is in the `native-fp8-invariant-sweep` directory above,
including `semantic-revalidation.json` and the unmodified failure response.

Further operator isolation found exact valid-row agreement for MegaMoE
batches through 256, and for the tested FP8 dense projections, convolution
and GDN state updates. A BF16 dense projection differed in one element at
batch 128 (relative L2 6.70e-8); this does not establish the cause of the
remaining C32 failure. The first real-checkpoint vision diagnostic used an
incorrect resize-field mapping and is not validation of the online fixture
geometry. Its corrected rerun did not execute: precheck found the preceding
diagnostic's Bazel process still active. Work stopped under the user's
precheck-failure rule. The original diagnostic then exited and released its
GPU lock. No sampling settings or output cap were changed.

## Further diagnosis on 2026-10-09 (08:20)

The corrected actual-geometry vision diagnostic passed all 46 comparisons.
The grids were [23,44,80], [9,44,80], [31,44,80], and [41,44,80].
Large Prefill operator isolation passed 87 comparisons, and EP4 MegaMoE
Prefill isolation passed 44 comparisons with capacities through 65536 tokens.
These operator tests use synthetic inputs/weights where indicated in their
reports; they do not establish complete model or cache equivalence.

A real-model diagnostic sampled 11 token positions per request at all 60
decoder layers. All 56 comparisons across requests/ranks were bit-identical.
Serial and four-concurrent preflights passed, but C32 again failed during
warmup: 52 requests completed, 45 passed, and seven video-2 responses were flagged by
the then-current repeated-block validation (six at 5379 tokens, one at 5272). The launcher
stopped new requests, drained, and cleaned up. No measurement window was
valid; these diagnostic runs are not throughput results. Sparse Prefill
sampling does not rule out differences in unsampled positions or cache state.

A Decode Graph layer/step probe has passed its GPU unit test. Its full-model
run did not start: selected GPUs were occupied and the launcher idle precheck
failed before Bazel. Work stopped under the user's precheck-failure rule.
All owned services exited. Temporary diagnostic targets were archived and
removed; original dirty files were preserved. The remaining degeneration is
unresolved; C512 has not been validated. No request sampling or output cap
was changed. See continuation-results-20261009.json and the probe directories
under the diagnostic evidence path above.

## Validation correction and Decode probe repair (2026-10-09 09:10)

The seven flagged responses above were false positives in the combined
reasoning/final repetition count: each contained three JSON drafts before
`</think>` and the same final answer after it. Repetition checks now operate
on the two sections independently, keeping the same thresholds. All 76 CPU
smoke/preset tests passed, including a regression that failed before this
change and still rejects four repeated blocks within the reasoning section.
Revalidation passes all 52 saved diagnostic responses. The earlier C32
request 302 (8211 tokens, repeated timestamp blocks) remains a failure;
this validation correction is not a resolution of model degeneration.
The interrupted diagnostic still has no valid performance windows.

The Decode probe also incorrectly used block-table column zero as a request
identifier. LINEAR cache tables with prefix reuse disabled can leave that
column empty and materialize only the prompt tail. The revised probe uses
the original prompt's tail block, honors table width and row stride, and
keeps the frozen-snapshot behavior. CPU Triton interpretation reproduces the
old missed sample and passes 16 sparse-table/fixture cases with the fix.
A separate interpreter audit confirms final-state write/read locations for
four real prompt lengths and block-boundary cases. These checks do not replace
a GPU Graph regression or a real-model Decode comparison.

All selected GPUs remained occupied (roughly 254 GiB each at 09:08); this
continuation launched no GPU test or service. The v2 diagnostic code and
launchers are saved under the same evidence directory. GPU Graph validation
and the first-divergent-layer comparison remain pending. Original requests,
sampling, chunk=0, and Decode Graph=1 are unchanged.


### 2026-10-09 后续定位（未完成性能验收）

- 修正诊断探针真实 hidden width 4096 下首点 CAS 所有权竞争，B1/B8 GPU Graph 回归通过；旧快照比较排除第一个采样点。该修复仅影响诊断。
- C32 诊断完成 62/62 条请求，全部 HTTP 200、stop 且质量通过；视频 1–4 最长输出分别为 3284、5379、5729、3011 token。主动结束诊断，未完成两个窗口，不计性能验收通过。
- 首个有效差异为视频 3 Decode 第 22 步、第 5 层输出，及视频 4 第 46 步、第 16 层输出；对应残差一致。仍需比较实际归一化输入及 MoE 内部边界，不能据此判定根因。
- 真实权重的 240 次门控对照在 B≤64 一致；B≥128 有少量 router logits 舍入差异，共享专家门控一致。输入由保存残差重建，这不能替代模型内实际输入对照。
- 更细探针启动时 batch 128 Graph 捕获报 CUBLAS_STATUS_EXECUTION_FAILED，业务请求尚未开始，原因未定。重试时 GPU 空闲预检失败（8 卡均被占用），按约束停止；本次服务进程已退出。
- 证据位于仓库 `build_logs/e1pd4-output-degeneration-debug-20261008-2219/` 下 `decode-model-probe-v2/`、`actual-gates-and-probe-race/`、`decode-moe-probe-run/` 和 `decode-moe-probe-run-v2/`。下一轮使用已准备的细粒度探针，服务就绪后再开启后台 dump。


### 2026-10-09 共享专家门控候选修复

实际模型内部对照锁定首个差异：视频 3 第 22 个 Decode 步的第 5 层，及视频 4 第 46 步的第 16 层。两处 MoE 输入、router logits、路由专家输出与共享专家输出均逐位相同，只有共享专家标量门控不同，随后 MoE 输出分叉。

真实输入与权重的独立复现：视频 3 门控在 B1 为 -6.09375，B8 为 -6.0625；FP64 点积为 -6.0781270443949325，接近 BF16 舍入中点。视频 4 同样出现一个 BF16 ULP 的批次差异。固定每行 FP32 归约后，60 次真实输入对照全部批次一致，原路径其中 14 次不一致。合成 GPU 回归覆盖非连续输入、bias、空批次、CUDA Graph 重放及不支持形状的回退，全部通过；预设等 CPU 回归 76 项通过。

候选环境变量为 `RTP_BF16_SCALAR_LINEAR=1`，本混合预设启用；CUDA BF16 单输出线性投影、K≤8192 使用固定行归约，其余输入回退到原 F.linear，普通 Decode Graph 保持。环境变量在模块构造时读取。微基准 B128 从约 3.67us 降到 1.75us，B512 从 5.12us 降到 2.00us，B65536 从 82.53us 到 79.68us。

当前完整 C32/64/128/256/512 回归位于 `build_logs/e1pd4-output-degeneration-debug-20261008-2219/scalar-gate-invariant-sweep/`，保留 240 秒预热、两段 300 秒窗口、4 客户端进程，关闭 chunk，不设置请求 max_tokens。此处记录为候选，完整模型验收结论以该轮最终结果为准。根因证据及单算子验证位于 `decode-moe-probe-v2/` 和 `scalar-gate-validation/`。


共享门控候选验收更新：全阶段固定归约版本在视频 3 串行预检退化为 32000 token、连续重复并截断，已拒绝，未进入 C32。该结果保存在 `scalar-gate-invariant-sweep/`。后续版本使用 `RTP_BF16_SCALAR_LINEAR_MAX_ROWS=256`，仅覆盖当前小批次/Decode 范围，长输入 Prefill 保持原 F.linear 路径；新增超界回退回归通过。该版本运行目录为 `scalar-gate-decode-only-sweep/`，完整验收仍在进行。上面的 B65536 微基准仅是原型算子数据，当前服务超过 256 行时使用原路径。


16:00 验收更新：路由 fixed64 候选的 C256 仍未通过。869 条已发送，保存 868 条完成记录，其中 867 条成功、1 条视频 3（index=520）生成至 32000 token，出现重复并以 length 截断；另 1 条在途请求在失败清理时取消。虽然两个测量时段结束前未报错，排空失败使整档无效。C512 未启动。原始失败响应、进程日志与 stop-after-failure.json 已保存在该轮目录。此修复仅解决已证实的一处路由投影差异，不是最终解决方案；继续使用包含 GDN BF16 BA 输入/输出的逐层诊断。


### 2026-10-09 16:15 GDN BA 门控候选修复

保留共享门控和 router 修复后，真实模型的首差进一步定位到 GDN BF16 BA 投影：四视频分别在 step0/layer12、28、10、20（从0计数），真实 BA 输入一致，BA 输出首次偏离。四个 rank 均复现；最大绝对差分别为 0.00390625、7.62939453125e-6、1.52587890625e-5、0.0009765625。证据为 `decode-router-fixed-probe/comparison.json`。

实际输入/权重的 888 次对照覆盖 B1–B512、请求位置、Graph 重放及行/列布局；旧路径 162 次不同，固定64行分块后0次不同（`gdn-gate-validation/`）。新增 `RTP_BF16_GDN_LINEAR=1`，仅用于 CUDA BF16 K4096/N128、65≤M≤256，保持 M≤64 和大 Prefill 原路径。BF16 router 和 GDN 共用同一固定分块实现，但有独立开关和形状检查。

三种门控共8项 GPU 测试及76项 CPU 测试通过。B96 BA 微基准约4.82us→14.36us，B128约4.78us→11.86us。`decode-all-gates-fixed-probe-run/` 正在验证前三十二个 Decode 步、全部60层，确认模型内首差是否消除；这不是性能验收。完整 C256/C512 尚待后续无插桩回归，不能宣称故障解决。


16:25 模型内复查通过：`decode-all-gates-fixed-probe/comparison.json` 中四视频×四rank共16组 B96 对 B1 比较，各组18240个有效点（前32个Decode步、全部60层），全部逐位一致。诊断主动停止后的 exit=3 是有记录的清理中断，不是性能验收结果。临时诊断target已移除。无插桩的完整C256/C512已在 `all-gates-fixed-c256-c512/` 开始；仅最终含排空的结果可用于验收。


### 2026-10-09 17:10 最终 C256/C512 验收通过

本轮包含共享专家标量门控、BF16 router 和 GDN BA 三处批次相关舍入修复；保留前序每请求 FP8 Decode attention 修复。混合预设启用三项 opt-in 开关，普通模块默认关闭。标量修复仅用于 M≤256；router/GDN 固定64行分块仅用于 65≤M≤256 及指定形状。大 Prefill 保持原路径。

无插桩完整回归已结束，runner exit=0、result=PASSED。Encoder GPU0、PD Fusion GPU4–7；FP8、N7、chunk=0、Decode Graph=1；4客户端、每档240秒预热、2×300秒测量、等待全部在途完成。请求不设置 max_tokens，未改采样或 thinking；后端默认32000及上下文限制仍存在，全部响应均为 stop。

| 并发 | 全阶段成功/总数 | 测量成功 | QPS | 输出 token/s | E2E P50/P95 秒 | 预热/排空完成数 |
|---|---:|---:|---:|---:|---:|---:|
| C256 | 901/901 | 560 | 0.9333 | 2693.23 | 290.68/393.83 | 85/256 |
| C512 | 1142/1142 | 537 | 0.8950 | 2614.11 | 403.67/734.43 | 93/512 |

两档均0错误、两段窗口均有效、四条请求均有测量成功样本。全部2043条压力请求都与同轮串行基线完整文本逐字一致，四视频输出固定为3121/1714/3854/2857 token；输入token为24599/8977/30102/40642。测量期实际平均在途分别约256/512，未以低实际并发冒充高并发。C512吞吐较C256略低、尾延迟明显增大，本轮说明输出退化未复现，不代表排队开销消失。

三种门控共8项GPU回归和76项CPU回归通过。真实模型32个Decode步、60层、四视频×四rank共16组内部对照均逐位一致。最终运行四rank均有 Prefill/Decode/Graph capture/replay、MegaMoE、原生FP8注意力证据，fallback为0；Encoder保留2051条embedding记录。响应reuse字段为0，复用缓存及MM结果缓存关闭，URL哈希元数据缓存沿用原配置。

所有本轮服务已退出，GPU计算进程清空；46个被测源码文件与当前实现哈希一致，原有无关脏文件保持。完整响应、日志、请求/媒体哈希、实际命令和源码快照保留在 `build_logs/e1pd4-output-degeneration-debug-20261008-2219/all-gates-fixed-c256-c512/`；逐视频、预热/测量/排空及实际在途统计见 `final-aggregate.json`，清理和执行证据见 `final-execution-cleanup.json`。非流式不报告TTFT。

## 2026-10-09 18:49 single-kernel gate optimization

`RTP_BF16_GATE_KERNEL=1` is opt-in globally and defaults on in this mixed preset.
The unified-config follow-up makes this the only switch for all three gates. During an explicit Decode
scope, Router K4096/N512 and GDN BA K4096/N128 use one Triton kernel for every M.
The scalar gate also uses its existing fixed-row kernel for every Decode M.
Unknown phase and Prefill retain their previous paths; large Prefill is unchanged.
Normal Decode Graph and FP8 settings remain enabled.

The geometry is independent of M, with no M specialization/autotuning. Each CTA
computes all K partitions locally with FP32 accumulation: two for Router and
four for GDN. GDN partial sums are added as `((p0+p1)+p2)+p3`, with floating-point
fusion disabled. A balanced tree changes rare BF16 midpoint rounding cases;
a permanent regression covers this distinction. There are no atomics, padding
buffers, intermediate GEMM outputs or result-copy kernels.

Thirteen GPU unit cases and 76 CPU cases pass. GPU coverage includes M1–1024,
non-multiple tile tails, strided inputs, row/column weight layouts, bias,
unsupported/empty fallback, live CUDA Graph replay, and phase/flag boundaries.
176 real-input comparisons show zero fixed64-reference or batch differences.
Across four videos and all 60 layers, 420 groups / 13440 sampled input rows
match the recorded runtime gate outputs exactly. This is tested equivalence,
not a claim that every possible input matches cuBLAS rounding.

Pure kernel timings below use actual model weights/input, 30 warmups and 100
profiled CUDA Graph replays on the same GPU. Timings sum CUDA kernel durations
per invocation; CPU launch overhead and memcpy/memset time are excluded and
recorded separately. Input preparation and compilation are excluded; no
`torch.compile` is used. Every new path has exactly one `_bf16_gate_gemm` event
and zero copy/memset events. Graph replay latency is saved as a separate metric.

| Projection | M | Native us | fixed64 us | Single kernel us | Reduction vs fixed64 |
|---|---:|---:|---:|---:|---:|
| router | 1 | 7.178 | 7.176 | 8.972 | -25.0% |
| router | 64 | 6.874 | 6.873 | 9.033 | -31.4% |
| router | 96 | 6.678 | 15.414 | 9.130 | 40.8% |
| router | 128 | 6.643 | 12.789 | 9.184 | 28.2% |
| router | 256 | 7.580 | 26.231 | 13.152 | 49.9% |
| router | 512 | 6.914 | 52.582 | 22.052 | 58.1% |
| router | 1024 | 7.390 | 106.295 | 40.806 | 61.6% |
| gdn | 1 | 5.195 | 5.194 | 6.460 | -24.4% |
| gdn | 64 | 5.421 | 5.426 | 6.572 | -21.1% |
| gdn | 96 | 6.287 | 11.609 | 6.604 | 43.1% |
| gdn | 128 | 6.205 | 9.922 | 6.621 | 33.3% |
| gdn | 256 | 6.967 | 20.770 | 6.731 | 67.6% |
| gdn | 512 | 6.715 | 41.065 | 12.651 | 69.2% |
| gdn | 1024 | 7.769 | 82.566 | 24.129 | 70.8% |

For M<=64, fixed64 is the original native call: the new path costs about
20–32% more to keep a fixed reduction. For M>256, fixed64 is an extrapolated
reference workaround, not the old service path (which fell back to native).
The new path is slower than native at several sizes, especially large Router M;
do not interpret the fixed64 comparison as an across-the-board native speedup.
The earlier 30-replay M128 probe is retained separately; this table uses the
broader 100-replay measurement.

C512 acceptance: Encoder GPU0 / PD Fusion GPUs4–7, N7, chunk0, Decode Graph1,
4 clients, 240s warmup, two 300s windows, full drain, no diagnostic hook.
Only runtime video paths and omission of request max_tokens differ from the
original templates. Sampling/thinking remain intact. The backend default32000
and sequence capacity65536 still apply; every response stops normally.

| Phase | Success / errors | QPS | Output token/s | E2E P50/P95 s | Average in flight |
|---|---:|---:|---:|---:|---:|
| warmup | 96 / 0 | 0.4000 | 828.48 | 166.74/236.68 | 511.98 |
| measurement | 545 / 0 | 0.9083 | 2604.41 | 429.91/696.63 | 512.00 |
| drain | 512 / 0 | 1.1299 | 3524.57 | 613.84/732.22 | 305.55 |

Both windows are valid: 298 / 247 successful completions, 0 errors, four videos
represented in each, output throughput 2757.73 / 2451.10 token/s. All 1153
pressure responses and eight preflights match the old fixed64 serial full text.
Output lengths remain 3121/1714/3854/2857; input tokens24599/8977/30102/40642.
The measurement average is2604.42 token/s versus2614.11 in the preceding
fixed64 C512 run (-0.37%); this run establishes no end-to-end throughput gain.
Total workload time1293.14s includes453.14s drain. Nonstream TTFT is not reported.

| Video | Total successful | Measured successful | Measured QPS | Output token/s | E2E P50/P95 s | Average in flight |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 310 | 145 | 0.2417 | 754.24 | 535.58/683.27 | 155.43 |
| 2 | 269 | 142 | 0.2367 | 405.65 | 327.12/473.87 | 66.25 |
| 3 | 291 | 130 | 0.2167 | 835.03 | 450.10/747.01 | 161.02 |
| 4 | 283 | 128 | 0.2133 | 609.49 | 538.04/655.33 | 129.30 |

All four ranks retain Prefill/Decode, Graph capture/replay, MegaMoE and native
FP8 execution evidence; fallback count0. Encoder has1161 successful embedding
records. Reuse fields are zero; reuse/MM result caches remain disabled while
URL hash metadata cache retains the prior configuration. All eight original
request/media hashes and all four exact runtime request transformations pass.
Owned services and temporary benchmark targets are removed; GPUs are released
and the six unrelated dirty files retain their original hashes.

Evidence root: `build_logs/e1pd4-output-degeneration-debug-20261008-2219/gate-kernel-optimization/`.
`c512-v2/` retains complete responses/logs, actual configurations and source
snapshots; `final-aggregate.json` contains every video's warmup/measurement/drain
counts, latency and in-flight statistics. `final-execution-cleanup.json`,
`fixture-sha256-verified.json`, `request-fidelity-verified.json` and backend
environment evidence record acceptance checks. `range-benchmark/` contains
78 timing rows and raw traces; `accepted-tests-final/` and `final-regression-v3/`
retain correctness results. The earlier `c512/` prototype is rejected: a balanced
GDN reduction changed video1's serial text; it never reached C512 measurement.

All 49 snapshotted source files were audited before this documentation update.
Only test/BUILD's trailing blank lines differed; normalized contents were equal
and `build-format-only-cleanup.json` records that cleanup. All seven optimized
code files match their accepted hashes after temporary benchmark removal.
The documentation update is recorded separately; original run snapshots are
never rewritten to hide post-run changes.

## 2026-10-09 single environment variable follow-up

Merged the four switches and removed the scalar row-limit environment override.
`RTP_BF16_GATE_KERNEL` alone enables/disables all three BF16 gate repairs.
The kernel math and validated phase boundaries are unchanged. Explicit0 in the
mixed launcher's environment is forwarded to Bazel, worker metadata and services;
an unset launcher environment keeps the correctness preset's default1.

All four GPU targets passed (14 unit cases), including the unified switch,
removed-option isolation, constructor snapshot, phase boundaries and Graph
regressions. All77 CPU cases passed, including explicit0/1 launcher-to-worker
forwarding and asserting that only one BF16 variable is emitted. The full C512
measurement above predates this configuration refactor and was not rerun.
Evidence: `build_logs/e1pd4-output-degeneration-debug-20261008-2219/gate-unified-env/`.
