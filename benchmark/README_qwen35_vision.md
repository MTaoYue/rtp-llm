# Qwen3.5 vision encoder CUDA optimization

The runner loads only the 333 real BF16 visual tensors from Qwen3.5-397B-A17B-FP8. It strictly validates names, shapes, dtypes and completeness. The language model is never instantiated. The frozen reference is read from git commit ee510269110af5ac3cd04c22e6dfc0460a27b2f8.

## Runtime API

Existing forward(hidden_states, grid_thw) and weight format are unchanged. CUDA inference defaults to optimized. Use RTP_QWEN35_VISION_MODE=baseline to disable, or encoder.configure_vision("optimized", fused_rope=True, fused_pos=True, cache_metadata=True). Modes: baseline, optimized, compile, cudagraph, compile_cudagraph. Explicit instance configuration takes precedence over config.vision_execution_mode and environment. Compile and graph modes require explicit activation.

Training, enabled autograd, non-CUDA inputs, extra forward kwargs, and existing internal PPU attention/RoPE patches use the reference path. Missing CUDA dependencies fall back with a warning. Compile/capture failures are logged and counted in vision_stats(). The benchmark asserts requested compile/graph modes actually ran.

Grid metadata is shared across blocks. Position interpolation preserves BF16 multiplication/addition rounding; out-of-place RoPE preserves FP32 intermediates and supports strided QKV views. Inductor uses native LayerNorm/GELU boundaries for reference numerics, with implicit CUDA Graphs disabled.

Graphs match the full ordered grid, input shape/stride, dtype, device and attention backend. graph_capacity defaults to 8; excess shapes run without capture. Metadata uses a 32-entry LRU. Replay copies into private input buffers and clones both outputs. A per-instance lock and CUDA event serialize cross-stream buffer reuse. Weight loading and device/dtype conversion clear caches; ordinary parameter mutations invalidate via version counters. After directly modifying inference-created parameters (no version counters), call clear_vision_cache(). Also clear after manually replacing submodules or changing configuration.

## Reproduction

Run from the workspace directory containing this checkout as `github-opensource`. Check nvidia-smi first. Results use GPU 0, NVIDIA L20D, BF16, SDPA, torch 2.11.0+cu130, Triton 3.6.0, transformers 5.16.1. CPU threads and seed are fixed in the runner.

~~~bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=8 python3 github-opensource/benchmark/run_qwen35_vision_suite.py --checkpoint ../model/Qwen3.5-397B-A17B-FP8 --output artifacts/reproduce --inputs artifacts/real-inputs
CUDA_VISIBLE_DEVICES=0 python3 github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/test/vision_cuda_test.py
~~~

Defaults: 20 warmups, 100 samples, five rounds with alternating mode order and one process per case. Each case retains a runtime per mode across rounds. Wall time includes metadata, graph input copy and independent outputs; CUDA Events are also recorded. Both encoders use identical weights and SDPA. Inputs, reference source, environment, raw JSON/CSV, first-call time, cache counters and per-sample times are saved. Peak memory is the entire comparison process, including the reference and retained runtimes, not isolated serving memory. First-call time may reuse disk compiler caches; use --fresh-cache and a NEW output directory for a separate cold experiment.

## Real inputs, ablations and profiles

~~~bash
PYTHONPATH=/tmp/qwen35-vision-tools python3 github-opensource/benchmark/prepare_qwen35_vision_inputs.py --dataset ../dataset --checkpoint ../model/Qwen3.5-397B-A17B-FP8 --output artifacts/real-inputs-new
CUDA_VISIBLE_DEVICES=0 python3 github-opensource/benchmark/qwen35_vision.py --output artifacts/ablation-new --cases image_448 --modes baseline,optimized --no-rope --no-pos
CUDA_VISIBLE_DEVICES=0 python3 github-opensource/benchmark/qwen35_vision.py --output artifacts/profile-new --cases image_448 --rounds 1 --iterations 5 --profile
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=8 python3 github-opensource/benchmark/qwen35_vision.py --output artifacts/cold-new --cases image_448 --rounds 1 --iterations 5 --fresh-cache
~~~

PyAV is needed only for preparing local video frames; no remote downloads are made. The manifest records decode/processor time and actual processed grids outside encoder timing. Toggle --no-rope, --no-pos and --no-cache separately or together. Mode comparisons isolate compile and explicit graph contributions. Profiling exports Chrome traces and operator tables separately from formal timing.

Both outputs must be finite, equal shape/dtype, relative L2 <= 0.01, cosine >= 0.999 and max absolute error <= 0.05 * max(1, max(abs(reference))). Regressions also cover intermediate kernels, strided QKV, alias safety, changed content, equal-token different grids, shape switching, graph capacity, cross-stream replay, weight/device invalidation, CPU/PPU fallback and injected failures.

The parent VISION_ENCODER_REPORT.md and artifacts/summary.{json,csv} summarize 18 cases, final-code rechecks and ten fusion/cache ablations. Original trials and profiles remain under artifacts; full/ and recheck/ hold formal results. These are standalone Python/kernel tests; no Bazel invocation is required.

Real-weight state/capacity checks (nine ordered grid keys, eight captured graphs, changed-content replay and retained output validation):

~~~bash
CUDA_VISIBLE_DEVICES=0 python3 github-opensource/benchmark/validate_qwen35_vision_state.py --checkpoint ../model/Qwen3.5-397B-A17B-FP8 --output artifacts/real-state-new
~~~

## Custom batch size

The dedicated entry point reuses all standard correctness and timing checks without changing the original 18-case matrix. It defaults to batch 128 and resolution 448; resolution must be divisible by 32.

~~~bash
CUDA_VISIBLE_DEVICES=0 python3 github-opensource/benchmark/qwen35_vision_batch.py --batch-size 128 --resolution 448 --modes baseline,optimized,cudagraph --output artifacts/batch128_448_reproduce
~~~

The parent BATCH128_VISION_REPORT.md reports five alternating rounds and compares CUDA Graph directly with optimized eager execution.

## Eager fusion follow-up (batch 128)

The CUDA inference `optimized` path includes two independently switchable additions:

- `fused_patch=True`: fuse the Conv3d output bias and positional addition in Triton. The convolution remains cuDNN; round the bias result to the activation dtype before adding positions. Compilation retains its automatic equivalent fusion.
- `batch_attention=True`: for SDPA, batch independent image/frame segments when **all segment lengths are equal**. Each image/frame has its own attention batch dimension. Unequal lengths and other attention backends retain the segmented path. Position encoding and graph keys still use complete ordered grids, so equal token counts do not imply equal image geometry.

These options do not invoke `torch.compile` or CUDA Graph in `optimized` mode. LayerNorm and GELU retain their native rounding boundaries. CUDA Graph and compile remain explicit modes.

The benchmark accepts `--no-patch-fusion` and `--no-batch-attention`. Benchmark-only mode aliases permit paired alternating ablations:

| Alias | Patch bias + position | Batched SDPA |
|---|---|---|
| eager_before | off | off |
| eager_patch | on | off |
| eager_batch | off | on |
| optimized | on | on |

Run from the parent worktree:

```bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=8 python3 github-opensource/benchmark/qwen35_vision_batch.py --batch-size 128 --resolution 448 --modes eager_before,eager_patch,eager_batch,optimized,compile,cudagraph --output artifacts/b128-eager-fusions-reproduce --profile
CUDA_VISIBLE_DEVICES=0 python3 github-opensource/benchmark/qwen35_vision.py --modes eager_before,optimized --inputs artifacts/real-inputs --output artifacts/fusion-matrix-reproduce
```

Defaults remain 20 warmups, 100 measurements, and 5 alternating rounds. The batch-128 case is 128 independent 448×448 images (100,352 patch tokens); this does not represent a batch of 128 varying resolutions.

The completed paired measurements are summarized in the parent
BATCH128_FUSION_REPORT.md (with JSON/CSV). Regenerate that report from the
preserved original result directories:

~~~bash
python3 github-opensource/benchmark/summarize_qwen35_fusions.py --artifacts artifacts --output BATCH128_FUSION_REPORT.md
~~~

To reproduce the pre-migration eager/compile comparison with current source,
disable both new options (the pre-migration source and full patch are also saved
under artifacts/pre-eager-fusion):

~~~bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=8 TORCH_COMPILE_DEBUG=1 python3 github-opensource/benchmark/qwen35_vision_batch.py --batch-size 128 --resolution 448 --modes optimized,compile --no-patch-fusion --no-batch-attention --output artifacts/b128-before-reproduce --profile
~~~

Compiler disk cache and debug settings affect cold compilation time; the saved
environment.json and debug output record the original run. Formal latency excludes
compilation and profiling. Peak memory includes both models in the comparison
process and is not a standalone deployment memory estimate.

## 128 complete mixed-resolution videos

The batch unit is 128 complete clips, sampled at 6 fps for 3–8 seconds. All 4200 frames (2100 temporal groups, 4,352,772 patch tokens) are submitted in one encoder invocation. Temporal groups are independent attention sequences. There is no encoder microbatching.

The fixture derives 128 distinct center-cropped/resized clips from three local source videos; it is not 128 independent original videos. It has 64 landscape and 64 portrait clips across eight resolutions. Source SHA256, crop, timestamps, sampled frame indices, final ordered grid and processed-input SHA256 are saved. Decoding measures three unique sources once and must not be interpreted as the decoding cost of 128 independent files.

Run these commands from the parent worktree directory. Inputs and raw results remain outside the source checkout.

~~~bash
PYTHONPATH=/tmp/qwen35-vision-tools python3 github-opensource/benchmark/prepare_qwen35_mixed_videos.py --dataset ../dataset --checkpoint ../model/Qwen3.5-397B-A17B-FP8 --output artifacts/mixed-video128 --batch-unit videos --batch-size 128 --derive --seed 20260911
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=8 python3 github-opensource/benchmark/qwen35_mixed_video.py --checkpoint ../model/Qwen3.5-397B-A17B-FP8 --input artifacts/mixed-video128/inputs/mixed128.pt --manifest artifacts/mixed-video128/manifest.json --reference-dir artifacts/mixed-video128-safe-reference --output artifacts/mixed-video128-formal --modes baseline,optimized,compile,compile_cudagraph --safe-native-norm --warmup 5 --iterations 20 --rounds 3
python3 github-opensource/benchmark/summarize_qwen35_mixed_video.py --input artifacts/mixed-video128-formal --output artifacts/mixed-video128-summary
~~~

Use a different reference directory and omit --safe-native-norm to preserve a completely pristine reference. The installed PyTorch native CUDA LayerNorm was observed to corrupt full tensors above the 2^32 flattened-element boundary. The safe reference changes only native LayerNorm row partitioning (1,000,000 rows per call); the optimized implementation independently sizes partitions below 2^31 elements. Neither splits the encoder batch. Reference metadata prevents silently mixing these policies. Correctness thresholds remain relative L2 <= 1%, cosine >= 0.999 and max absolute error <= 0.05 * max(1, reference maximum magnitude), checked separately for every video's hidden and merger outputs with FP64 metrics.

Cross-batch-size comparisons are additional diagnostics. Full-tensor captures traced the approximately 1.66% final-output discrepancy to a BF16 cuDNN patch Conv3d kernel switch at 2**22 patches on the tested L20D / PyTorch 2.11.0+cu130 environment, separately from the oversized LayerNorm issue. The earlier four-row sample missed the sparse patch-output differences. A diagnostic wrapper limiting each patch convolution to 1,000,000 patches made all 128 videos / 256 outputs bitwise equal to individual runs, while all blocks and the merger still processed the full batch. This Conv3d workaround is not yet integrated into the production encoder; its threshold is environment-specific and requires revalidation elsewhere. Fixed-batch performance results predate this diagnostic workaround. See the companion CROSS_BATCH_DIAGNOSIS.md report and diagnostic JSON for reproduction details.

For compiled inference with SDPA and more than 128 variable-length sequences, segmented native attention is an opaque custom operation. It retains the original per-sequence SDPA, transpose and concatenation behavior while preventing thousands of attention calls from being unrolled into the compiler graph. The remaining tensor core is compiled with implicit CUDA Graph disabled. This boundary changes neither the eager attention path nor small compiled cases.

Compile and CUDA Graph modes remain opt-in. Graph keys include the complete ordered grid, dtype, device and attention backend. Independent output cloning protects prior outputs. Allocation failure releases the cached graph and falls back to the uncaptured core; successful hits, captures and failures must be reported before attributing latency to a graph. In this fixture eager CUDA Graph hit a replay-output OOM, whereas compiled CUDA Graph passed the preflight. A zero-hit fallback is not a CUDA Graph performance result.

The summarizer rejects incomplete formal runs and failed correctness/compilation/graph checks. It emits pooled and per-round p50, p95, raw wall/CUDA-event timings, video/frame throughput, cold starts and peak allocated/reserved memory. Use --allow-partial only for progress inspection.

Large-address kernel regressions are opt-in and require at least 40 GiB free GPU memory:

~~~bash
CUDA_VISIBLE_DEVICES=0 RUN_VISION_LARGE_TESTS=1 python3 github-opensource/rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/test/vision_large_address_test.py
~~~
Interrupted complete-video runs can continue with the identical command plus --resume. Completed mode/round records are retained; arguments and GPU/dependency/input/reference policy metadata must match. Resume environment files preserve the new process loading details separately. Run long jobs with nohup and redirected stdin/stdout when using a browser terminal.

Reconfiguring execution mode or calling clear_vision_cache() explicitly releases
owned graph buffers after waiting for the runtime completion event. This also
works if Dynamo or a caller retains the old runtime. The 128-video regression
captures and replays two freshly constructed compiled graphs in one process;
see artifacts/mixed-video128-lifecycle-preflight/.
