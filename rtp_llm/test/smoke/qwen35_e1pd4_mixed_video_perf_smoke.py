"""Four original video requests, E1PD4 N7 configurable concurrency; dry-run unless --execute."""

import os
import sys
from pathlib import Path

import qwen35_e1pd4_perf_smoke as reference
import qwen35_epd_fusion_4gpu_smoke as smoke

TARGET = "//rtp_llm/test/smoke:qwen35_e1pd4_mixed_video_perf_smoke"
QWEN35_OPTIMIZATION_ENV = {
    "RTP_QWEN35_GDN_PREFILL_BACKEND": "flashinfer",
    "RTP_QWEN35_FUSED_CONV_QKV_NORM": "1",
    "RTP_QWEN35_FUSED_GATED_RMSNORM_FP8": "1",
    "RTP_QWEN35_GDN_DECODE_BACKEND": "flashinfer",
    "RTP_QWEN35_DECODE_FUSION": "1",
}
SNAPSHOT_FILES = reference.SNAPSHOT_FILES + (
    "rtp_llm/models_py/triton_kernels/common/bf16_gate_linear.py",
    "rtp_llm/models_py/triton_kernels/qwen35_decode_fusion/env.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/bf16_gate_linear_test.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/f16_linear.py",
    "rtp_llm/models_py/triton_kernels/common/scalar_linear.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/router_linear.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/bf16_router_linear_test.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/bf16_gdn_linear_test.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/bf16_scalar_linear_test.py",
    "rtp_llm/models_py/modules/factory/linear/impl/cuda/test/BUILD",
    "rtp_llm/openai/renderers/reasoning_tool_base_renderer.py",
    "rtp_llm/test/frontend_streaming_utf8_replay_test.py",
    "rtp_llm/models_py/triton_kernels/common/fused_fp8_qkv_cache.py",
    "rtp_llm/models_py/modules/factory/attention/cuda_impl/test/native_fp8_mha_test.py",
    "rtp_llm/openai/openai_endpoint.py",
    "rtp_llm/config/generate_config.py",
    "rtp_llm/cpp/engine_base/stream/GenerateStream.cc",
    "rtp_llm/cpp/normal_engine/NormalEngine.cc",
    "rtp_llm/cpp/normal_engine/NormalEngine.h",
    "rtp_llm/cpp/normal_engine/test/EngineTest.cc",
    "rtp_llm/test/smoke/qwen35_e1pd4_mixed_video_perf_smoke.py",
    "rtp_llm/test/smoke/qwen35_e1pd4_mixed_video_perf_smoke.bzl",
    "rtp_llm/test/smoke/qwen35_mixed_video.py",
    "rtp_llm/test/smoke/qwen35_mixed_video_test.py",
)


def add_arguments(parser):
    reference.add_arguments(parser)
    parser.add_argument(
        "--grid-sync-diag",
        action="store_true",
        help="enable DeepGEMM stage diagnostics; requires --preflight-only",
    )
    parser.add_argument(
        "--mega-moe-chunk-tokens",
        type=int,
        choices=(0, 8192),
        default=0,
        help="MegaMoE chunk size; 0 disables chunking (default), 8192 enables it",
    )
    parser.add_argument(
        "--omit-max-tokens",
        action="store_true",
        help="omit max_tokens in runtime requests; backend defaults and context limits still apply",
    )
    parser.description = __doc__
    parser.set_defaults(
        mixed_video=True,
        native_fp8_attn=1,
        max_input_tokens=65536,
        data_dir=str(Path(__file__).resolve().parent / "qwen35_mixed_video_data"),
    )


def configure(a, parser):
    try:
        levels = smoke.positive_sizes(a.steady_concurrency)
    except ValueError:
        parser.error("invalid mixed-video concurrency list")
    if not levels or any(n not in (4, 8, 16, 32, 64, 128, 256, 512) for n in levels):
        parser.error("mixed-video concurrency must contain only 4,8,16,32,64,128,256,512")
    reference.configure(
        a,
        parser,
        allowed_concurrency=(a.steady_concurrency,),
    )
    if not a.mixed_video or a.max_input_tokens != 65536:
        parser.error(
            "the mixed video preset fixes mixed-video and max-input-tokens=65536"
        )
    a.bazel_option.extend(
        f"--test_env={key}={value}"
        for key, value in QWEN35_OPTIMIZATION_ENV.items()
    )
    a.preset_metadata["environment"].update(QWEN35_OPTIMIZATION_ENV)
    # This correctness preset defaults on; an explicit environment value wins.
    gate_kernel = os.environ.get("RTP_BF16_GATE_KERNEL", "1")
    a.bazel_option.append("--test_env=RTP_BF16_GATE_KERNEL=" + gate_kernel)
    a.preset_metadata["environment"]["RTP_BF16_GATE_KERNEL"] = gate_kernel
    chunk_tokens = str(a.mega_moe_chunk_tokens)
    a.bazel_option.append("--test_env=RTP_MEGAMOE_CHUNK_TOKENS=" + chunk_tokens)
    a.extra_test_args.append("--mega-moe-chunk-tokens=" + chunk_tokens)
    a.preset_metadata["environment"]["RTP_MEGAMOE_CHUNK_TOKENS"] = chunk_tokens
    if a.omit_max_tokens:
        a.extra_test_args.append("--omit-max-tokens")
    if a.grid_sync_diag:
        if not a.preflight_only:
            parser.error(
                "--grid-sync-diag requires --preflight-only; diagnostic runs are not performance measurements"
            )
        a.bazel_option = [
            option.replace(
                "DG_MEGA_MOE_FP8_GRID_SYNC_DIAG=0", "DG_MEGA_MOE_FP8_GRID_SYNC_DIAG=1"
            ).replace("MEGA_MOE_LOG_INPUTS=0", "MEGA_MOE_LOG_INPUTS=1")
            for option in a.bazel_option
        ]
        snapshot_dir = str(Path(a.output).resolve() / "mega-moe-snapshots")
        a.bazel_option.append("--test_env=MEGA_MOE_SNAPSHOT_DIR=" + snapshot_dir)
        a.extra_test_args.append("--grid-sync-diag")
        a.preset_metadata["environment"].update(
            DG_MEGA_MOE_FP8_GRID_SYNC_DIAG="1",
            MEGA_MOE_LOG_INPUTS="1",
            MEGA_MOE_SNAPSHOT_DIR=snapshot_dir,
        )
    a.bazel_target = TARGET
    a.snapshot_files = SNAPSHOT_FILES
    a.preset_metadata.update(
        workload="four-original-video-requests",
        attention_precision="FP8 query/KV",
        stream=False,
        max_tokens=None if a.omit_max_tokens else 8192,
        request_transform=(
            "absolute_video_path_and_omit_max_tokens"
            if a.omit_max_tokens
            else "absolute_video_path"
        ),
        max_input_tokens=65536,
        selection="uniform-per-request",
        comparison_note="Different workload and input capacity from the fixed-video reference",
    )


validate_runtime = reference.validate_runtime


def main(argv=None):
    return smoke.main(argv, preset=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
