"""Standalone real-weight Qwen3.5 vision correctness/performance benchmark."""

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "rtp_llm/multimodal/multimodal_mixins/qwen3_5_moe/qwen3_5_moe_vit.py"


def load_module(source=SOURCE):
    spec = importlib.util.spec_from_file_location(
        "qwen35_" + source.stem, source, submodule_search_locations=[str(SOURCE.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(checkpoint, source=SOURCE):
    start = time.perf_counter()
    module = load_module(source)
    config = module.Qwen3_5MoeVisionConfig(
        **json.loads((checkpoint / "config.json").read_text())["vision_config"]
    )
    config._attn_implementation = module.default_attn_impl
    model = module.Qwen3_5MoeVisionModel(config).eval().to(dtype=torch.bfloat16)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "model.visual."
    visual = {k: v for k, v in index.items() if k.startswith(prefix)}
    assert visual, "No vision weights in checkpoint"
    state = {}
    dtypes = Counter()
    for shard in sorted(set(visual.values())):
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as f:
            for key, value in visual.items():
                if value == shard:
                    tensor = f.get_tensor(key)
                    dtypes[str(tensor.dtype)] += 1
                    state[key[len(prefix) :]] = tensor
    assert all(t.dtype == torch.bfloat16 for t in state.values()), dtypes
    model.load_state_dict(state, strict=True)
    model.cuda()
    torch.cuda.synchronize()
    return model, {
        "load_seconds": time.perf_counter() - start,
        "weight_dtypes": dict(dtypes),
        "weight_count": len(state),
        "backend": module.default_attn_impl,
        "source": str(source),
    }


def cases():
    result = [(f"image_{s}", [(1, s // 16, s // 16)]) for s in (224, 448, 896, 1344)]
    result += [("wide", [(1, 28, 56)]), ("tall", [(1, 56, 28)])]
    result += [(f"batch_{b}", [(1, 28, 28)] * b) for b in (2, 4, 8)]
    result += [
        ("mixed", [(1, 14, 14), (1, 28, 56), (1, 56, 28)]),
        ("video_t2", [(2, 28, 28)]),
        ("video_t4", [(4, 28, 28)]),
    ]
    return result


def compare(actual, reference):
    result = {}
    for name in ("last_hidden_state", "pooler_output"):
        a, r = getattr(actual, name), getattr(reference, name)
        assert a.shape == r.shape and a.dtype == r.dtype
        a, r = a.float(), r.float()
        delta = a - r
        rel = (delta.norm() / r.norm().clamp_min(1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(), r.flatten(), dim=0
        ).item()
        maximum = delta.abs().max().item()
        limit = 0.05 * max(1.0, r.abs().max().item())
        passed = (
            bool(torch.isfinite(a).all())
            and rel <= 0.01
            and cos >= 0.999
            and maximum <= limit
        )
        result[name] = dict(
            relative_l2=rel,
            cosine=cos,
            max_abs=maximum,
            max_abs_limit=limit,
            passed=passed,
        )
    return result


def measure(fn, warmup, iterations):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    cold = (time.perf_counter() - start) * 1000
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(iterations):
        start = time.perf_counter()
        begin.record()
        out = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000)
        gpu.append(begin.elapsed_time(end))
    return out, dict(
        cold_ms=cold,
        wall_ms=wall,
        gpu_ms=gpu,
        p50_ms=float(np.median(wall)),
        p95_ms=float(np.percentile(wall, 95)),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("../model/Qwen3.5-397B-A17B-FP8"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--modes", default="baseline,optimized,compile,cudagraph,compile_cudagraph"
    )
    p.add_argument("--cases", default="all")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--no-pos", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-patch-fusion", action="store_true")
    p.add_argument("--no-batch-attention", action="store_true")
    p.add_argument("--inputs", type=Path)
    p.add_argument("--fresh-cache", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fresh_cache:
        for name, suffix in (
            ("TORCHINDUCTOR_CACHE_DIR", "inductor-cache"),
            ("TRITON_CACHE_DIR", "triton-cache"),
        ):
            cache = args.output / suffix
            if cache.exists():
                raise ValueError(f"Fresh cache directory already exists: {cache}")
            os.environ[name] = str(cache.resolve())
    torch.manual_seed(20260910)
    torch.set_num_threads(8)
    model, meta = load_model(args.checkpoint)
    reference_source = args.output / "reference_vit.py"
    reference_source.write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                "ee510269110af5ac3cd04c22e6dfc0460a27b2f8:"
                + str(SOURCE.relative_to(ROOT)),
            ]
        )
    )
    reference_model, _ = load_model(args.checkpoint, reference_source)
    meta.update(
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        versions={
            x: importlib.metadata.version(x)
            for x in ("torch", "triton", "transformers", "safetensors")
        },
        seed=20260910,
        commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    )
    (args.output / "environment.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    selected = [
        (n, g) for n, g in cases() if args.cases == "all" or n in args.cases.split(",")
    ]
    if args.inputs:
        selected += [
            (f"real_{f.stem}", f)
            for f in sorted(args.inputs.glob("*.pt"))
            if args.cases == "all" or f"real_{f.stem}" in args.cases.split(",")
        ]
    rows = []
    with torch.inference_mode():
        for name, grid in selected:
            if isinstance(grid, Path):
                saved = torch.load(grid, weights_only=True)
                x = saved["pixel_values"].cuda().bfloat16()
                grid = saved["grid_thw"].tolist()
            else:
                x = torch.randn(
                    sum(t * h * w for t, h, w in grid),
                    3 * 2 * 16 * 16,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
            g = torch.tensor(grid, device="cuda", dtype=torch.int64)
            if hasattr(model, "configure_vision"):
                model.configure_vision("baseline")
            reference = reference_model(x, g)
            assert all(
                torch.isfinite(t).all()
                for t in (reference.last_hidden_state, reference.pooler_output)
            )
            torch.save(
                dict(pixel_values=x.cpu(), grid_thw=g.cpu()),
                args.output / (name + "_input.pt"),
            )
            runtime_by_mode = {}
            for repeat in range(args.rounds):
                modes = args.modes.split(",")
                if repeat % 2:
                    modes = modes[::-1]
                for mode in modes:
                    if hasattr(model, "configure_vision"):
                        # Benchmark-only aliases permit paired, alternating ablations.
                        ablations = {
                            "eager_before": (False, False),
                            "eager_patch": (True, False),
                            "eager_batch": (False, True),
                        }
                        patch_on, batch_on = ablations.get(
                            mode,
                            (not args.no_patch_fusion, not args.no_batch_attention),
                        )
                        model.configure_vision(
                            "optimized" if mode in ablations else mode,
                            fused_rope=not args.no_rope,
                            fused_patch=patch_on,
                            batch_attention=batch_on,
                            fused_pos=not args.no_pos,
                            cache_metadata=not args.no_cache,
                        )
                    else:
                        assert mode == "baseline"
                    if mode in runtime_by_mode:
                        model._vision_runtime = runtime_by_mode[mode]
                    fn = (
                        (lambda: reference_model(x, g))
                        if mode == "baseline"
                        else (lambda: model(x, g))
                    )
                    out, timing = measure(fn, args.warmup, args.iterations)
                    if mode != "baseline":
                        runtime_by_mode[mode] = model._vision_runtime
                    accuracy = compare(out, reference)
                    stats = (
                        model.vision_stats() if hasattr(model, "vision_stats") else {}
                    )
                    row = dict(
                        case=name,
                        grid=grid,
                        mode=mode,
                        round=repeat,
                        accuracy=accuracy,
                        stats=stats,
                        **timing,
                    )
                    if mode in ("compile", "compile_cudagraph"):
                        assert (
                            stats.get("compile_graphs", 0) > 0
                            and stats.get("compile_failures", 0) == 0
                        ), stats
                    if mode in ("cudagraph", "compile_cudagraph"):
                        assert (
                            stats.get("graph_hits", 0) > 0
                            and stats.get("graph_failures", 0) == 0
                        ), stats
                    row["tokens_per_second"] = x.shape[0] * 1000 / row["p50_ms"]
                    rows.append(row)
                    (args.output / "results.json").write_text(
                        json.dumps(rows, indent=2)
                    )
                    print(
                        name,
                        mode,
                        repeat,
                        round(row["p50_ms"], 3),
                        accuracy,
                        stats,
                        flush=True,
                    )
                    assert all(v["passed"] for v in accuracy.values()), (
                        name,
                        mode,
                        accuracy,
                    )
                    if args.profile and repeat == 0:
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ],
                            record_shapes=True,
                        ) as prof:
                            fn()
                            torch.cuda.synchronize()
                        prof.export_chrome_trace(
                            str(args.output / f"{name}_{mode}_trace.json")
                        )
                        (args.output / f"{name}_{mode}_profile.txt").write_text(
                            prof.key_averages().table(
                                sort_by="self_cuda_time_total", row_limit=40
                            )
                        )
    keys = [
        "case",
        "mode",
        "round",
        "p50_ms",
        "p95_ms",
        "cold_ms",
        "tokens_per_second",
        "peak_allocated",
        "peak_reserved",
    ]
    with (args.output / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
