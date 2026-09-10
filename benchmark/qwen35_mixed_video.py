"""Benchmark one complete mixed-video batch with a CPU-stored pristine reference."""

import argparse
import csv
import gc
import importlib.metadata
import json
import subprocess
import time
import traceback
import types
from pathlib import Path

import torch
from qwen35_vision import ROOT, SOURCE, load_model, measure

FIELDS = ("last_hidden_state", "pooler_output")


def safe_reference_norm(module, x):
    """Independent reference workaround; only partition native LayerNorm rows."""
    if x.is_cuda and x.numel() >= 2**31:
        return torch.cat(
            [
                torch.nn.functional.layer_norm(
                    part,
                    module.normalized_shape,
                    module.weight,
                    module.bias,
                    module.eps,
                )
                for part in x.split(1000000)
            ],
            dim=0,
        )
    return torch.nn.functional.layer_norm(
        x, module.normalized_shape, module.weight, module.bias, module.eps
    )


def install_safe_reference_norm(model):
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.forward = types.MethodType(safe_reference_norm, module)


def compare_videos(actual, reference, grid):
    results = []
    offsets = {name: 0 for name in FIELDS}
    for index, (t, h, w) in enumerate(grid):
        fields = {}
        for name, divisor in zip(FIELDS, (1, 4)):
            count = t * h * w // divisor
            offset = offsets[name]
            a = getattr(actual, name)[offset : offset + count].cpu()
            r = reference[name][offset : offset + count]
            assert a.shape == r.shape and a.dtype == r.dtype
            # Bound temporary memory to one video; comparisons are outside timing.
            finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
            rf = r.double()
            limit = 0.05 * max(1.0, rf.abs().max().item())
            if torch.equal(a, r):
                rel, cosine, maximum = 0.0, 1.0, 0.0
            else:
                af = a.double()
                delta = af - rf
                rel = (delta.norm() / rf.norm().clamp_min(1e-12)).item()
                cosine = torch.nn.functional.cosine_similarity(
                    af.flatten(), rf.flatten(), dim=0
                ).item()
                maximum = delta.abs().max().item()
            fields[name] = dict(
                relative_l2=rel,
                cosine=cosine,
                max_abs=maximum,
                max_abs_limit=limit,
                passed=finite and rel <= 0.01 and cosine >= 0.999 and maximum <= limit,
            )
            offsets[name] += count
        results.append(dict(video=index, accuracy=fields))
    for name in FIELDS:
        assert (
            offsets[name] == getattr(actual, name).shape[0] == reference[name].shape[0]
        )
    checks = [v for row in results for v in row["accuracy"].values()]
    return dict(
        passed=all(v["passed"] for v in checks),
        max_abs=max(v["max_abs"] for v in checks),
        max_relative_l2=max(v["relative_l2"] for v in checks),
        min_cosine=min(v["cosine"] for v in checks),
        per_video=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", default="optimized,compile,cudagraph")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--safe-native-norm",
        action="store_true",
        help="Explicit native LayerNorm overflow workaround for reference/baseline",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue completed mode/round records"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    stage = "load_input"
    rows = []
    previous = None
    if args.resume:
        previous = json.loads((args.output / "environment.json").read_text())
        for key, value in vars(args).items():
            if key != "resume":
                value = str(value) if isinstance(value, Path) else value
                assert (
                    previous["arguments"][key] == value
                ), f"Resume argument mismatch: {key}"
        rows = json.loads((args.output / "results.json").read_text())
        assert all(
            r["accuracy"]["passed"] and len(r["wall_ms"]) == args.iterations
            for r in rows
        )
    elif (args.output / "results.json").exists():
        raise ValueError(
            "Output already contains results; use --resume or another directory"
        )
    completed = {(r["round"], r["mode"]) for r in rows}
    assert len(completed) == len(rows)
    torch.set_num_threads(8)
    try:
        manifest = json.loads(args.manifest.read_text())
        inputs = torch.load(args.input, weights_only=True, mmap=True)
        grid = inputs["grid_thw"].tolist()
        assert grid == [c["grid_thw"] for c in manifest["clips"]]
        x = inputs["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
        g = inputs["grid_thw"].cuda()
        del inputs
        reference_source = args.reference_dir / "reference_vit.py"
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
        reference_path = args.reference_dir / "outputs.pt"
        reference_meta = args.reference_dir / "metadata.json"
        with torch.inference_mode():
            if not reference_path.exists():
                stage = "pristine_full_batch_reference"
                print("START", stage, "tokens", x.shape[0], flush=True)
                ref_model, meta = load_model(args.checkpoint, reference_source)
                if args.safe_native_norm:
                    install_safe_reference_norm(ref_model)
                start = time.perf_counter()
                out = ref_model(x, g)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                reference = {name: getattr(out, name).cpu() for name in FIELDS}
                torch.save(reference, reference_path)
                meta["safe_native_norm"] = args.safe_native_norm
                meta.update(
                    input_sha256=manifest["input_sha256"],
                    full_batch_seconds=elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                )
                reference_meta.write_text(json.dumps(meta, indent=2))
                print("DONE reference", elapsed, "seconds", flush=True)
                del ref_model, out, reference
                gc.collect()
                torch.cuda.empty_cache()
            assert (
                json.loads(reference_meta.read_text())["input_sha256"]
                == manifest["input_sha256"]
            )
            assert (
                json.loads(reference_meta.read_text()).get("safe_native_norm", False)
                == args.safe_native_norm
            ), "Reference LayerNorm policy mismatch"
            if args.reference_only:
                return
            reference = torch.load(reference_path, weights_only=True, mmap=True)
            model, meta = load_model(args.checkpoint)
            if args.safe_native_norm:
                install_safe_reference_norm(model)
            meta["safe_native_norm"] = args.safe_native_norm
            meta.update(
                arguments={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                gpu=torch.cuda.get_device_name(),
                versions={
                    name: importlib.metadata.version(name)
                    for name in ("torch", "triton", "transformers")
                },
                video_count=len(grid),
                sampled_frames=manifest["sampled_frames"],
                patch_tokens=x.shape[0],
                input_sha256=manifest["input_sha256"],
                note="Full input in one encoder call. CPU reference and per-video accuracy checks excluded from timing. Runtime recreated per mode/round to release graph pools.",
            )
            if previous is not None:
                for key in (
                    "gpu",
                    "versions",
                    "input_sha256",
                    "safe_native_norm",
                    "backend",
                ):
                    assert (
                        meta[key] == previous[key]
                    ), f"Resume environment mismatch: {key}"
                (args.output / f"resume-environment-{time.time_ns()}.json").write_text(
                    json.dumps(meta, indent=2)
                )
            else:
                (args.output / "environment.json").write_text(
                    json.dumps(meta, indent=2)
                )
            for repeat in range(args.rounds):
                modes = args.modes.split(",")
                if repeat % 2:
                    modes.reverse()
                for mode in modes:
                    if (repeat, mode) in completed:
                        print("SKIP completed", repeat, mode, flush=True)
                        continue
                    stage = f"round_{repeat}_{mode}"
                    print("START", stage, flush=True)
                    model.configure_vision(mode)
                    gc.collect()
                    torch.cuda.empty_cache()
                    out, timing = measure(
                        lambda: model(x, g), args.warmup, args.iterations
                    )
                    stats = (
                        dict(model._vision_runtime.stats)
                        if model._vision_runtime
                        else {}
                    )
                    accuracy = compare_videos(out, reference, grid)
                    assert accuracy["passed"], accuracy
                    row = dict(
                        case="mixed_video128",
                        mode=mode,
                        round=repeat,
                        accuracy=accuracy,
                        stats=stats,
                        **timing,
                    )
                    rows.append(row)
                    (args.output / "results.json").write_text(
                        json.dumps(rows, indent=2)
                    )
                    flat_keys = [
                        "case",
                        "mode",
                        "round",
                        "cold_ms",
                        "p50_ms",
                        "p95_ms",
                        "peak_allocated",
                        "peak_reserved",
                    ]
                    with (args.output / "results.csv").open("w") as stream:
                        writer = csv.DictWriter(stream, fieldnames=flat_keys)
                        writer.writeheader()
                        writer.writerows(
                            {key: r[key] for key in flat_keys} for r in rows
                        )
                    print(
                        "DONE",
                        stage,
                        "p50_ms",
                        timing["p50_ms"],
                        "max_abs",
                        accuracy["max_abs"],
                        "stats",
                        stats,
                        flush=True,
                    )
                    del out
                    model.configure_vision("baseline")
                    gc.collect()
                    torch.cuda.empty_cache()
    except Exception as exc:
        failure = dict(
            stage=stage,
            type=type(exc).__name__,
            reason=str(exc),
            traceback=traceback.format_exc(),
        )
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2))
        raise


if __name__ == "__main__":
    main()
