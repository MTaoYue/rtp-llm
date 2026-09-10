"""Build reproducible mixed-resolution videos for the standalone vision benchmark."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from transformers import AutoVideoProcessor


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(dataset):
    records = []
    for path in sorted(dataset.rglob("*.mp4")):
        if "__MACOSX" in path.parts:
            continue
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration
                else container.duration / 1e6
            )
            if duration >= 3:
                records.append(
                    dict(
                        source=str(path.resolve()),
                        width=stream.width,
                        height=stream.height,
                        duration=duration,
                        fps=float(stream.average_rate),
                        sha256=sha256(path),
                    )
                )
    return records


def make_plan(sources, batch_unit, batch_size, derive, seed):
    rng = np.random.default_rng(seed)
    if batch_unit == "frames":
        if batch_size != 128:
            raise ValueError(
                "Frame-budget fixture currently defines exactly 128 frames"
            )
        counts = [18, 24, 38, 48]
    else:
        counts = [18 + 6 * (i % 6) for i in range(batch_size)]
        rng.shuffle(counts)
    if not derive and len(sources) < len(counts):
        raise ValueError(
            "Insufficient distinct source videos; use --derive to explicitly permit derived clips"
        )
    clips = []
    used = set()
    # Width/height before optional portrait transpose, all multiples of 32.
    sizes = [(640, 352), (864, 480), (1024, 576), (1280, 704)]
    for i, count in enumerate(counts):
        duration = count / 6
        candidates = [
            s
            for s in sources
            if s["duration"] + 1e-6 >= duration and (derive or s["source"] not in used)
        ]
        if not candidates:
            raise ValueError(f"No available source for {duration}s clip {i}")
        source = candidates[int(rng.integers(len(candidates)))]
        used.add(source["source"])
        start = float(rng.uniform(0, max(0, source["duration"] - duration)))
        width, height = sizes[(i // 2) % len(sizes)]
        portrait = bool(i % 2) if derive else source["height"] > source["width"]
        if portrait:
            width, height = height, width
        if not derive:
            # Preserve the original aspect ratio and never upscale actual source dimensions.
            scale = min(width / source["width"], height / source["height"], 1.0)
            width = max(32, int(source["width"] * scale) // 32 * 32)
            height = max(32, int(source["height"] * scale) // 32 * 32)
        clips.append(
            dict(
                index=i,
                source=source["source"],
                source_sha256=source["sha256"],
                start_seconds=start,
                duration_seconds=duration,
                fps=6,
                sampled_frames=count,
                width=width,
                height=height,
                orientation="portrait" if portrait else "landscape",
                transform="center_crop_resize" if derive else "resize",
                grid_thw=[count // 2, height // 16, width // 16],
            )
        )
    return clips


def sample_indices(timestamps, targets):
    right = np.searchsorted(timestamps, targets).clip(0, len(timestamps) - 1)
    left = np.maximum(0, right - 1)
    return np.where(
        np.abs(timestamps[left] - targets) <= np.abs(timestamps[right] - targets),
        left,
        right,
    )


def decode(source):
    start = time.perf_counter()
    frames, timestamps = [], []
    with av.open(source) as container:
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError(f"Missing presentation timestamp: {source}")
            timestamps.append(float(frame.time))
            frames.append(frame.to_ndarray(format="rgb24"))
    timestamps = np.asarray(timestamps)
    timestamps -= timestamps[0]
    assert np.all(np.diff(timestamps) > 0)
    return frames, timestamps, time.perf_counter() - start


def transform(frame, width, height, crop):
    image = Image.fromarray(frame)
    crop_box = (0, 0, image.width, image.height)
    if crop:
        ratio = width / height
        if image.width / image.height > ratio:
            cropped_width = int(image.height * ratio)
            left = (image.width - cropped_width) // 2
            crop_box = (left, 0, left + cropped_width, image.height)
        else:
            cropped_height = int(image.width / ratio)
            top = (image.height - cropped_height) // 2
            crop_box = (0, top, image.width, top + cropped_height)
        image = image.crop(crop_box)
    return np.asarray(image.resize((width, height), Image.Resampling.BICUBIC)), crop_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-unit", choices=("videos", "frames"), default="videos")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--derive", action="store_true")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = inventory(args.dataset)
    clips = make_plan(sources, args.batch_unit, args.batch_size, args.derive, args.seed)
    token_count = sum(int(np.prod(c["grid_thw"])) for c in clips)
    manifest = dict(
        batch_unit=args.batch_unit,
        batch_size=args.batch_size,
        seed=args.seed,
        derived_clips=args.derive,
        distinct_sources=len(set(c["source"] for c in clips)),
        video_count=len(clips),
        sampled_frames=sum(c["sampled_frames"] for c in clips),
        temporal_groups=sum(c["grid_thw"][0] for c in clips),
        patch_tokens=token_count,
        input_bf16_gib=token_count * 1536 * 2 / 2**30,
        all_spatial_lengths_equal=len(
            set(c["grid_thw"][1] * c["grid_thw"][2] for c in clips)
        )
        == 1,
        clips=clips,
        sources=sources,
    )
    target = args.output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("clips", "sources")}
        ),
        flush=True,
    )
    if args.manifest_only:
        return
    torch.set_num_threads(8)
    processor = AutoVideoProcessor.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    decoded = {
        source: decode(source) for source in sorted(set(c["source"] for c in clips))
    }
    manifest["decode_seconds"] = sum(value[2] for value in decoded.values())
    manifest["decode_cache"] = (
        "Each distinct source decoded once; clips select timestamps from that cache"
    )
    parts, grids = [], []
    preprocess_start = time.perf_counter()
    for clip in clips:
        frames, timestamps, _ = decoded[clip["source"]]
        targets = clip["start_seconds"] + np.arange(clip["sampled_frames"]) / 6
        indices = sample_indices(timestamps, targets)
        assert len(np.unique(indices)) == clip["sampled_frames"]
        rendered = [
            transform(frames[index], clip["width"], clip["height"], args.derive)
            for index in indices
        ]
        clip["sample_indices"] = indices.tolist()
        clip["target_timestamps"] = targets.tolist()
        clip["actual_timestamps"] = timestamps[indices].tolist()
        clip["crop_box"] = list(rendered[0][1])
        output = processor(
            videos=[np.stack([frame for frame, _ in rendered])],
            do_sample_frames=False,
            do_resize=False,
            return_tensors="pt",
        )
        grid = output["video_grid_thw"]
        assert grid.tolist() == [clip["grid_thw"]], (grid, clip)
        pixels = output["pixel_values_videos"].to(torch.bfloat16)
        assert pixels.shape == (int(np.prod(clip["grid_thw"])), 1536)
        assert torch.isfinite(pixels).all()
        parts.append(pixels)
        grids.append(grid)
        if clip["index"] % 16 == 0:
            print("prepared", clip["index"], clip["grid_thw"], flush=True)
    manifest["preprocess_seconds"] = time.perf_counter() - preprocess_start
    pack_start = time.perf_counter()
    inputs = dict(pixel_values=torch.cat(parts), grid_thw=torch.cat(grids))
    manifest["pack_seconds"] = time.perf_counter() - pack_start
    input_dir = args.output / "inputs"
    input_dir.mkdir(exist_ok=True)
    input_path = input_dir / "mixed128.pt"
    save_start = time.perf_counter()
    torch.save(inputs, input_path)
    manifest["save_seconds"] = time.perf_counter() - save_start
    manifest["input_path"] = str(input_path.resolve())
    manifest["input_sha256"] = sha256(input_path)
    manifest["input_dtype"] = str(inputs["pixel_values"].dtype)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("Saved", input_path, manifest["input_sha256"], flush=True)


if __name__ == "__main__":
    main()
