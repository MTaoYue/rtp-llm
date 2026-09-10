"""Save real local video frames and processor outputs; no network downloads."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoVideoProcessor

p = argparse.ArgumentParser()
p.add_argument("--dataset", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
image_processor = AutoImageProcessor.from_pretrained(
    a.checkpoint, local_files_only=True
)
video_processor = AutoVideoProcessor.from_pretrained(
    a.checkpoint, local_files_only=True
)
records = []
for index, path in enumerate(sorted(a.dataset.glob("*.mp4"))[:3]):
    start = time.perf_counter()
    frames = []
    timestamps = []
    with av.open(str(path)) as container:
        for number, frame in enumerate(container.decode(video=0)):
            if number % 15 == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                timestamps.append(frame.time)
                if len(frames) == 8:
                    break
    if not frames:
        raise ValueError(f"No frames decoded: {path}")
    while len(frames) < 8:
        frames.append(frames[-1])
        timestamps.append(timestamps[-1])
    decode = time.perf_counter() - start
    for kind in ("image", "video"):
        start = time.perf_counter()
        if kind == "image":
            out = image_processor(
                images=frames[0],
                return_tensors="pt",
                min_pixels=448 * 448,
                max_pixels=448 * 448,
            )
            pixels, grid = out["pixel_values"], out["image_grid_thw"]
        else:
            out = video_processor(
                videos=[np.stack(frames)],
                return_tensors="pt",
                do_sample_frames=False,
                size=dict(shortest_edge=448 * 448, longest_edge=448 * 448),
            )
            pixels, grid = out["pixel_values_videos"], out["video_grid_thw"]
        preprocess = time.perf_counter() - start
        target = a.output / f"clip{index}_{kind}.pt"
        torch.save(dict(pixel_values=pixels, grid_thw=grid), target)
        record = dict(
            source=str(path),
            kind=kind,
            timestamps=timestamps,
            decode_seconds=decode,
            preprocess_seconds=preprocess,
            grid=grid.tolist(),
            shape=list(pixels.shape),
            input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )
        records.append(record)
        print(record, flush=True)
(a.output / "manifest.json").write_text(json.dumps(records, indent=2))
