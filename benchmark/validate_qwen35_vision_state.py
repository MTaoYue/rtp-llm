"""Real-weight graph state/capacity regression, separate from timing."""

import argparse
import json
import subprocess
from pathlib import Path

import torch
from qwen35_vision import ROOT, SOURCE, compare, load_model

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(20260910)
torch.set_num_threads(8)
source = a.output / "reference_vit.py"
source.write_bytes(
    subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            "ee510269110af5ac3cd04c22e6dfc0460a27b2f8:" + str(SOURCE.relative_to(ROOT)),
        ]
    )
)
model, meta = load_model(a.checkpoint)
reference, _ = load_model(a.checkpoint, source)
model.configure_vision("cudagraph")
grids = [((1, 2, 4),), ((1, 4, 2),), ((1, 2, 2), (1, 2, 4)), ((1, 2, 4), (1, 2, 2))] + [
    ((1, 2, n),) for n in [6, 8, 10, 12, 14]
]
rows = []
with torch.inference_mode():
    saved = []
    cfg = model.config
    width = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2
    for i, grid in enumerate(grids):
        x = torch.randn(
            sum(t * h * w for t, h, w in grid),
            width,
            device="cuda",
            dtype=torch.bfloat16,
        )
        g = torch.tensor(grid, device="cuda", dtype=torch.int64)
        out = model(x, g)
        metrics = compare(out, reference(x, g))
        assert all(v["passed"] for v in metrics.values()), metrics
        saved.append(
            (x, g, out, out.last_hidden_state.clone(), out.pooler_output.clone())
        )
        torch.save(
            dict(pixel_values=x.cpu(), grid_thw=g.cpu()), a.output / f"input_{i}.pt"
        )
        rows.append(dict(grid=grid, accuracy=metrics, stats=model.vision_stats()))
    assert model.vision_stats()["graph_captures"] == 8, model.vision_stats()
    assert model.vision_stats()["graph_capacity_fallbacks"] == 1, model.vision_stats()
    x, g, old, hidden, pool = saved[0]
    for changed in [x * 0.5, -x, x]:
        metrics = compare(model(changed, g), reference(changed, g))
        assert all(v["passed"] for v in metrics.values()), metrics
        assert torch.equal(old.last_hidden_state, hidden)
        assert torch.equal(old.pooler_output, pool)
        rows.append(dict(grid=grids[0], accuracy=metrics, stats=model.vision_stats()))
    assert model.vision_stats()["graph_hits"] == 3, model.vision_stats()
    assert model.vision_stats()["graph_failures"] == 0, model.vision_stats()
(a.output / "results.json").write_text(
    json.dumps(
        dict(environment=meta, checks=rows, stats=model.vision_stats()), indent=2
    )
)
print("PASS real-weight state checks", len(rows), model.vision_stats(), flush=True)
