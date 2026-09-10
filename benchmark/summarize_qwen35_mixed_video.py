"""Aggregate the complete-video benchmark without mixing cold starts or fallbacks."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    rows = json.loads((args.input / "results.json").read_text())
    env = json.loads((args.input / "environment.json").read_text())
    modes = env["arguments"]["modes"].split(",")
    rounds = int(env["arguments"]["rounds"])
    iterations = int(env["arguments"]["iterations"])
    if not args.allow_partial:
        assert len(rows) == len(modes) * rounds, "Incomplete experiment"
    summary = []
    raw = []
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        if not selected:
            continue
        if not args.allow_partial:
            assert {r["round"] for r in selected} == set(range(rounds))
        for r in selected:
            assert r["accuracy"]["passed"]
            assert len(r["wall_ms"]) == len(r["gpu_ms"]) == iterations
            if mode == "compile_cudagraph":
                assert r["stats"]["graph_hits"] >= iterations
                assert r["stats"]["graph_failures"] == 0
            if mode.startswith("compile"):
                assert r["stats"]["compile_failures"] == 0
            for i, (wall, event) in enumerate(zip(r["wall_ms"], r["gpu_ms"])):
                raw.append(
                    dict(
                        mode=mode,
                        round=r["round"],
                        iteration=i,
                        wall_ms=wall,
                        cuda_event_ms=event,
                    )
                )
        wall = [v for r in selected for v in r["wall_ms"]]
        event = [v for r in selected for v in r["gpu_ms"]]
        median = float(np.median(wall))
        summary.append(
            dict(
                mode=mode,
                rounds=len(selected),
                samples=len(wall),
                wall_p50_ms=median,
                wall_p95_ms=float(np.percentile(wall, 95)),
                cuda_p50_ms=float(np.median(event)),
                cuda_p95_ms=float(np.percentile(event, 95)),
                videos_per_second=env["video_count"] * 1000 / median,
                frames_per_second=env["sampled_frames"] * 1000 / median,
                peak_allocated_gib=max(r["peak_allocated"] for r in selected) / 2**30,
                peak_reserved_gib=max(r["peak_reserved"] for r in selected) / 2**30,
                cold_ms=[r["cold_ms"] for r in selected],
                round_p50_ms=[r["p50_ms"] for r in selected],
                graph_hits=sum(r["stats"].get("graph_hits", 0) for r in selected),
                graph_captures=sum(
                    r["stats"].get("graph_captures", 0) for r in selected
                ),
                compile_graphs=sum(
                    r["stats"].get("compile_graphs", 0) for r in selected
                ),
                compile_seconds=sum(
                    r["stats"].get("compile_seconds", 0) for r in selected
                ),
                max_abs=max(r["accuracy"]["max_abs"] for r in selected),
                max_relative_l2=max(r["accuracy"]["max_relative_l2"] for r in selected),
                min_cosine=min(r["accuracy"]["min_cosine"] for r in selected),
            )
        )
    by_mode = {r["mode"]: r for r in summary}
    for row in summary:
        for reference in ("baseline", "optimized", "compile"):
            if reference in by_mode:
                row["latency_reduction_vs_" + reference + "_pct"] = 100 * (
                    1 - row["wall_p50_ms"] / by_mode[reference]["wall_p50_ms"]
                )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                complete=len(rows) == len(modes) * rounds,
                environment=env,
                summary=summary,
            ),
            indent=2,
        )
    )
    for filename, records in [("summary.csv", summary), ("raw-times.csv", raw)]:
        with (args.output / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for row in summary:
        print(
            row["mode"],
            "p50_ms",
            round(row["wall_p50_ms"], 3),
            "p95_ms",
            round(row["wall_p95_ms"], 3),
            "peak_GiB",
            round(row["peak_allocated_gib"], 3),
            "max_abs",
            row["max_abs"],
            flush=True,
        )


if __name__ == "__main__":
    main()
