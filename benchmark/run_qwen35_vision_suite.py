"""Run each input case in an isolated process with stable per-mode runtimes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from qwen35_vision import cases

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--inputs", type=Path)
p.add_argument("--checkpoint", type=Path)
p.add_argument(
    "--modes", default="baseline,optimized,compile,cudagraph,compile_cudagraph"
)
p.add_argument("--cases", default="all")
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--resume", action="store_true")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
selected = [n for n, _ in cases()]
if a.inputs:
    selected += ["real_" + f.stem for f in sorted(a.inputs.glob("*.pt"))]
if a.cases != "all":
    selected = [n for n in selected if n in a.cases.split(",")]
assert selected
for name in selected:
    out = a.output / name
    if a.resume and (out / "complete.json").exists():
        continue
    command = [
        sys.executable,
        str(Path(__file__).with_name("qwen35_vision.py")),
        "--output",
        str(out),
        "--cases",
        name,
        "--modes",
        a.modes,
        "--rounds",
        str(a.rounds),
        "--iterations",
        str(a.iterations),
        "--warmup",
        str(a.warmup),
    ]
    if a.inputs:
        command += ["--inputs", str(a.inputs)]
    if a.checkpoint:
        command += ["--checkpoint", str(a.checkpoint)]
    print("START", name, flush=True)
    with (a.output / (name + ".log")).open("w") as f:
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        print("FAILED", name, flush=True)
        sys.exit(result.returncode)
    (out / "complete.json").write_text(json.dumps(dict(command=command, returncode=0)))
    print("PASS", name, flush=True)
print("SUITE COMPLETE", len(selected), flush=True)
