"""Aggregate validated benchmark samples; fail on missing/failed cases."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--artifacts", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
modes = ["baseline", "optimized", "compile", "cudagraph", "compile_cudagraph"]
rows = []
cases = {}
for directory in sorted((a.artifacts / "full").iterdir()):
    if not directory.is_dir() or not (directory / "complete.json").exists():
        continue
    replacement = a.artifacts / "recheck" / directory.name
    if (replacement / "complete.json").exists():
        directory = replacement
    data = json.loads((directory / "results.json").read_text())
    assert len(data) == 25, (directory, len(data))
    rows.extend(data)
    stats = {}
    for mode in modes:
        subset = [r for r in data if r["mode"] == mode]
        assert len(subset) == 5 and all(
            v["passed"] for r in subset for v in r["accuracy"].values()
        )
        samples = [v for r in subset for v in r["wall_ms"]]
        ordered = sorted(samples)
        stats[mode] = dict(
            p50_ms=statistics.median(samples),
            p95_ms=ordered[math.ceil(len(ordered) * 0.95) - 1],
            gpu_p50_ms=statistics.median([v for r in subset for v in r["gpu_ms"]]),
            round_p50_ms=[r["p50_ms"] for r in subset],
            first_call_ms=subset[0]["cold_ms"],
            max_process_allocated_gib=max(r["peak_allocated"] for r in subset) / 2**30,
            max_process_reserved_gib=max(r["peak_reserved"] for r in subset) / 2**30,
            tokens_per_second=statistics.median(
                [r["tokens_per_second"] for r in subset]
            ),
            max_relative_l2=max(
                v["relative_l2"] for r in subset for v in r["accuracy"].values()
            ),
            max_abs_error=max(
                v["max_abs"] for r in subset for v in r["accuracy"].values()
            ),
            execution_stats=subset[-1]["stats"],
        )
    cases[data[0]["case"]] = dict(grid=data[0]["grid"], metrics=stats)
assert len(cases) == 18, len(cases)
summary = {}
for mode in modes[1:]:
    ratios = [
        v["metrics"][mode]["p50_ms"] / v["metrics"]["baseline"]["p50_ms"]
        for v in cases.values()
    ]
    geo = math.exp(statistics.mean(map(math.log, ratios)))
    summary[mode] = dict(
        geomean_speedup=1 / geo,
        geomean_latency_reduction_percent=(1 - geo) * 100,
        worst_latency_ratio=max(ratios),
        regressions_over_5_percent=[
            k
            for k, v in cases.items()
            if v["metrics"][mode]["p50_ms"] > v["metrics"]["baseline"]["p50_ms"] * 1.05
        ],
    )
assert summary["optimized"]["geomean_latency_reduction_percent"] >= 10
assert not summary["optimized"]["regressions_over_5_percent"]
ablations = []
for path in sorted(a.artifacts.glob("ablation_*/results.json")):
    data = json.loads(path.read_text())
    assert len(data) == 10
    assert all(v["passed"] for r in data for v in r["accuracy"].values())
    b = statistics.median(
        [x for r in data if r["mode"] == "baseline" for x in r["wall_ms"]]
    )
    o = statistics.median(
        [x for r in data if r["mode"] == "optimized" for x in r["wall_ms"]]
    )
    ablations.append(
        dict(
            name=path.parent.name,
            baseline_ms=b,
            optimized_ms=o,
            speedup=b / o,
            first_call_ms=next(r["cold_ms"] for r in data if r["mode"] == "optimized"),
        )
    )
assert len(ablations) == 10, len(ablations)
result = dict(summary=summary, cases=cases, ablations=ablations)
(a.artifacts / "summary.json").write_text(json.dumps(result, indent=2))
with (a.artifacts / "summary.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "case",
            "mode",
            "p50_ms",
            "p95_ms",
            "gpu_p50_ms",
            "first_call_ms",
            "tokens_per_second",
            "max_process_allocated_gib",
            "max_relative_l2",
            "max_abs_error",
        ]
    )
    for name, v in cases.items():
        for mode, m in v["metrics"].items():
            writer.writerow(
                [name, mode]
                + [
                    m[k]
                    for k in (
                        "p50_ms",
                        "p95_ms",
                        "gpu_p50_ms",
                        "first_call_ms",
                        "tokens_per_second",
                        "max_process_allocated_gib",
                        "max_relative_l2",
                        "max_abs_error",
                    )
                ]
            )
lines = [
    "# Qwen3.5 VL Vision Encoder 单卡优化验证",
    "",
    "## 环境与口径",
    "",
    "- 主仓库基线：0c87fb54c3；开源子模块基线：ee510269110af5ac3cd04c22e6dfc0460a27b2f8。",
    "- vLLM 参考：a97dacb710，qwen3_vl.py 的位置插值、元数据和图捕获设计。",
    "- GPU：单张 NVIDIA L20D（SM 10.3），CUDA_VISIBLE_DEVICES=0。PyTorch 2.11.0+cu130、Triton 3.6.0、Transformers 5.16.1。",
    "- 本地 Qwen3.5-397B-A17B-FP8 检查点中 333 个视觉权重均为 BF16；严格加载，只运行视觉模型。",
    "- 所有模式使用相同 SDPA 后端；基线直接从固定 Git 提交提取原始源码，单独实例化。",
    "- 18 场景 × 5 模式 × 5 轮，每轮预热 20 次、测量 100 次；相邻轮反转模式顺序。",
    "- 主表为 500 次墙钟样本的 p50，包含 CPU 元数据处理、同步、图输入复制和独立输出复制；CUDA Events、p95、冷启动和显存见 summary.csv。",
    "- 每个场景独立进程；不同模式运行时在五轮间保持，图缓存容量/动态形状另行回归。显存为进程峰值，包含两个视觉模型及本进程保留的模式缓存，不等同于单模式独占部署显存。",
    "- first_call_ms 是当前磁盘缓存状态下该模式首次调用；编译/捕获统计单独保存在原始 JSON。",
    "",
    "## 性能结果",
    "",
    "| 模式 | 几何平均加速 | 几何平均延迟下降 | 超过5%退化场景 |",
    "|---|---:|---:|---:|",
]
for mode, s in summary.items():
    lines.append(
        f"| {mode} | {s['geomean_speedup']:.2f}× | {s['geomean_latency_reduction_percent']:.1f}% | {len(s['regressions_over_5_percent'])} |"
    )
lines += [
    "",
    "| 场景 | baseline ms | optimized ms | compile ms | CUDA Graph ms | compile+Graph ms |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, v in cases.items():
    lines.append(
        "| "
        + name
        + " | "
        + " | ".join(f"{v['metrics'][m]['p50_ms']:.3f}" for m in modes)
        + " |"
    )
max_l2 = max(
    m["max_relative_l2"] for c in cases.values() for m in c["metrics"].values()
)
max_abs = max(m["max_abs_error"] for c in cases.values() for m in c["metrics"].values())
lines += [
    "",
    "## 正确性与实现",
    "",
    f"全部 450 个场景/模式/轮次检查通过。两个输出（last_hidden_state、pooler_output）的最大相对 L2 误差为 {max_l2:.8g}，最大绝对误差为 {max_abs:.8g}。",
    "",
    "- 单次准备 grid、cu_seqlens、长度和位置元数据；最多保留32个元数据项。",
    "- RoPE 使用独立输出，支持 packed QKV 非连续视图；FP32 运算禁止 FMA 合并。",
    "- 位置插值保留 RTP 的 BF16 乘法及逐步加法舍入，维持空间合并和时间帧顺序。",
    "- compile 保留原生 LayerNorm/GELU 边界。初版 Inductor 改写曾产生超过1%的最终误差，已定位并修复，未放宽阈值。",
    "- 显式 CUDA Graph 以完整有序 grid、device、dtype、shape、stride、backend 建键，最多8个形状。回放复制输入并克隆两个输出，通过事件保证跨流顺序。",
    "- 编译/捕获失败记录并回退；CPU、训练及已安装的内部 attention/RoPE patch 保留原路径。",
    "- 权重重载和设备转换清空运行时；常规参数版本变化自动失效。直接修改 inference tensor 权重后须调用 clear_vision_cache()。",
    "",
    "## 消融",
    "",
    "| 变体与场景 | 同次基线 ms | 变体 ms | 加速 | 首次调用 ms |",
    "|---|---:|---:|---:|---:|",
]
for row in ablations:
    lines.append(
        f"| {row['name']} | {row['baseline_ms']:.3f} | {row['optimized_ms']:.3f} | {row['speedup']:.2f}× | {row['first_call_ms']:.2f} |"
    )
lines += [
    "",
    "位置插值融合主要影响首次遇到的 grid；缓存命中后的 steady-state 不重复执行插值。no_cache 与 no_cache_no_pos 用于隔离插值融合收益。",
    "",
    "## 使用与产物",
    "",
    "默认 CUDA 推理使用 optimized。设置 RTP_QWEN35_VISION_MODE=baseline 可回退；设为 compile、cudagraph 或 compile_cudagraph 可显式启用对应模式。也可配置 vision_execution_mode 或调用 model.configure_vision(mode, fused_rope=True, fused_pos=True, cache_metadata=True, graph_capacity=8)。",
    "",
    "复现命令与选项见开源子模块 benchmark/README_qwen35_vision.md。完整原始样本、真实视频处理清单、输入张量、环境、编译计数及 trace 存于 artifacts/。所有 GPU 作业串行执行；本结果限独立 vision encoder，未运行语言模型生成。",
    "",
]
a.output.write_text("\n".join(lines))
print(json.dumps(summary, indent=2))
print("Report:", a.output)
