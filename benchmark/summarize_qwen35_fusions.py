"""Summarize paired eager fusion benchmarks without rerunning GPU work."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["mode"])].append(row)
    summary = []
    for (case, mode), group in groups.items():
        group.sort(key=lambda row: row["round"])
        wall = [v for row in group for v in row["wall_ms"]]
        gpu = [v for row in group for v in row["gpu_ms"]]
        accuracy = [v for row in group for v in row["accuracy"].values()]
        assert all(v["passed"] for v in accuracy), (case, mode)
        summary.append(
            dict(
                case=case,
                mode=mode,
                samples=len(wall),
                rounds=len(group),
                p50_ms=float(np.median(wall)),
                p95_ms=float(np.percentile(wall, 95)),
                gpu_p50_ms=float(np.median(gpu)),
                cold_ms=group[0]["cold_ms"],
                peak_allocated_gib=max(r["peak_allocated"] for r in group) / 2**30,
                peak_reserved_gib=max(r["peak_reserved"] for r in group) / 2**30,
                round_p50_ms=[r["p50_ms"] for r in group],
                max_abs=max(v["max_abs"] for v in accuracy),
                max_relative_l2=max(v["relative_l2"] for v in accuracy),
                min_cosine=min(v["cosine"] for v in accuracy),
                stats=group[-1]["stats"],
            )
        )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    def read(name):
        return json.loads((a.artifacts / name).read_text())

    before = aggregate(read("b128-eager-compile/results.json"))
    after = aggregate(read("b128-eager-fusions/results.json"))
    matrix = aggregate(read("fusion-matrix/results.json"))
    assert len(before) == 2 and all(
        r["samples"] == 500 and r["rounds"] == 5 for r in before
    )
    assert len(after) == 6 and all(
        r["samples"] == 500 and r["rounds"] == 5 for r in after
    )
    assert len(matrix) == 36 and all(
        r["samples"] == 500 and r["rounds"] == 5 for r in matrix
    )
    old = {r["mode"]: r for r in before}
    new = {r["mode"]: r for r in after}
    delta = (old["compile"]["p50_ms"] / old["optimized"]["p50_ms"] - 1) * 100
    saving = (1 - new["optimized"]["p50_ms"] / new["eager_before"]["p50_ms"]) * 100
    paired = []
    for case in sorted({r["case"] for r in matrix}):
        m = {r["mode"]: r for r in matrix if r["case"] == case}
        speed = m["eager_before"]["p50_ms"] / m["optimized"]["p50_ms"]
        paired.append(
            dict(
                case=case,
                before_ms=m["eager_before"]["p50_ms"],
                after_ms=m["optimized"]["p50_ms"],
                speedup=speed,
                latency_reduction_percent=(1 - 1 / speed) * 100,
            )
        )
    regressions = read("fusion-regressions.json")
    assert len(regressions) == 4 and all(r["returncode"] == 0 for r in regressions)
    state = read("fusion-state.json/results.json")
    state_checks = [v for r in state["checks"] for v in r["accuracy"].values()]
    assert len(state["checks"]) == 12 and all(v["passed"] for v in state_checks)
    compile_graph = aggregate(read("fusion-b128-compile-graph/results.json"))
    assert len(compile_graph) == 1 and compile_graph[0]["samples"] == 5
    assert compile_graph[0]["stats"]["compile_failures"] == 0
    assert compile_graph[0]["stats"]["graph_failures"] == 0
    unit_log = (a.artifacts / "fusion-regression-unit.log").read_text()
    assert "Ran 16 tests" in unit_log and unit_log.rstrip().endswith("OK")
    validation = dict(
        unit_tests=16,
        formal_mode_round_checks=sum(r["rounds"] for r in before + after + matrix),
        formal_max_abs=max(r["max_abs"] for r in before + after + matrix),
        state_checks=len(state["checks"]),
        state_max_abs=max(v["max_abs"] for v in state_checks),
        state_stats=state["stats"],
        compile_graph_smoke=compile_graph[0],
    )
    summary = dict(
        before=before,
        after=after,
        matrix=paired,
        matrix_details=matrix,
        validation=validation,
        old_compile_latency_increase_percent=delta,
        eager_latency_reduction_percent=saving,
        matrix_geomean_speedup=float(
            np.exp(np.mean(np.log([r["speedup"] for r in paired])))
        ),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    with a.output.with_suffix(".csv").open("w") as f:
        fields = [
            "phase",
            "case",
            "mode",
            "p50_ms",
            "p95_ms",
            "gpu_p50_ms",
            "cold_ms",
            "peak_allocated_gib",
            "peak_reserved_gib",
            "max_abs",
            "max_relative_l2",
            "min_cosine",
            "samples",
            "rounds",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for phase, rows in [("before", before), ("after", after), ("matrix", matrix)]:
            for row in rows:
                w.writerow(dict(phase=phase, **row))
    lines = [
        "# Batch 128：compile 对比及 eager 融合报告",
        "",
        f"修改前 compile 延迟比 optimized eager 增加 {delta:.2f}%；接入有效改动后，eager 相对同轮旧 eager 延迟降低 {saving:.2f}%。",
        "",
        "测试条件：单张 L20D / GPU 0，128 张 448×448 独立图片，共 100352 patch tokens。加载 Qwen3.5-397B-A17B-FP8 的全部 333 个真实视觉权重张量，实际均为 BF16；SDPA/cuDNN 后端，不加载语言模型。随机种子 20260910。",
        "每个配置每轮预热 20 次，测量 100 次，交替顺序重复 5 轮。主指标为包含元数据准备、图输入复制和输出处理的墙钟延迟；CUDA Events 同时计时。模型加载和预处理不计入该延迟。",
        "",
        "## 修改前的正式对比",
        "",
        "| 模式 | p50 ms | p95 ms | CUDA p50 ms | 首次调用 s |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in before:
        lines.append(
            f"| {r['mode']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['gpu_p50_ms']:.3f} | {r['cold_ms']/1000:.3f} |"
        )
    lines += [
        "",
        "此处 optimized 为移植前 eager。原始源码和补丁保存在 artifacts/pre-eager-fusion；编译 debug 与生成的 output_code.py 保留在 torch_compile_debug。",
        "Profiler 定位：eager 每层一次 cat，共 27 次，GPU 约 1.9 ms；Inductor 把 cat/transpose 降低为每张图片一次复制，共 3456 次，GPU 约 8.2 ms。attention 本身仍为 3456 次。这项自动融合造成退化，没有移植。",
        "",
        "## 逐项消融与最终模式",
        "",
        "| 模式 | p50 ms | p95 ms | CUDA p50 ms | 图片/秒 | 首次调用 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in after:
        lines.append(
            f"| {r['mode']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['gpu_p50_ms']:.3f} | {128000/r['p50_ms']:.1f} | {r['cold_ms']/1000:.3f} |"
        )
    lines += [
        "",
        "eager_before：关闭两项新增优化；eager_patch：仅 bias/位置融合；eager_batch：仅等长度 attention 合批；optimized：两项同时开启，仍不调用 torch.compile 或 CUDA Graph。compile/cudagraph 为接入改动后的显式模式。",
        "",
        "移植的 Triton 算子融合 Conv3d 输出的 bias 与位置编码相加，显式保留 bias 相加后的 BF16 舍入。另行实现的 SDPA 合批来自 profiler 分析，并非 Inductor 自动生成：每层 128 次 attention 调用合并为 1 次，整个 encoder 3456 次降为 27 次；图片/帧仍使用独立 batch 维度。",
        "LayerNorm、GELU 继续保留原生精度边界。仅当所有帧/图片段长度相同时合批；不同长度保留逐段实现，完整有序 grid 仍用于位置编码和 CUDA Graph key。",
        "",
        "## 输入矩阵回归",
        "",
        "| 场景 | 旧 eager ms | 新 eager ms | 延迟降低 |",
        "|---|---:|---:|---:|",
    ]
    for r in paired:
        lines.append(
            f"| {r['case']} | {r['before_ms']:.3f} | {r['after_ms']:.3f} | {r['latency_reduction_percent']:.2f}% |"
        )
    lines += [
        "",
        f"矩阵几何平均加速比：{summary['matrix_geomean_speedup']:.4f}×。各项数值误差、逐轮中位数、编译/图缓存统计见同名 JSON；原始逐次数据见 results.json/CSV。",
        "",
        "显存统计是同一比较进程的峰值，包含两个模型和缓存，不能作为单独部署显存或 CUDA Graph 增量显存。Profiler 开销不用于正式延迟比较。编译冷启动受磁盘缓存和 debug 配置影响；不是服务启动时间承诺。",
        "",
        "复现命令及开关见 github-opensource/benchmark/README_qwen35_vision.md 的 Eager fusion follow-up。原始数据目录：artifacts/b128-eager-compile、artifacts/b128-eager-fusions、artifacts/fusion-matrix。",
        "微基准与额外回归：artifacts/fusion_candidate_results.json、artifacts/sdpa_batch_results.json、artifacts/fusion-regressions.json 及对应日志。",
    ]
    lines += [
        "",
        "## 正确性与执行建议",
        "",
        f"16 项 Python/kernel 回归全部通过。修改前、消融和 18 场景矩阵共 {validation['formal_mode_round_checks']} 个模式/轮次正确性检查均通过，last_hidden_state 与 pooler_output 最大绝对误差均为 {validation['formal_max_abs']}。数值阈值保持原定标准。",
        "真实权重下另通过 12 项图状态检查：9 个完整 grid key、8 图容量、容量溢出回退、同形状不同内容回放及历史输出不被覆盖。记录见 artifacts/fusion-state.json/results.json（fusion-state.json 是输出目录）。",
        "batch128 的 compile_cudagraph 补充正确性冒烟测试也通过：仅 1 轮、2 次预热、5 次测量，不纳入五轮性能结论。记录见 artifacts/fusion-b128-compile-graph/results.json。",
        "",
        "本次 batch128×448 建议直接使用 optimized eager。新 compile 与新 eager 仅相差约 0.18%，CUDA Graph 与新 eager 基本持平，没有足够实测收益支持默认启用它们。主要收益来自 SDPA 等长度合批；bias/位置融合单独仅节省约 0.26 ms。其他分辨率的 batch128 及混合分辨率 batch128 未做此次性能测量。",
        "18 场景矩阵相对移植前 eager 的几何平均加速约 1.093×，最大场景延迟增加 1.38%，没有超过 5% 的场景退化。这是已有 optimized 路径之上的增量收益，不是相对最初 pristine encoder 的总收益。",
        "",
        "源码补丁及 SHA256 清单：artifacts/qwen35-vision.patch、artifacts/source-sha256.json；移植前快照和补丁：artifacts/pre-eager-fusion/。",
    ]
    a.output.write_text(chr(10).join(lines) + chr(10))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
