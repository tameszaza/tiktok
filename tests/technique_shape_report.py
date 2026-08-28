#!/usr/bin/env python3
"""Measure all technique combinations across representative model shapes."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiktok.tests.matrix_report.technique_shape import (  # noqa: E402
    TECHNIQUES,
    build_shape_cases,
    run_technique_shape_sweep,
)
from tiktok.tests.matrix_report.technique_shape_plot import (  # noqa: E402
    write_technique_shape_plot,
)


def write_outputs(output_dir: Path, results, accuracy_trials: int, warmup: int, repeats: int, rounds: int) -> None:
    csv_path = output_dir / "technique_shape_results.csv"
    image_path = output_dir / "technique_shape.png"
    report_path = output_dir / "TECHNIQUE_SHAPE_REPORT.md"
    fields = ["shape_group", "shape", "setup", "technique", "accuracy", "failed", "checked", "baseline_ms", "optimized_ms", "speedup"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "shape_group": result.shape.group, "shape": result.shape.label, "setup": result.shape.setup,
                "technique": result.technique, "accuracy": result.accuracy, "failed": result.failed,
                "checked": result.checked, "baseline_ms": f"{result.baseline_ms:.4f}",
                "optimized_ms": f"{result.optimized_ms:.4f}", "speedup": f"{result.speedup:.3f}",
            })
    write_technique_shape_plot(image_path, results)
    lines = [
        f"# Technique × Model-Shape Experiment ({len(results)} Configurations)", "",
        f"Generated: {date.today().isoformat()} by `technique_shape_report.py`.", "",
        f"This is a full factorial experiment with {len(build_shape_cases())} model shapes and {len(TECHNIQUES)} technique combinations ({len(build_shape_cases()) * len(TECHNIQUES)} rows). Bits in each combination are ordered QKV / SDPA / Triton LN / In-place / Shape-specialized LN. `00000` is a second true BaselineTransformer control. The modular ablation model toggles only the named technique while keeping the lab.py equations, weights, inputs, and eager FP32 baseline fixed; it does not modify the supplied benchmark outside UserOptimizedTransformer. Compilation startup is not involved. Each technique is timed beside its baseline with alternating order to cancel GPU clock/thermal drift. Identical configurations repeated by different ablation groups reuse one timing measurement (for example B8/S128 = D512/H8 = L6 = F2048), so labels cannot disagree because of run order. Timing uses {accuracy_trials} accuracy trial(s), {warmup} warm-up calls, {repeats} CUDA-event repeats, and {rounds} benchmark round(s).", "",
        "![Technique versus shape speedup](technique_shape.png)", "",
        "Speedup is `baseline median latency / optimized median latency`. Accuracy uses lab.py's unchanged criterion: absolute error ≤ 0.001 OR relative error ≤ 1%.", "",
        "The sweep and the standalone `lab.py` result answer different questions: this table keeps every ablation eager so individual techniques can be compared, while the main result may enable `--compile-user`. GPU clocks and power state can change the absolute milliseconds; compare the baseline/optimized ratio within a row, not an absolute millisecond value from a different run.", "",
        "## Duplicate-shape control", "",
    ]
    # Several factorial axes intentionally spell the exact same workload. A
    # consistency table makes this explicit and catches stale reports where a
    # later axis was measured at a different GPU clock.
    aliases = {}
    for result in results:
        key = (result.shape.config, result.shape.padding_ratio)
        aliases.setdefault(key, []).append(result)
    lines += [
        "The following labels are aliases for one identical model configuration; their timing rows are reused, not re-measured:", "",
        "| Labels | Baseline spread (ms) | Optimized spread (ms) |", "| --- | ---: | ---: |",
    ]
    alias_count = 0
    for grouped in aliases.values():
        labels = list(dict.fromkeys(result.shape.label for result in grouped))
        if len(labels) < 2:
            continue
        alias_count += 1
        by_technique = {}
        for result in grouped:
            by_technique.setdefault(result.technique, []).append(result)
        baseline_spread = max(max(r.baseline_ms for r in values) - min(r.baseline_ms for r in values) for values in by_technique.values())
        optimized_spread = max(max(r.optimized_ms for r in values) - min(r.optimized_ms for r in values) for values in by_technique.values())
        lines.append(f"| {' / '.join(labels)} | {baseline_spread:.4f} | {optimized_spread:.4f} |")
    if alias_count == 0:
        lines.append("| (none) | 0.0000 | 0.0000 |")
    lines += ["", "## Technique-combination summary", "",
              "| Bits (QKV/SDPA/LN/in-place/shape-LN) | Mean speedup | Median speedup | Best shape | Best speedup | Accuracy failures |", "| --- | ---: | ---: | --- | ---: | ---: |"]
    for technique, _ in TECHNIQUES:
        rows = [result for result in results if result.technique == technique]
        best = max(rows, key=lambda result: result.speedup)
        mean = sum(result.speedup for result in rows) / len(rows)
        ordered = sorted(result.speedup for result in rows)
        median = ordered[len(ordered) // 2]
        failures = sum(result.failed for result in rows)
        lines.append(f"| {technique} | {mean:.3f}× | {median:.3f}× | {best.shape.label} | {best.speedup:.3f}× | {failures} |")
    combo_means = {
        technique: sum(result.speedup for result in results if result.technique == technique) /
        len([result for result in results if result.technique == technique])
        for technique, _ in TECHNIQUES
    }
    best_combo = max((technique for technique, _ in TECHNIQUES), key=combo_means.get)
    best_row = max(results, key=lambda result: result.speedup)

    def describe(bits: str) -> str:
        enabled = [name for bit, name in zip(bits, ("QKV", "SDPA", "Triton LN", "In-place", "Shape-specialized LN")) if bit == "1"]
        return "none (baseline control)" if not enabled else " + ".join(enabled)

    lines += ["", "## Interpretation", "",
              f"- Best average combination: **{best_combo} ({describe(best_combo)})**, mean {combo_means[best_combo]:.3f}× across all shapes.",
              f"- Best individual measurement: **{best_row.technique} ({describe(best_row.technique)})** on {best_row.shape.label} at {best_row.speedup:.3f}×.",
              "- The Triton-LN-only (`00100`) and shape-specialized-LN-only (`00001`) rows stay near 1× on average; their launch cost is visible when the FFN dominates, so the fused path should be selected by measured shape.",
              "- QKV or SDPA by themselves are close to 1× on many shapes; their launch and layout overhead can outweigh the saved work for small or unfavorable GEMM sizes.",
              "- Choose the combination by shape rather than assuming `11111` always wins. The heat map and CSV expose the per-shape winner."]
    lines += ["", f"## All {len(results)} measurements", "", "| Group | Shape | Setup | Technique | Accuracy | Baseline ms | Optimized ms | Speedup |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    for result in results:
        lines.append(f"| {result.shape.group} | {result.shape.label} | {result.shape.setup} | {result.technique} | {result.accuracy} ({result.failed}/{result.checked}) | {result.baseline_ms:.4f} | {result.optimized_ms:.4f} | **{result.speedup:.3f}×** |")
    lines += ["", "Raw data: `technique_shape_results.csv`."]
    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {image_path}")
    print(f"wrote {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    # The sweep is intended to compare techniques, not to produce a fast but
    # noisy screening number.  One short CUDA-event batch can be distorted by
    # clock ramping or a competing process (the old defaults produced a false
    # ~2x outlier). These defaults match the stable standalone benchmark;
    # callers can still lower them explicitly for a quick exploratory pass.
    parser.add_argument("--accuracy-trials", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    args = parser.parse_args()
    results = run_technique_shape_sweep(build_shape_cases(), args.accuracy_trials, args.warmup, args.repeats, args.benchmark_rounds)
    expected = len(build_shape_cases()) * len(TECHNIQUES)
    if len(results) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(results)}")
    write_outputs(args.output_dir, results, args.accuracy_trials, args.warmup, args.repeats, args.benchmark_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
