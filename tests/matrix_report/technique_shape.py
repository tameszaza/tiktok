"""Run a technique-by-model-shape factorial sweep outside the supplied lab.py."""

from __future__ import annotations

import statistics
import gc
from dataclasses import dataclass

import torch
import torch.nn as nn

from .ablation import AblationOptions, ConfigurableOptimizedTransformer

try:
    from tiktok.lab import (
        BaselineTransformer,
        TransformerConfig,
        benchmark_once,
        compare_outputs,
        copy_model_weights,
        generate_random_case,
        warmup_model,
    )
except ModuleNotFoundError:
    from lab import (  # type: ignore[no-redef]
        BaselineTransformer,
        TransformerConfig,
        benchmark_once,
        compare_outputs,
        copy_model_weights,
        generate_random_case,
        warmup_model,
    )


@dataclass(frozen=True)
class ShapeCase:
    group: str
    label: str
    setup: str
    config: TransformerConfig
    padding_ratio: float = 0.0


@dataclass(frozen=True)
class TechniqueResult:
    shape: ShapeCase
    technique: str
    accuracy: str
    failed: int
    checked: int
    baseline_ms: float
    optimized_ms: float
    speedup: float
    dtype: str = "float32"


TECHNIQUE_NAMES = ("QKV", "SDPA", "Triton LN", "In-place", "Shape-specialized LN")


def _all_techniques() -> tuple[tuple[str, AblationOptions | None], ...]:
    combinations: list[tuple[str, AblationOptions | None]] = []
    for mask in range(32):
        options = tuple(bool(mask & (1 << bit)) for bit in range(5))
        # ``00000`` is deliberately a second true BaselineTransformer control.
        # Build the display name in option order; Python's integer formatting
        # prints the most-significant bit first and would silently reverse the
        # documented QKV/SDPA/LN/in-place/shape-LN columns.
        name = "".join("1" if enabled else "0" for enabled in options)
        combinations.append((name, None if mask == 0 else AblationOptions(*options)))
    return tuple(sorted(combinations, key=lambda item: item[0]))


TECHNIQUES = _all_techniques()


def _shape(group: str, label: str, batch: int = 8, seq: int = 128, d_model: int = 512,
           heads: int = 8, ffn: int = 2048, layers: int = 6, causal: bool = False,
           padding_ratio: float = 0.0) -> ShapeCase:
    config = TransformerConfig(batch, seq, d_model, heads, ffn, layers, causal)
    setup = f"B={batch},S={seq},D={d_model},H={heads},FFN={ffn},L={layers}"
    if causal:
        setup += ",causal=True"
    if padding_ratio:
        setup += f",pad={padding_ratio:.2f}"
    return ShapeCase(group, label, setup, config, padding_ratio)


def build_shape_cases() -> tuple[ShapeCase, ...]:
    """Return shapes covering batch/sequence, width, depth, FFN, and masks."""
    cases: list[ShapeCase] = []
    for batch, seq in ((1, 32), (1, 64), (1, 128), (1, 256), (2, 32), (2, 64), (2, 128), (2, 256), (4, 32), (4, 64), (4, 128), (4, 256), (8, 32), (8, 128), (1, 512), (2, 512)):
        cases.append(_shape("batch×sequence", f"B{batch} S{seq}", batch=batch, seq=seq))
    for d_model, heads in ((128, 4), (256, 4), (384, 8), (512, 8), (768, 12), (1024, 16)):
        cases.append(_shape("hidden×head", f"D{d_model} H{heads}", d_model=d_model, heads=heads, ffn=4 * d_model))
    cases.append(_shape("hidden×head", "D1536 H24", batch=2, seq=64, d_model=1536, heads=24, ffn=6144, layers=2))
    cases.append(_shape("hidden×head", "D2048 H32", batch=1, seq=32, d_model=2048, heads=32, ffn=8192, layers=1))
    for layers in (1, 2, 4, 6, 8, 12):
        cases.append(_shape("layer count", f"L{layers}", layers=layers))
    for ffn in (256, 512, 1024, 2048, 4096, 8192):
        cases.append(_shape("FFN width", f"F{ffn}", ffn=ffn))
    cases.append(_shape("masking", "causal B2 S128", batch=2, seq=128, layers=2, causal=True))
    cases.append(_shape("masking", "padding B4 S64", batch=4, seq=64, layers=2, padding_ratio=0.25))
    cases.append(_shape("masking", "causal+padding B4 S64", batch=4, seq=64, layers=2, causal=True, padding_ratio=0.25))
    return tuple(cases)


def _accuracy(
    baseline,
    optimized,
    shape: ShapeCase,
    dtype: torch.dtype,
    trials: int,
    device: torch.device,
    seed: int,
) -> tuple[str, int, int]:
    failed = 0
    checked = 0
    passed = True
    for trial in range(trials):
        x, mask = generate_random_case(shape.config, device, dtype, seed + trial, shape.padding_ratio, 1.0)
        with torch.inference_mode():
            comparison = compare_outputs(baseline(x, mask), optimized(x, mask), rtol=0.01, atol=0.001)
        passed &= comparison.passed
        failed += comparison.failed_elements
        checked += comparison.total_elements
    return ("PASS" if passed else "FAIL", failed, checked)


def run_technique_shape_sweep(
    shapes: tuple[ShapeCase, ...],
    accuracy_trials: int = 2,
    warmup: int = 10,
    repeats: int = 40,
    rounds: int = 2,
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
) -> list[TechniqueResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; restore the NVIDIA driver before running the sweep")
    dtype_names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    if dtype not in dtype_names:
        raise ValueError(f"unsupported sweep dtype: {dtype}")
    dtype_name = dtype_names[dtype]
    device = torch.device("cuda")
    results: list[TechniqueResult] = []
    # The factorial axes intentionally repeat the default configuration (the
    # B8/S128, D512/H8, L6, and F2048 rows).  A duplicate shape is the same
    # workload, not a new experiment; reuse one measured result so those four
    # labels cannot disagree merely because the GPU clock changed later.
    measured_by_config: dict[tuple[object, ...], list[TechniqueResult]] = {}
    for shape_index, shape in enumerate(shapes):
        shape.config.validate()
        config_key = (
            shape.config.batch_size, shape.config.seq_len, shape.config.d_model,
            shape.config.num_heads, shape.config.ffn_dim, shape.config.num_layers,
            shape.config.causal, shape.padding_ratio,
        )
        if config_key in measured_by_config:
            for measured in measured_by_config[config_key]:
                result = TechniqueResult(
                    shape, measured.technique, measured.accuracy, measured.failed,
                    measured.checked, measured.baseline_ms, measured.optimized_ms,
                    measured.speedup, dtype_name,
                )
                print(
                    f"[{shape.group}] {shape.label} / {result.technique}: "
                    f"reused identical configuration measurement "
                    f"speedup={result.speedup:.3f}x",
                    flush=True,
                )
                results.append(result)
            continue
        x, mask = generate_random_case(shape.config, device, dtype, seed + 100000 + shape_index, shape.padding_ratio, 1.0)
        baseline = BaselineTransformer(shape.config).to(device=device, dtype=dtype).eval()
        # Build one variant at a time so a large 32-combination sweep does not
        # keep dozens of full models resident on the GPU.  The baseline is
        # deliberately timed beside each variant below.  Measuring it once at
        # the beginning of a 32-variant sweep made later rows sensitive to GPU
        # clock/thermal drift (and could create spurious 2x differences between
        # duplicate configurations such as B8/S128 and F2048).
        warmup_model(baseline, x, mask, warmup, device)

        shape_results: list[TechniqueResult] = []
        for technique_index, (name, options) in enumerate(TECHNIQUES):
            if options is None:
                variant = BaselineTransformer(shape.config).to(device=device, dtype=dtype).eval()
            else:
                variant = ConfigurableOptimizedTransformer(shape.config, options).to(device=device, dtype=dtype).eval()
            copy_model_weights(baseline, variant)
            warmup_model(variant, x, mask, warmup, device)
            accuracy, failed, checked = _accuracy(
                baseline, variant, shape, dtype, accuracy_trials, device,
                seed + shape_index * 31 + technique_index,
            )
            # Pair baseline and variant measurements, alternating order on
            # every round.  This cancels slow clock changes and makes the
            # speedup denominator local to the technique being tested.
            baseline_samples = []
            optimized_samples = []
            for round_index in range(rounds):
                if round_index % 2 == 0:
                    baseline_samples += benchmark_once(baseline, x, mask, repeats, device)
                    optimized_samples += benchmark_once(variant, x, mask, repeats, device)
                else:
                    optimized_samples += benchmark_once(variant, x, mask, repeats, device)
                    baseline_samples += benchmark_once(baseline, x, mask, repeats, device)
            baseline_ms = statistics.median(baseline_samples)
            optimized_ms = statistics.median(optimized_samples)
            result = TechniqueResult(
                shape, name, accuracy, failed, checked, baseline_ms,
                optimized_ms, baseline_ms / optimized_ms, dtype_name,
            )
            print(f"[{shape.group}] {shape.label} / {name}: {accuracy} failed={failed}/{checked} baseline={baseline_ms:.4f} ms optimized={optimized_ms:.4f} ms speedup={result.speedup:.3f}x", flush=True)
            results.append(result)
            shape_results.append(result)
            del variant
            gc.collect()
            torch.cuda.empty_cache()
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
        measured_by_config[config_key] = shape_results
    return results
