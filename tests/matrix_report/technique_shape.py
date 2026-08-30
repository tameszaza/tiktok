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
    """Return the 14 configurations listed in the workshop appendix (Fig. 3.7).

    These are deliberately kept as explicit records instead of generating a
    Cartesian product: the appendix varies one parameter at a time and also
    contains two stress cases (B=10000 and S=100000) that may exceed a local
    GPU's memory capacity.
    """
    group = "appendix test shape"
    return (
        _shape(group, "#1 B64 D128 H4 S128 L4", batch=64, seq=128, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#2 B1 D128 H4 S128 L4", batch=1, seq=128, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#3 B4 D128 H4 S128 L4", batch=4, seq=128, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#4 B16 D128 H4 S128 L4", batch=16, seq=128, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#5 B128 D128 H4 S128 L4", batch=128, seq=128, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#6 B10000 D128 H4 S128 L4", batch=10000, seq=128, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#7 B64 D32 H4 S128 L4", batch=64, seq=128, d_model=32, heads=4, ffn=32, layers=4, causal=True),
        _shape(group, "#8 B64 D1024 H4 S128 L4", batch=64, seq=128, d_model=1024, heads=4, ffn=1024, layers=4, causal=True),
        _shape(group, "#9 B64 D128 H1 S128 L4", batch=64, seq=128, d_model=128, heads=1, ffn=128, layers=4, causal=True),
        _shape(group, "#10 B64 D128 H2 S128 L4", batch=64, seq=128, d_model=128, heads=2, ffn=128, layers=4, causal=True),
        _shape(group, "#11 B64 D128 H16 S128 L4", batch=64, seq=128, d_model=128, heads=16, ffn=128, layers=4, causal=True),
        _shape(group, "#12 B64 D128 H4 S32 L4", batch=64, seq=32, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#13 B64 D128 H4 S1024 L4", batch=64, seq=1024, d_model=128, heads=4, ffn=128, layers=4, causal=True),
        _shape(group, "#14 B32 D1024 H16 S100000 L2", batch=32, seq=100000, d_model=1024, heads=16, ffn=1024, layers=2, causal=True),
    )


def _accuracy(
    baseline,
    optimized,
    shape: ShapeCase,
    trials: int,
    device: torch.device,
    seed: int,
    batch_limit: int | None = None,
) -> tuple[str, int, int]:
    failed = 0
    checked = 0
    passed = True
    config = shape.config
    if batch_limit is not None:
        config = TransformerConfig(
            min(config.batch_size, batch_limit), config.seq_len, config.d_model,
            config.num_heads, config.ffn_dim, config.num_layers, config.causal,
        )
    for trial in range(trials):
        x, mask = generate_random_case(config, device, torch.float32, seed + trial, shape.padding_ratio, 1.0)
        # The appendix contains no padding dimension. An all-true key mask is
        # mathematically redundant, but passing it with causal=True prevents
        # PyTorch from selecting its fused memory-efficient attention backend.
        if shape.padding_ratio == 0.0:
            mask = None
        with torch.inference_mode():
            comparison = compare_outputs(baseline(x, mask), optimized(x, mask), rtol=0.01, atol=0.001)
        passed &= comparison.passed
        failed += comparison.failed_elements
        checked += comparison.total_elements
    return ("PASS" if passed else "FAIL", failed, checked)


def _input_oom_reason(shape: ShapeCase, device: torch.device) -> str | None:
    """Return a reason only when the input itself cannot fit this GPU."""
    free_bytes, _ = torch.cuda.mem_get_info(device)
    activation_bytes = (
        shape.config.batch_size * shape.config.seq_len * shape.config.d_model * 4
    )
    # Do not reject a shape merely because the explicit baseline's SxS matrix
    # is too large: the whole point of fused attention is to avoid that matrix.
    # Skip only when even the input activation cannot coexist with basic model
    # state. The baseline is attempted separately and may legitimately OOM.
    if activation_bytes > 0.70 * free_bytes:
        activation_gib = activation_bytes / (1024 ** 3)
        free_gib = free_bytes / (1024 ** 3)
        return f"OOM (input activation {activation_gib:.2f} GiB exceeds 70% of {free_gib:.2f} GiB free GPU memory)"
    return None


def _recover_from_oom() -> None:
    """Release failed temporary allocations so another technique can run."""
    gc.collect()
    torch.cuda.empty_cache()


def run_technique_shape_sweep(
    shapes: tuple[ShapeCase, ...],
    accuracy_trials: int = 2,
    warmup: int = 10,
    repeats: int = 40,
    rounds: int = 2,
    seed: int = 1234,
) -> list[TechniqueResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; restore the NVIDIA driver before running the sweep")
    device = torch.device("cuda")
    results: list[TechniqueResult] = []
    # The appendix list is explicit and contains no generated Cartesian-product
    # aliases. Keep the cache for safety if a future appendix revision adds a
    # duplicate configuration.
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
                    measured.speedup,
                )
                print(
                    f"[{shape.group}] {shape.label} / {result.technique}: "
                    f"reused identical configuration measurement "
                    f"speedup={result.speedup:.3f}x",
                    flush=True,
                )
                results.append(result)
            continue
        input_oom = _input_oom_reason(shape, device)
        if input_oom is not None:
            print(f"[{shape.group}] {shape.label}: {input_oom}", flush=True)
            unavailable = [
                TechniqueResult(shape, name, input_oom, 0, 0, float("nan"), float("nan"), float("nan"))
                for name, _ in TECHNIQUES
            ]
            results.extend(unavailable)
            measured_by_config[config_key] = unavailable
            continue
        try:
            x, mask = generate_random_case(shape.config, device, torch.float32, seed + 100000 + shape_index, shape.padding_ratio, 1.0)
        except torch.OutOfMemoryError:
            _recover_from_oom()
            reason = "OOM (input allocation failed; baseline and optimized paths not run)"
            unavailable = [
                TechniqueResult(shape, name, reason, 0, 0, float("nan"), float("nan"), float("nan"))
                for name, _ in TECHNIQUES
            ]
            results.extend(unavailable)
            measured_by_config[config_key] = unavailable
            continue
        if shape.padding_ratio == 0.0:
            mask = None
        baseline = BaselineTransformer(shape.config).to(device=device).eval()
        # Build one variant at a time so a large 32-combination sweep does not
        # keep dozens of full models resident on the GPU.  The baseline is
        # deliberately timed beside each variant below.  Measuring it once at
        # the beginning of a 32-variant sweep made later rows sensitive to GPU
        # clock/thermal drift (and could create spurious 2x differences between
        # duplicate configurations such as B8/S128 and F2048).
        baseline_oom = False
        try:
            warmup_model(baseline, x, mask, max(warmup, 1), device)
        except torch.OutOfMemoryError:
            baseline_oom = True
            _recover_from_oom()
            print(
                f"[{shape.group}] {shape.label}: baseline OOM; "
                "attempting fused-attention variants",
                flush=True,
            )

        shape_results: list[TechniqueResult] = []
        for technique_index, (name, options) in enumerate(TECHNIQUES):
            if options is None:
                variant = BaselineTransformer(shape.config).to(device=device).eval()
            else:
                variant = ConfigurableOptimizedTransformer(shape.config, options).to(device=device).eval()
            copy_model_weights(baseline, variant)
            try:
                warmup_model(variant, x, mask, max(warmup, 1), device)
            except torch.OutOfMemoryError:
                _recover_from_oom()
                accuracy = "OOM (optimized full shape)"
                result = TechniqueResult(
                    shape, name, accuracy, 0, 0,
                    float("nan"), float("nan"), float("nan"),
                )
                print(
                    f"[{shape.group}] {shape.label} / {name}: "
                    "baseline=OOM optimized=OOM",
                    flush=True,
                )
                results.append(result)
                shape_results.append(result)
                del variant
                _recover_from_oom()
                continue
            validation_batch = 1 if baseline_oom else None
            accuracy, failed, checked = _accuracy(
                baseline, variant, shape, accuracy_trials, device,
                seed + shape_index * 31 + technique_index,
                batch_limit=validation_batch,
            )
            if baseline_oom:
                accuracy += " (batch=1 validation; full baseline OOM)"
            # Pair baseline and variant measurements, alternating order on
            # every round.  This cancels slow clock changes and makes the
            # speedup denominator local to the technique being tested.
            baseline_samples = []
            optimized_samples = []
            if baseline_oom:
                for _ in range(rounds):
                    optimized_samples += benchmark_once(variant, x, mask, repeats, device)
            else:
                for round_index in range(rounds):
                    if round_index % 2 == 0:
                        baseline_samples += benchmark_once(baseline, x, mask, repeats, device)
                        optimized_samples += benchmark_once(variant, x, mask, repeats, device)
                    else:
                        optimized_samples += benchmark_once(variant, x, mask, repeats, device)
                        baseline_samples += benchmark_once(baseline, x, mask, repeats, device)
            baseline_ms = statistics.median(baseline_samples) if baseline_samples else float("nan")
            optimized_ms = statistics.median(optimized_samples)
            speedup = baseline_ms / optimized_ms if baseline_samples else float("nan")
            result = TechniqueResult(shape, name, accuracy, failed, checked, baseline_ms, optimized_ms, speedup)
            if baseline_oom:
                print(f"[{shape.group}] {shape.label} / {name}: {accuracy} failed={failed}/{checked} baseline=OOM optimized={optimized_ms:.4f} ms speedup=N/A", flush=True)
            else:
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
