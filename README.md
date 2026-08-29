# Transformer GPU Kernel Workshop Submission

This project accelerates the supplied PyTorch Transformer benchmark while
keeping its baseline implementation as the correctness reference.

The submission path combines:

- a custom Triton kernel for fused residual addition, token-mask zeroing, and
  LayerNorm;
- one packed QKV projection in the validated FP32 fast path;
- PyTorch CUDA scaled-dot-product attention; and
- optional `torch.compile` integration around the custom Triton operator.

## Files

- `lab.py` — supplied benchmark plus `UserOptimizedTransformer` submission
- `transformer_kernels.py` — submission-owned Triton/CUDA kernel
- `tests/technique_shape_report.py` — factorial technique × shape benchmark runner
- `tests/operation_profile_report.py` — CUDA-event timing for each major operation
- `tests/flash_attention_report.py` — correctness and latency comparison for the
  submission-owned tiled online-softmax attention candidate
- `tests/matrix_report/` — modular technique sweep, ablation helper, and plotting code
- `TECHNICAL_REPORT.md` — environment, design rationale, and measured results
- `OPERATION_PROFILE_REPORT.md` — operation bottlenecks and optimization plan
- `FLASH_ATTENTION_REPORT.md` — measured explicit-attention, SDPA, and Triton
  FlashAttention candidate results
- `PROBLEM_STATEMENT.md` — Markdown transcription of the workshop requirements
- `TECHNIQUE_SHAPE_REPORT.md` — full technique-combination versus shape measurements
- `technique_shape_results.csv` / `technique_shape.png` — per-technique data and heat map

## Setup

Use Linux with an NVIDIA CUDA GPU and a CUDA-enabled PyTorch installation.
The tested environment uses Python 3.14.4, PyTorch 2.12.1+cu130, Triton, and an
RTX 4060 Laptop GPU.

From the parent `Documents` directory, verify the existing environment:

```bash
.venv/bin/python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available())"
```

## Reproduce the main result

```bash
.venv/bin/python tiktok/lab.py \
  --device cuda --dtype float32 \
  --compile-user --compile-mode reduce-overhead \
  --accuracy-trials 2 --warmup 30 --repeats 80 --benchmark-rounds 5
```

The latest default test passed every checked output element and measured a
1.218x median-latency speedup on the test machine. Results vary with GPU power
state, temperature, driver, and PyTorch version.

For the full technique-combination versus shape experiment (39 shapes × 32
technique combinations = 1,248 rows), run:

```bash
.venv/bin/python tiktok/tests/technique_shape_report.py
```

For a faster representative sweep, select shape families and/or cap the
number of labels. The filters affect only the report runner; `lab.py` is
unchanged:

```bash
# 10 deterministic shapes (320 technique rows)
.venv/bin/python tiktok/tests/technique_shape_report.py --max-shapes 10

# All batch/sequence and masking shapes only
.venv/bin/python tiktok/tests/technique_shape_report.py --groups batch masking

# Lower-precision reports (artifacts receive a dtype suffix)
.venv/bin/python tiktok/tests/technique_shape_report.py --dtype float16
.venv/bin/python tiktok/tests/technique_shape_report.py --dtype bfloat16
```

The shape report accepts `float32`, `float16`, and `bfloat16`. Each invocation
measures one dtype; FP32 keeps the historical artifact names, while lower
precision writes `technique_shape_results_<dtype>.csv`,
`technique_shape_<dtype>.png`, and `TECHNIQUE_SHAPE_REPORT_<DTYPE>.md`.

For operation-level timing and an automatically generated bottleneck report:

```bash
.venv/bin/python tiktok/tests/operation_profile_report.py
```

To reproduce the isolated FlashAttention candidate experiment:

```bash
.venv/bin/python tiktok/tests/flash_attention_report.py --warmup 10 --repeats 50
```

Test causal attention and padding:

```bash
.venv/bin/python tiktok/lab.py \
  --device cuda --dtype float32 --causal --padding-ratio 0.25 \
  --compile-user --compile-mode reduce-overhead
```

## Correctness behavior

The standalone `lab.py` FP32 CUDA inference path uses the custom Triton fusion.
Its FP16/BF16, CPU, and gradient-enabled execution still use the strict
reference-order submission path. The matrix shape report is a separate ablation
runner: its CUDA FP16/BF16 variants exercise the selected QKV, SDPA, Triton
LayerNorm, in-place, and shape-specialized options, and record any accuracy
failures under the same per-element tolerance. No approximation or quantization
is used.

## Limitations and future work

The custom kernel currently optimizes inference only and specializes the
residual/LayerNorm boundary rather than reimplementing GEMMs. A separate
FlashAttention-style Triton kernel implements tiled online softmax exactly and
is retained as a measured candidate; on this RTX 4060, PyTorch SDPA's mature
vendor backend was faster on every tested candidate shape, so SDPA remains the
active fast-attention choice. This avoids claiming a paper speedup that the
local hardware does not reproduce. Future work is shape-specific autotuning of
the candidate's sequence tiles and warp count.

## Contributions

AI assistance was used to inspect the workload, research fused-attention and
Triton integration techniques, generate candidate kernels, diagnose numerical
differences, and run the shape/precision benchmark matrix. The participant is
responsible for reviewing, presenting, and submitting the implementation.
