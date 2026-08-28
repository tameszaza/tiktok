"""CUDA-event profiling of individual Transformer operations.

The supplied benchmark reports only end-to-end latency.  This module isolates
the same operations used by the reference and submission paths so that a
technique is selected because it reduces a measured bottleneck, not because a
paper's headline number is assumed to transfer to this GPU and shape.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiktok.lab import (
    BaselineTransformer,
    BaselineTransformerBlock,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
)


@dataclass(frozen=True)
class OperationResult:
    shape: str
    setup: str
    operation: str
    baseline_ms: float
    optimized_ms: float
    speedup: float
    baseline_fraction: float
    optimized_fraction: float


def _cuda_median(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    """Return median per-call CUDA time after warmup.

    One event pair surrounds the repeated calls, avoiding a host synchronize
    between every operation while retaining device-side timing.  The warmup
    executes Triton compilation and allocator setup before measurements.
    """

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, seq_len, d_model = x.shape
    head_dim = d_model // heads
    return x.view(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()


def _baseline_attention_core(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)


def _profile_one(
    label: str,
    config: TransformerConfig,
    warmup: int,
    repeats: int,
    seed: int,
) -> list[OperationResult]:
    device = torch.device("cuda")
    baseline = BaselineTransformer(config).to(device=device).eval()
    optimized = UserOptimizedTransformer(config).to(device=device).eval()
    copy_model_weights(baseline, optimized)
    x, mask = generate_random_case(config, device, torch.float32, seed, 0.0, 1.0)
    del mask  # Operation isolation uses the unmasked hot path.
    layer_b: BaselineTransformerBlock = baseline.layers[0]
    layer_o: BaselineTransformerBlock = optimized.layers[0]

    norm_b = layer_b.norm1(x)
    norm_o = layer_o.norm1(x)
    heads = layer_b.attention.num_heads
    head_dim = layer_b.attention.head_dim
    q_b = _split_heads(layer_b.attention.q_proj(norm_b), heads)
    k_b = _split_heads(layer_b.attention.k_proj(norm_b), heads)
    v_b = _split_heads(layer_b.attention.v_proj(norm_b), heads)
    qkv_o = F.linear(
        norm_o,
        optimized._packed_qkv_weight[0],
        optimized._packed_qkv_bias[0],
    ).view(x.shape[0], x.shape[1], 3, heads, head_dim)
    q_o, k_o, v_o = qkv_o.unbind(dim=2)
    q_o, k_o, v_o = q_o.transpose(1, 2), k_o.transpose(1, 2), v_o.transpose(1, 2)
    attn_b = _baseline_attention_core(q_b, k_b, v_b, head_dim)
    attn_o = F.scaled_dot_product_attention(q_o, k_o, v_o, dropout_p=0.0)
    context_b = attn_b.transpose(1, 2).contiguous().view_as(x)
    context_o = attn_o.transpose(1, 2).contiguous().view_as(x)

    from transformer_kernels import (
        fused_residual_layer_norm,
        fused_residual_layer_norm_inplace,
    )

    residual = x.contiguous()
    residual_in_b = x.contiguous()
    residual_in_o = x.contiguous()
    update_b = layer_b.attention.out_proj(context_b).contiguous()
    update_o = layer_o.attention.out_proj(context_o).contiguous()

    def b_norm1():
        return layer_b.norm1(x)

    def o_norm1():
        return layer_o.norm1(x)

    def b_qkv():
        q = layer_b.attention.q_proj(norm_b)
        k = layer_b.attention.k_proj(norm_b)
        v = layer_b.attention.v_proj(norm_b)
        return q, k, v

    def o_qkv():
        return F.linear(
            norm_o,
            optimized._packed_qkv_weight[0],
            optimized._packed_qkv_bias[0],
        ).view(x.shape[0], x.shape[1], 3, heads, head_dim)

    def b_attention():
        return _baseline_attention_core(q_b, k_b, v_b, head_dim)

    def o_attention():
        return F.scaled_dot_product_attention(q_o, k_o, v_o, dropout_p=0.0)

    def b_out_projection():
        return layer_b.attention.out_proj(context_b)

    def o_out_projection():
        return layer_o.attention.out_proj(context_o)

    def b_residual_ln():
        values = residual + update_b
        return F.layer_norm(
            values, (config.d_model,), layer_b.norm2.weight, layer_b.norm2.bias, layer_b.norm2.eps
        )

    # Match the active submission's shape specialization.  The custom kernel
    # is only selected when the activation is large enough to amortize a
    # one-program-per-row launch; otherwise the model uses native LayerNorm.
    use_fused_ln = x.numel() >= 131072

    def o_residual_ln():
        if use_fused_ln:
            return fused_residual_layer_norm(
                residual, update_o, layer_o.norm2.weight, layer_o.norm2.bias,
                None, layer_o.norm2.eps, False,
            )
        values = residual + update_o
        return values, F.layer_norm(
            values, (config.d_model,), layer_o.norm2.weight, layer_o.norm2.bias, layer_o.norm2.eps
        )

    def b_residual_ln_inplace():
        values = residual_in_b + update_b
        residual_in_b.copy_(values)
        return F.layer_norm(
            residual_in_b, (config.d_model,), layer_b.norm2.weight, layer_b.norm2.bias, layer_b.norm2.eps
        )

    def o_residual_ln_inplace():
        if use_fused_ln:
            return fused_residual_layer_norm_inplace(
                residual_in_o, update_o, layer_o.norm2.weight, layer_o.norm2.bias,
                None, layer_o.norm2.eps, False,
            )
        values = residual_in_o + update_o
        residual_in_o.copy_(values)
        return F.layer_norm(
            residual_in_o, (config.d_model,), layer_o.norm2.weight, layer_o.norm2.bias, layer_o.norm2.eps
        )

    norm2_b = b_residual_ln()
    norm2_o = o_residual_ln()[1]

    def b_ffn():
        return layer_b.ffn_out(F.gelu(layer_b.ffn_in(norm2_b), approximate="none"))

    def o_ffn():
        return layer_o.ffn_out(F.gelu(layer_o.ffn_in(norm2_o), approximate="none"))

    operations = (
        ("norm1", b_norm1, o_norm1),
        ("QKV projection", b_qkv, o_qkv),
        ("attention core", b_attention, o_attention),
        ("output projection", b_out_projection, o_out_projection),
        ("residual + norm2 (out-of-place)", b_residual_ln, o_residual_ln),
        ("residual + norm2 (in-place)", b_residual_ln_inplace, o_residual_ln_inplace),
        ("FFN (linear + GELU + linear)", b_ffn, o_ffn),
    )
    baseline_times = [(_cuda_median(b, warmup, repeats)) for _, b, _ in operations]
    optimized_times = [(_cuda_median(o, warmup, repeats)) for _, _, o in operations]
    baseline_total = sum(baseline_times)
    optimized_total = sum(optimized_times)
    setup = (
        f"B={config.batch_size},S={config.seq_len},D={config.d_model},"
        f"H={config.num_heads},FFN={config.ffn_dim},L={config.num_layers}"
    )
    results = []
    for (operation, _, _), b_ms, o_ms in zip(operations, baseline_times, optimized_times):
        results.append(OperationResult(
            label, setup, operation, b_ms, o_ms, b_ms / o_ms,
            b_ms / baseline_total, o_ms / optimized_total,
        ))
    return results


def run_operation_profile(
    cases: tuple[tuple[str, TransformerConfig], ...],
    warmup: int = 10,
    repeats: int = 40,
    seed: int = 1234,
) -> list[OperationResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for operation profiling")
    results: list[OperationResult] = []
    for index, (label, config) in enumerate(cases):
        config.validate()
        results.extend(_profile_one(label, config, warmup, repeats, seed + index))
    return results
