"""Full-factorial ablation runner kept outside the supplied benchmark file."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import product
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tiktok.lab import (
        BaselineTransformer, BaselineTransformerBlock, TransformerConfig,
        benchmark_once, compare_outputs, copy_model_weights,
        generate_random_case, warmup_model,
    )
except ModuleNotFoundError:
    from lab import (  # type: ignore[no-redef]
        BaselineTransformer, BaselineTransformerBlock, TransformerConfig,
        benchmark_once, compare_outputs, copy_model_weights,
        generate_random_case, warmup_model,
    )


@dataclass(frozen=True)
class AblationOptions:
    fused_qkv: bool
    fused_sdpa: bool
    fused_layer_norm: bool
    inplace_residual: bool
    shape_specialized_ln: bool = False


class ConfigurableOptimizedTransformer(BaselineTransformer):
    """External ablation model; the supplied lab.py remains unchanged."""

    def __init__(self, config: TransformerConfig, options: AblationOptions) -> None:
        super().__init__(config)
        self.options = options
        self.register_buffer("_packed_qkv_weight", torch.empty(0), persistent=False)
        self.register_buffer("_packed_qkv_bias", torch.empty(0), persistent=False)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        weights, biases = [], []
        for layer in self.layers:
            weights.append(torch.cat([layer.attention.q_proj.weight, layer.attention.k_proj.weight, layer.attention.v_proj.weight], dim=0))
            biases.append(torch.cat([layer.attention.q_proj.bias, layer.attention.k_proj.bias, layer.attention.v_proj.bias], dim=0))
        self._packed_qkv_weight = torch.stack(weights, dim=0)
        self._packed_qkv_bias = torch.stack(biases, dim=0)
        return result

    def _attention(self, layer: BaselineTransformerBlock, index: int, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        heads, head_dim = layer.attention.num_heads, layer.attention.head_dim
        strict = x.dtype in (torch.float16, torch.bfloat16) or (self.config.causal and mask is not None)
        if self.options.fused_qkv and not strict:
            qkv = F.linear(x, self._packed_qkv_weight[index], self._packed_qkv_bias[index]).view(batch, seq_len, 3, heads, head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        else:
            q = layer.attention._split_heads(layer.attention.q_proj(x))
            k = layer.attention._split_heads(layer.attention.k_proj(x))
            v = layer.attention._split_heads(layer.attention.v_proj(x))
        if self.options.fused_sdpa and not strict:
            sdpa_mask = None if mask is None else mask[:, None, None, :]
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=self.config.causal if sdpa_mask is None else False)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
            if self.config.causal:
                upper = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(upper, float("-inf"))
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            context = torch.matmul(torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype), v)
        output = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = layer.attention.out_proj(output)
        if mask is not None:
            output = output.masked_fill(~mask[..., None], 0)
        return output

    def _boundary(self, residual: torch.Tensor, update: torch.Tensor, norm: nn.LayerNorm, mask: Optional[torch.Tensor], zero_output: bool, inplace: bool) -> tuple[torch.Tensor, torch.Tensor]:
        use_fused_ln = self.options.fused_layer_norm and not (
            self.options.shape_specialized_ln and residual.numel() < 131072
        ) and residual.numel() // residual.shape[-1] <= 65536
        if use_fused_ln:
            try:
                from transformer_kernels import fused_residual_layer_norm, fused_residual_layer_norm_inplace
            except ModuleNotFoundError:
                from tiktok.transformer_kernels import fused_residual_layer_norm, fused_residual_layer_norm_inplace
            if inplace:
                return residual, fused_residual_layer_norm_inplace(residual, update, norm.weight, norm.bias, mask, norm.eps, zero_output)
            return fused_residual_layer_norm(residual, update, norm.weight, norm.bias, mask, norm.eps, zero_output)
        values = residual + update
        if mask is not None:
            values = values.masked_fill(~mask[..., None], 0)
        normalized = F.layer_norm(values, (values.shape[-1],), norm.weight, norm.bias, norm.eps)
        if zero_output and mask is not None:
            normalized = normalized.masked_fill(~mask[..., None], 0)
        if inplace:
            residual.copy_(values)
            return residual, normalized
        return values, normalized

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.device.type != "cuda" or torch.is_grad_enabled() or x.dtype != torch.float32:
            return super().forward(x, mask)
        normalized = self.layers[0].norm1(x)
        for index, layer in enumerate(self.layers):
            attention = self._attention(layer, index, normalized, mask)
            x, normalized_for_ffn = self._boundary(x, attention, layer.norm2, mask, False, index > 0 and self.options.inplace_residual)
            ffn = layer.ffn_out(F.gelu(layer.ffn_in(normalized_for_ffn), approximate="none"))
            last = index + 1 == len(self.layers)
            next_norm = self.final_norm if last else self.layers[index + 1].norm1
            x, normalized = self._boundary(x, ffn, next_norm, mask, last, index > 0 and self.options.inplace_residual)
        return normalized


@dataclass(frozen=True)
class AblationConfig:
    model: TransformerConfig
    accuracy_trials: int = 3
    warmup: int = 15
    repeats: int = 60
    benchmark_rounds: int = 2
    seed: int = 1234


@dataclass(frozen=True)
class AblationResult:
    options: AblationOptions
    accuracy: str
    failed: int
    checked: int
    baseline_ms: float
    optimized_ms: float
    speedup: float


def all_option_combinations() -> list[AblationOptions]:
    return [AblationOptions(*values) for values in product((False, True), repeat=4)]


def _accuracy(baseline, optimized, config: AblationConfig, trial: int) -> tuple[str, int, int]:
    x, mask = generate_random_case(config.model, torch.device("cuda"), torch.float32, config.seed + trial, 0.0, 1.0)
    with torch.inference_mode():
        result = compare_outputs(baseline(x, mask), optimized(x, mask), rtol=0.01, atol=0.001)
    return ("PASS" if result.passed else "FAIL", result.failed_elements, result.total_elements)


def run_ablation(config: AblationConfig) -> list[AblationResult]:
    config.model.validate()
    device = torch.device("cuda")
    x, mask = generate_random_case(config.model, device, torch.float32, config.seed + 100000, 0.0, 1.0)
    baseline = BaselineTransformer(config.model).to(device=device).eval()
    results: list[AblationResult] = []
    for options in all_option_combinations():
        optimized = ConfigurableOptimizedTransformer(config.model, options).to(device=device).eval()
        copy_model_weights(baseline, optimized)
        checks = [_accuracy(baseline, optimized, config, trial) for trial in range(config.accuracy_trials)]
        accuracy = "PASS" if all(check[0] == "PASS" for check in checks) else "FAIL"
        failed, checked = sum(check[1] for check in checks), sum(check[2] for check in checks)
        warmup_model(baseline, x, mask, config.warmup, device)
        warmup_model(optimized, x, mask, config.warmup, device)
        baseline_samples, optimized_samples = [], []
        for round_index in range(config.benchmark_rounds):
            if round_index % 2 == 0:
                baseline_samples += benchmark_once(baseline, x, mask, config.repeats, device)
                optimized_samples += benchmark_once(optimized, x, mask, config.repeats, device)
            else:
                optimized_samples += benchmark_once(optimized, x, mask, config.repeats, device)
                baseline_samples += benchmark_once(baseline, x, mask, config.repeats, device)
        baseline_ms, optimized_ms = statistics.median(baseline_samples), statistics.median(optimized_samples)
        result = AblationResult(options, accuracy, failed, checked, baseline_ms, optimized_ms, baseline_ms / optimized_ms)
        print(f"{options}: {accuracy} failed={failed}/{checked} baseline={baseline_ms:.4f} ms optimized={optimized_ms:.4f} ms speedup={result.speedup:.3f}x", flush=True)
        results.append(result)
    return results
