"""Submission-owned Triton kernels for the optimized Transformer.

Only inference forward kernels live here.  The benchmark's PyTorch reference
implementation does not import or call these functions directly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _tl_software_rsqrt(x):
    """Approximate 1/sqrt(x) using only Triton arithmetic/bit operations.

    This deliberately avoids an explicit CUDA libdevice call.  A Quake-style
    exponent seed followed by two Newton-Raphson steps is accurate enough for
    FP32 LayerNorm while keeping the experiment self-contained in ``tl``.
    The caller supplies a positive variance+epsilon value.
    """

    x = tl.maximum(x, 1.0e-12)
    bits = x.to(tl.int32, bitcast=True)
    bits = 0x5F3759DF - (bits >> 1)
    estimate = bits.to(tl.float32, bitcast=True)
    half_x = 0.5 * x
    estimate = estimate * (1.5 - half_x * estimate * estimate)
    estimate = estimate * (1.5 - half_x * estimate * estimate)
    return estimate


@triton.jit
def _residual_layer_norm_kernel(
    residual_ptr,
    update_ptr,
    weight_ptr,
    bias_ptr,
    valid_row_ptr,
    residual_out_ptr,
    normalized_out_ptr,
    row_stride,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_VALID_MASK: tl.constexpr,
    ZERO_PADDED_OUTPUT: tl.constexpr,
    USE_TL_SOFTWARE_RSQRT: tl.constexpr,
):
    """Fuse residual add, row padding, and LayerNorm in one GPU program."""

    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    column_mask = offsets < columns
    row_offsets = row * row_stride + offsets

    residual = tl.load(residual_ptr + row_offsets, mask=column_mask, other=0.0)
    update = tl.load(update_ptr + row_offsets, mask=column_mask, other=0.0)
    values = (residual + update).to(tl.float32)

    if HAS_VALID_MASK:
        row_is_valid = tl.load(valid_row_ptr + row) != 0
        values = tl.where(row_is_valid, values, 0.0)
    else:
        row_is_valid = True

    mean = tl.sum(values, axis=0) / columns
    centered = tl.where(column_mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / columns
    variance_epsilon = variance + epsilon
    if USE_TL_SOFTWARE_RSQRT:
        inverse_std = _tl_software_rsqrt(variance_epsilon)
    else:
        inverse_std = tl.rsqrt(variance_epsilon)

    weight = tl.load(weight_ptr + offsets, mask=column_mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=column_mask, other=0.0)
    normalized = centered * inverse_std * weight + bias
    if ZERO_PADDED_OUTPUT:
        normalized = tl.where(row_is_valid, normalized, 0.0)

    tl.store(residual_out_ptr + row_offsets, values, mask=column_mask)
    tl.store(normalized_out_ptr + row_offsets, normalized, mask=column_mask)


def _num_warps(block_size: int, rows: int) -> int:
    """Select warps using Triton's published LayerNorm heuristic.

    The official Triton LayerNorm tutorial maps one program to one row and
    chooses ``min(max(BLOCK_SIZE // 256, 1), 8)`` warps.  Matching that rule
    avoids oversubscribing tiny rows (where four warps only add scheduling
    overhead) while still giving wide hidden dimensions enough reduction
    parallelism.  ``rows`` is retained in the signature for API stability and
    future shape-specific tuning, but the reference heuristic depends only on
    the feature block size.
    """

    del rows
    return min(max(block_size // 256, 1), 8)


@triton_op("transformer_workshop::fused_residual_layer_norm", mutates_args={})
def fused_residual_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
    epsilon: float,
    zero_padded_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual + update, layer_norm(residual + update))``.

    Invalid token rows are zeroed before normalization.  At intermediate
    boundaries their normalized value is therefore the LayerNorm bias, matching
    the reference.  The final boundary can also zero the normalized output.
    """

    if residual.device.type != "cuda":
        raise RuntimeError("the fused residual LayerNorm kernel requires CUDA")
    if residual.shape != update.shape:
        raise ValueError("residual and update must have identical shapes")
    if residual.shape[-1] != weight.numel() or weight.shape != bias.shape:
        raise ValueError("LayerNorm weight/bias shape does not match hidden size")
    if not residual.is_contiguous() or not update.is_contiguous():
        raise ValueError("the Triton kernel requires contiguous activations")

    columns = residual.shape[-1]
    rows = residual.numel() // columns
    block_size = triton.next_power_of_2(columns)
    max_fused_columns = 65536 // residual.element_size()
    if block_size > max_fused_columns:
        raise RuntimeError(
            f"hidden dimension {columns} is too large for the fused LayerNorm kernel"
        )

    if valid_rows is not None:
        valid_rows = valid_rows.reshape(-1).contiguous()
        if valid_rows.numel() != rows:
            raise ValueError("valid-token mask does not match activation rows")
        valid_row_ptr = valid_rows
    else:
        # The pointer is never read when HAS_VALID_MASK is false.
        valid_row_ptr = residual

    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(residual)
    num_warps = _num_warps(block_size, rows)
    wrap_triton(_residual_layer_norm_kernel)[(rows,)](
        residual,
        update,
        weight,
        bias,
        valid_row_ptr,
        residual_out,
        normalized_out,
        residual.stride(-2),
        columns=columns,
        epsilon=epsilon,
        BLOCK_SIZE=block_size,
        HAS_VALID_MASK=valid_rows is not None,
        ZERO_PADDED_OUTPUT=zero_padded_output,
        USE_TL_SOFTWARE_RSQRT=False,
        num_warps=num_warps,
    )
    return residual_out, normalized_out


@triton_op(
    "transformer_workshop::fused_residual_layer_norm_inplace",
    mutates_args={"residual"},
)
def fused_residual_layer_norm_inplace(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
    epsilon: float,
    zero_padded_output: bool,
) -> torch.Tensor:
    """In-place residual update plus LayerNorm for internal inference buffers."""

    if residual.device.type != "cuda":
        raise RuntimeError("the fused residual LayerNorm kernel requires CUDA")
    if residual.shape != update.shape or not residual.is_contiguous():
        raise ValueError("in-place residual/update tensors must match and be contiguous")

    columns = residual.shape[-1]
    rows = residual.numel() // columns
    block_size = triton.next_power_of_2(columns)
    max_fused_columns = 65536 // residual.element_size()
    if block_size > max_fused_columns:
        raise RuntimeError(
            f"hidden dimension {columns} is too large for the fused LayerNorm kernel"
        )

    if valid_rows is not None:
        valid_rows = valid_rows.reshape(-1).contiguous()
        valid_row_ptr = valid_rows
    else:
        valid_row_ptr = residual

    normalized_out = torch.empty_like(residual)
    wrap_triton(_residual_layer_norm_kernel)[(rows,)](
        residual,
        update,
        weight,
        bias,
        valid_row_ptr,
        residual,  # Safe after the load: one program owns each complete row.
        normalized_out,
        residual.stride(-2),
        columns=columns,
        epsilon=epsilon,
        BLOCK_SIZE=block_size,
        HAS_VALID_MASK=valid_rows is not None,
        ZERO_PADDED_OUTPUT=zero_padded_output,
        USE_TL_SOFTWARE_RSQRT=False,
        num_warps=_num_warps(block_size, rows),
    )
    return normalized_out


def fused_residual_layer_norm_tl(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
    epsilon: float,
    zero_padded_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Experimental all-``tl`` LayerNorm variant.

    The public optimized path deliberately uses the tuned ``tl.rsqrt``
    kernel above.  This candidate is kept as a directly callable benchmark
    target so the software reciprocal-square-root trade-off can be measured
    without changing the workshop operator schema.
    """

    if residual.device.type != "cuda":
        raise RuntimeError("the fused residual LayerNorm kernel requires CUDA")
    if residual.shape != update.shape:
        raise ValueError("residual and update must have identical shapes")
    if residual.shape[-1] != weight.numel() or weight.shape != bias.shape:
        raise ValueError("LayerNorm weight/bias shape does not match hidden size")
    if not residual.is_contiguous() or not update.is_contiguous():
        raise ValueError("the Triton kernel requires contiguous activations")

    columns = residual.shape[-1]
    rows = residual.numel() // columns
    block_size = triton.next_power_of_2(columns)
    max_fused_columns = 65536 // residual.element_size()
    if block_size > max_fused_columns:
        raise RuntimeError(
            f"hidden dimension {columns} is too large for the fused LayerNorm kernel"
        )
    if valid_rows is not None:
        valid_rows = valid_rows.reshape(-1).contiguous()
        if valid_rows.numel() != rows:
            raise ValueError("valid-token mask does not match activation rows")
        valid_row_ptr = valid_rows
    else:
        valid_row_ptr = residual
    residual_out = torch.empty_like(residual)
    normalized_out = torch.empty_like(residual)
    wrap_triton(_residual_layer_norm_kernel)[(rows,)](
        residual,
        update,
        weight,
        bias,
        valid_row_ptr,
        residual_out,
        normalized_out,
        residual.stride(-2),
        columns=columns,
        epsilon=epsilon,
        BLOCK_SIZE=block_size,
        HAS_VALID_MASK=valid_rows is not None,
        ZERO_PADDED_OUTPUT=zero_padded_output,
        USE_TL_SOFTWARE_RSQRT=True,
        num_warps=_num_warps(block_size, rows),
    )
    return residual_out, normalized_out


def fused_residual_layer_norm_inplace_tl(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
    epsilon: float,
    zero_padded_output: bool,
) -> torch.Tensor:
    """In-place counterpart of :func:`fused_residual_layer_norm_tl`."""

    if residual.device.type != "cuda":
        raise RuntimeError("the fused residual LayerNorm kernel requires CUDA")
    if residual.shape != update.shape or not residual.is_contiguous():
        raise ValueError("in-place residual/update tensors must match and be contiguous")
    columns = residual.shape[-1]
    rows = residual.numel() // columns
    block_size = triton.next_power_of_2(columns)
    max_fused_columns = 65536 // residual.element_size()
    if block_size > max_fused_columns:
        raise RuntimeError(
            f"hidden dimension {columns} is too large for the fused LayerNorm kernel"
        )
    if valid_rows is not None:
        valid_rows = valid_rows.reshape(-1).contiguous()
        if valid_rows.numel() != rows:
            raise ValueError("valid-token mask does not match activation rows")
        valid_row_ptr = valid_rows
    else:
        valid_row_ptr = residual
    normalized_out = torch.empty_like(residual)
    wrap_triton(_residual_layer_norm_kernel)[(rows,)](
        residual,
        update,
        weight,
        bias,
        valid_row_ptr,
        residual,
        normalized_out,
        residual.stride(-2),
        columns=columns,
        epsilon=epsilon,
        BLOCK_SIZE=block_size,
        HAS_VALID_MASK=valid_rows is not None,
        ZERO_PADDED_OUTPUT=zero_padded_output,
        USE_TL_SOFTWARE_RSQRT=True,
        num_warps=_num_warps(block_size, rows),
    )
    return normalized_out


@triton.jit
def _flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    seq_len,
    scale,
    head_dim: tl.constexpr,
    heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Tiled exact attention with online softmax accumulation.

    This is the algorithmic core described by FlashAttention: K/V tiles are
    streamed through SRAM-sized blocks and each query row keeps only its
    running max, normalizer, and output accumulator.  The score matrix is
    never materialized in global memory, so the extra memory is O(sequence
    length) rather than O(sequence length squared).
    """

    query_block = tl.program_id(axis=0)
    batch_head = tl.program_id(axis=1)
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)

    batch = batch_head // heads
    head = batch_head % heads
    q_base = batch * stride_qb + head * stride_qh
    k_base = batch * stride_kb + head * stride_kh
    v_base = batch * stride_vb + head * stride_vh
    o_base = batch * stride_ob + head * stride_oh

    q_ptrs = q_ptr + q_base + query_offsets[:, None] * stride_qm + dim_offsets[None, :] * stride_qd
    q_mask = (query_offsets[:, None] < seq_len) & (dim_offsets[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for key_start in tl.range(0, seq_len, BLOCK_N):
        keys = key_start + key_offsets
        k_ptrs = k_ptr + k_base + keys[:, None] * stride_km + dim_offsets[None, :] * stride_kd
        v_ptrs = v_ptr + v_base + keys[:, None] * stride_vm + dim_offsets[None, :] * stride_vd
        kv_mask = (keys[:, None] < seq_len) & (dim_offsets[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        valid_queries = query_offsets[:, None] < seq_len
        valid_keys = keys[None, :] < seq_len
        if CAUSAL:
            valid_keys = valid_keys & (keys[None, :] <= query_offsets[:, None])
        valid = valid_queries & valid_keys
        # Invalid query rows are kept finite to avoid -inf - -inf during the
        # online update; their stores are masked below.  Invalid key columns
        # still receive the mathematically correct -inf score.
        qk = tl.where(valid_keys, qk, -float("inf"))
        qk = tl.where(valid_queries, qk, 0.0)
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.where(m_new == -float("inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(valid, tl.exp(qk - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        accumulator = accumulator * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new

    # Padded query rows have no valid keys and therefore l_i == 0; use a
    # finite denominator even though those rows are masked on store.
    safe_l = tl.where(l_i > 0, l_i, 1.0)
    output = accumulator / safe_l[:, None]
    output = tl.where(query_offsets[:, None] < seq_len, output, 0.0)
    output_ptrs = output_ptr + o_base + query_offsets[:, None] * stride_om + dim_offsets[None, :] * stride_od
    tl.store(output_ptrs, output, mask=q_mask)


@triton_op("transformer_workshop::flash_attention", mutates_args={})
def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Exact tiled attention using online softmax (FlashAttention algorithm).

    The kernel intentionally accepts FP32 only.  The benchmark's FP16/BF16
    path remains on the supplied reference ordering because its correctness
    contract compares reduced-precision outputs.  ``query``, ``key`` and
    ``value`` must be contiguous ``[batch, heads, sequence, head_dim]``
    tensors, with head dimensions up to 128.
    """

    if query.device.type != "cuda":
        raise RuntimeError("FlashAttention kernel requires CUDA")
    if query.dtype != torch.float32 or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("FlashAttention kernel currently supports FP32 tensors only")
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key and value must have identical rank-4 shapes")
    if not query.is_contiguous() or not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("query, key and value must be contiguous")
    batch, heads, seq_len, head_dim = query.shape
    if head_dim > 128:
        raise ValueError("FlashAttention kernel supports head_dim <= 128")
    block_d = triton.next_power_of_2(head_dim)
    # The 64x64 tile is a good fit for the 32--64-wide heads in this
    # benchmark.  Halve both sequence tiles for 128-wide heads so the
    # accumulator and K/V tiles stay below the RTX 4060 shared-memory limit.
    block_m = 64 if head_dim <= 64 else 32
    block_n = 64 if head_dim <= 64 else 32
    output = torch.empty_like(query)
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    wrap_triton(_flash_attention_forward_kernel)[grid](
        query,
        key,
        value,
        output,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        seq_len,
        head_dim ** -0.5,
        head_dim=head_dim,
        heads=heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        CAUSAL=causal,
        num_warps=4,
    )
    return output
