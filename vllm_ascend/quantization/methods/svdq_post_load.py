#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Post-load construction and audit helpers for W4A8-SVDQ MoE factors."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import torch

FINAL_SVDQ_FACTOR_NAMES = (
    "gate_up_svdq_l1",
    "gate_svdq_l2",
    "up_svdq_l2",
    "down_svdq_l1",
    "down_svdq_l2",
)

FORBIDDEN_SVDQ_FACTOR_NAMES = (
    "gate_up_svdq_l2",
)

SVDQ_BF16_DEBUG_STAGE_NAMES = (
    "routing_input",
    "gate_up_l1_rank",
    "gate_rank_split",
    "up_rank_split",
    "gate_l2_output",
    "up_l2_output",
    "down_l1_rank",
    "down_l2_output",
)

SVDQ_MIXED_EPILOGUE_STAGE_NAMES = (
    "gate_mixed",
    "up_mixed",
    "hidden_bf16",
    "hidden_q",
    "hidden_scale",
    "down_mixed",
)

SVDQ_FINAL_COMBINE_STAGE_NAMES = (
    "row_weights",
    "weighted_expert_output",
    "combined_output",
)


def _as_tensor(value: torch.Tensor | torch.nn.Parameter) -> torch.Tensor:
    return value.data if isinstance(value, torch.nn.Parameter) else value


def _require_bf16(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be torch.bfloat16, got {tensor.dtype}.")


def _register_or_replace_parameter(layer: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    param = torch.nn.Parameter(tensor.detach(), requires_grad=False)
    if name in getattr(layer, "_parameters", {}):
        layer._parameters[name] = param
    else:
        layer.register_parameter(name, param)


def build_svdq_operator_factors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Build the five final operator-facing SVDQ factors from raw checkpoint factors."""
    gate_l1 = _as_tensor(layer.gate_svd_l1_raw)
    gate_l2 = _as_tensor(layer.gate_svd_l2_raw)
    up_l1 = _as_tensor(layer.up_svd_l1_raw)
    up_l2 = _as_tensor(layer.up_svd_l2_raw)
    down_l1 = _as_tensor(layer.down_svd_l1_raw)
    down_l2 = _as_tensor(layer.down_svd_l2_raw)

    raw_tensors = {
        "gate_svd_l1_raw": gate_l1,
        "gate_svd_l2_raw": gate_l2,
        "up_svd_l1_raw": up_l1,
        "up_svd_l2_raw": up_l2,
        "down_svd_l1_raw": down_l1,
        "down_svd_l2_raw": down_l2,
    }
    for name, tensor in raw_tensors.items():
        _require_bf16(name, tensor)
        if tensor.ndim != 3:
            raise ValueError(f"{name} must be rank-3 [experts, dim0, dim1], got {tuple(tensor.shape)}.")

    local_experts, gate_rank, hidden_size = gate_l1.shape
    if up_l1.shape[0] != local_experts or up_l1.shape[2] != hidden_size:
        raise ValueError(
            f"up_svd_l1_raw shape {tuple(up_l1.shape)} is incompatible with gate L1 {tuple(gate_l1.shape)}."
        )
    if gate_l2.shape[:1] != (local_experts,) or gate_l2.shape[2] != gate_rank:
        raise ValueError(f"gate_svd_l2_raw shape {tuple(gate_l2.shape)} is incompatible with gate rank {gate_rank}.")
    up_rank = up_l1.shape[1]
    if up_l2.shape[:1] != (local_experts,) or up_l2.shape[2] != up_rank:
        raise ValueError(f"up_svd_l2_raw shape {tuple(up_l2.shape)} is incompatible with up rank {up_rank}.")
    intermediate_size = gate_l2.shape[1]
    if up_l2.shape[1] != intermediate_size:
        raise ValueError(
            f"gate/up L2 intermediate sizes differ: gate={gate_l2.shape[1]}, up={up_l2.shape[1]}."
        )
    down_rank = down_l1.shape[1]
    if down_l1.shape != (local_experts, down_rank, intermediate_size):
        raise ValueError(f"down_svd_l1_raw has invalid shape {tuple(down_l1.shape)}.")
    if down_l2.shape != (local_experts, hidden_size, down_rank):
        raise ValueError(f"down_svd_l2_raw has invalid shape {tuple(down_l2.shape)}.")

    factors = {
        "gate_up_svdq_l1": torch.cat((gate_l1, up_l1), dim=1).contiguous(),
        "gate_svdq_l2": gate_l2.detach().contiguous(),
        "up_svdq_l2": up_l2.detach().contiguous(),
        "down_svdq_l1": down_l1.detach().contiguous(),
        "down_svdq_l2": down_l2.detach().contiguous(),
    }
    for name, tensor in factors.items():
        _register_or_replace_parameter(layer, name, tensor)

    layer.svdq_gate_rank = int(gate_rank)
    layer.svdq_up_rank = int(up_rank)
    layer.svdq_down_rank = int(down_rank)
    layer.svdq_gate_rank_offset = 0
    layer.svdq_up_rank_offset = int(gate_rank)
    return factors


def _sample_checksum(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        data = cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = cpu.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _storage_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        try:
            return int(tensor.storage().nbytes())
        except Exception:
            return int(tensor.numel() * tensor.element_size())


def _tensor_npu_format(tensor: torch.Tensor) -> str:
    if tensor.device.type != "npu":
        return "not_npu"
    try:
        import torch_npu  # type: ignore[import-untyped]

        return str(torch_npu.get_npu_format(tensor))
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def _factor_tp_local_dimensions(
    *,
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    gate_rank: int,
    up_rank: int,
    down_rank: int,
) -> dict[str, dict[str, int]]:
    return {
        "gate_up_svdq_l1": {
            "expert": local_experts,
            "rank": gate_rank + up_rank,
            "hidden": hidden_size,
        },
        "gate_svdq_l2": {
            "expert": local_experts,
            "intermediate": intermediate_size,
            "rank": gate_rank,
        },
        "up_svdq_l2": {
            "expert": local_experts,
            "intermediate": intermediate_size,
            "rank": up_rank,
        },
        "down_svdq_l1": {
            "expert": local_experts,
            "rank": down_rank,
            "intermediate": intermediate_size,
        },
        "down_svdq_l2": {
            "expert": local_experts,
            "hidden": hidden_size,
            "rank": down_rank,
        },
    }


def _tensor_metadata(
    name: str,
    tensor: torch.Tensor,
    *,
    expert_dimension: int,
    tp_local_dimensions: dict[str, int],
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "logical_shape": list(tensor.shape),
        "physical_shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_size_bytes": _storage_nbytes(tensor),
        "storage_offset": int(tensor.storage_offset()),
        "element_size_bytes": int(tensor.element_size()),
        "numel": int(tensor.numel()),
        "npu_format": _tensor_npu_format(tensor),
        "expert_dimension": expert_dimension,
        "tp_local_dimensions": tp_local_dimensions,
        "sample_checksum": _sample_checksum(tensor[0]) if tensor.shape[0] > 0 else None,
    }


def _svdq_operator_contract_metadata(layer: torch.nn.Module) -> dict[str, Any]:
    forbidden_present = [name for name in FORBIDDEN_SVDQ_FACTOR_NAMES if hasattr(layer, name)]
    if forbidden_present:
        raise ValueError(
            "forbidden fused SVDQ factors are present in the operator ABI: "
            f"{', '.join(forbidden_present)}."
        )

    missing_factors = [name for name in FINAL_SVDQ_FACTOR_NAMES if not hasattr(layer, name)]
    return {
        "operator_factor_names": list(FINAL_SVDQ_FACTOR_NAMES),
        "operator_factor_count": len(FINAL_SVDQ_FACTOR_NAMES),
        "missing_operator_factors": missing_factors,
        "forbidden_factor_names": list(FORBIDDEN_SVDQ_FACTOR_NAMES),
        "forbidden_factors_present": forbidden_present,
        "uses_fused_gate_up_l1": True,
        "uses_separate_gate_up_l2": True,
        "bf16_lowrank_stage_names": list(SVDQ_BF16_DEBUG_STAGE_NAMES),
        "mixed_epilogue_stage_names": list(SVDQ_MIXED_EPILOGUE_STAGE_NAMES),
        "final_combine_stage_names": list(SVDQ_FINAL_COMBINE_STAGE_NAMES),
        "pre_swiglu_addends": {
            "gate": ["w4a8_residual_gate", "bf16_gate_lowrank"],
            "up": ["w4a8_residual_up", "bf16_up_lowrank"],
        },
        "pre_final_combine_addends": {
            "down": ["w4a8_residual_down", "bf16_down_lowrank"],
        },
    }


def _branch_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual - expected).abs()
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def _stage_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool | int]:
    actual_finite = bool(torch.isfinite(actual).all().item()) if actual.numel() else True
    expected_finite = bool(torch.isfinite(expected).all().item()) if expected.numel() else True
    diff = (actual - expected).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    if diff.numel() and diff_finite:
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
    elif diff.numel():
        max_abs = float("inf")
        mean_abs = float("inf")
    else:
        max_abs = 0.0
        mean_abs = 0.0

    if expected.numel() and expected_finite:
        signal = float(expected.abs().max().item())
    else:
        signal = 0.0
    if signal > 0.0:
        max_signal_relative = max_abs / signal
        mean_signal_relative = mean_abs / signal
    else:
        max_signal_relative = 0.0 if max_abs == 0.0 else float("inf")
        mean_signal_relative = 0.0 if mean_abs == 0.0 else float("inf")

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_signal_relative": max_signal_relative,
        "mean_signal_relative": mean_signal_relative,
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "diff_finite": diff_finite,
        "numel": int(actual.numel()),
    }


def _stage_error_metadata(
    stages: dict[str, torch.Tensor],
    references: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | bool | int]]:
    return {name: _stage_error(stages[name], references[name]) for name in SVDQ_BF16_DEBUG_STAGE_NAMES}


def _evaluate_svdq_bf16_debug_stages(
    *,
    x: torch.Tensor,
    hidden: torch.Tensor,
    gate_up_l1: torch.Tensor,
    gate_l2: torch.Tensor,
    up_l2: torch.Tensor,
    down_l1: torch.Tensor,
    down_l2: torch.Tensor,
    gate_rank: int,
    up_rank: int,
    gate_offset: int,
    up_offset: int,
) -> dict[str, torch.Tensor]:
    """Evaluate named BF16 low-rank stages with final operator-facing factors."""
    fused_rank = x @ gate_up_l1.T
    gate_rank_state = fused_rank[:, gate_offset : gate_offset + gate_rank]
    up_rank_state = fused_rank[:, up_offset : up_offset + up_rank]
    down_rank_state = hidden @ down_l1.T
    return {
        "routing_input": x,
        "gate_up_l1_rank": fused_rank,
        "gate_rank_split": gate_rank_state,
        "up_rank_split": up_rank_state,
        "gate_l2_output": gate_rank_state @ gate_l2.T,
        "up_l2_output": up_rank_state @ up_l2.T,
        "down_l1_rank": down_rank_state,
        "down_l2_output": down_rank_state @ down_l2.T,
    }


def _stage_shape_metadata(stages: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(stages[name].shape) for name in SVDQ_BF16_DEBUG_STAGE_NAMES}


def _cpu_dynamic_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.detach().float().cpu()
    max_abs = x_fp32.abs().amax(dim=1)
    scale = (max_abs / 127.0).float()
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(x_fp32 / safe_scale[:, None]).clamp(-127, 127).to(torch.int8)
    q = torch.where((scale == 0)[:, None], torch.zeros_like(q), q)
    return q, scale


def build_svdq_mixed_epilogue_reference(
    *,
    residual_gate_up: torch.Tensor,
    gate_lowrank: torch.Tensor,
    up_lowrank: torch.Tensor,
    residual_down: torch.Tensor,
    down_lowrank: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> dict[str, Any]:
    """Build the mixed BF16/W4A8 epilogue reference for one expert slice.

    The W4A8 residual branch and BF16 low-rank branch are added before SwiGLU
    and before final down output emission. The hidden activation quantization
    uses the same per-row dynamic quantization formula validated for the NPU
    activation quantization probe.
    """
    if residual_gate_up.ndim != 2:
        raise ValueError(
            f"residual_gate_up must be rank-2 [tokens, 2 * intermediate], got {tuple(residual_gate_up.shape)}."
        )
    if residual_gate_up.shape[1] % 2 != 0:
        raise ValueError(f"residual_gate_up width must be even, got {residual_gate_up.shape[1]}.")

    tokens = int(residual_gate_up.shape[0])
    intermediate_size = int(residual_gate_up.shape[1] // 2)
    if tuple(gate_lowrank.shape) != (tokens, intermediate_size):
        raise ValueError(f"gate_lowrank expected shape {(tokens, intermediate_size)}, got {tuple(gate_lowrank.shape)}.")
    if tuple(up_lowrank.shape) != (tokens, intermediate_size):
        raise ValueError(f"up_lowrank expected shape {(tokens, intermediate_size)}, got {tuple(up_lowrank.shape)}.")
    if residual_down.shape != down_lowrank.shape:
        raise ValueError(
            "residual_down and down_lowrank must share shape, "
            f"got {tuple(residual_down.shape)} and {tuple(down_lowrank.shape)}."
        )
    if int(residual_down.shape[0]) != tokens:
        raise ValueError(f"down branch token count {residual_down.shape[0]} does not match gate/up tokens {tokens}.")
    if swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}.")

    residual_gate, residual_up = residual_gate_up.detach().float().cpu().chunk(2, dim=1)
    gate_mixed = residual_gate + gate_lowrank.detach().float().cpu()
    up_mixed = residual_up + up_lowrank.detach().float().cpu()
    if swiglu_limit > 0:
        gate_mixed = gate_mixed.clamp(min=-float(swiglu_limit), max=float(swiglu_limit))
        up_mixed = up_mixed.clamp(min=-float(swiglu_limit), max=float(swiglu_limit))
    hidden_bf16 = (torch.nn.functional.silu(gate_mixed) * up_mixed).to(torch.bfloat16)
    hidden_q, hidden_scale = _cpu_dynamic_quant_reference(hidden_bf16)
    down_mixed = residual_down.detach().float().cpu() + down_lowrank.detach().float().cpu()
    stages = {
        "gate_mixed": gate_mixed,
        "up_mixed": up_mixed,
        "hidden_bf16": hidden_bf16,
        "hidden_q": hidden_q,
        "hidden_scale": hidden_scale,
        "down_mixed": down_mixed,
    }
    return {
        "stages": stages,
        "stage_shapes": {name: list(stages[name].shape) for name in SVDQ_MIXED_EPILOGUE_STAGE_NAMES},
        "all_finite": all(
            bool(torch.isfinite(tensor.float()).all().item())
            for tensor in stages.values()
            if tensor.dtype != torch.int8
        ),
        "swiglu_limit": float(swiglu_limit),
    }


def _resolve_final_combine_indices(
    *,
    num_rows: int,
    num_tokens: int,
    top_k: int,
    expanded_row_idx: torch.Tensor | None,
    token_indices: torch.Tensor | None,
    topk_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if expanded_row_idx is not None:
        if token_indices is not None or topk_indices is not None:
            raise ValueError("expanded_row_idx cannot be combined with token_indices/topk_indices.")
        if expanded_row_idx.ndim != 1:
            raise ValueError(f"expanded_row_idx must be rank-1, got {tuple(expanded_row_idx.shape)}.")
        if int(expanded_row_idx.shape[0]) != num_rows:
            raise ValueError(f"expanded_row_idx length {expanded_row_idx.shape[0]} does not match rows {num_rows}.")
        flat_indices = expanded_row_idx.detach().long().abs().cpu()
        resolved_token_indices = torch.div(flat_indices, top_k, rounding_mode="floor")
        resolved_topk_indices = flat_indices.remainder(top_k)
    else:
        if token_indices is None or topk_indices is None:
            raise ValueError("provide either expanded_row_idx or both token_indices and topk_indices.")
        if token_indices.ndim != 1 or topk_indices.ndim != 1:
            raise ValueError(
                f"token_indices and topk_indices must be rank-1, got "
                f"{tuple(token_indices.shape)} and {tuple(topk_indices.shape)}."
            )
        if int(token_indices.shape[0]) != num_rows or int(topk_indices.shape[0]) != num_rows:
            raise ValueError(
                "token_indices/topk_indices lengths must match routed_output rows, "
                f"got {token_indices.shape[0]}, {topk_indices.shape[0]}, and {num_rows}."
            )
        resolved_token_indices = token_indices.detach().long().cpu()
        resolved_topk_indices = topk_indices.detach().long().cpu()

    if resolved_token_indices.numel():
        min_token = int(resolved_token_indices.min().item())
        max_token = int(resolved_token_indices.max().item())
        min_topk = int(resolved_topk_indices.min().item())
        max_topk = int(resolved_topk_indices.max().item())
        if min_token < 0 or max_token >= num_tokens:
            raise ValueError(f"token index range [{min_token}, {max_token}] is outside [0, {num_tokens}).")
        if min_topk < 0 or max_topk >= top_k:
            raise ValueError(f"top-k index range [{min_topk}, {max_topk}] is outside [0, {top_k}).")
    return resolved_token_indices, resolved_topk_indices


def build_svdq_final_combine_reference(
    *,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    expanded_row_idx: torch.Tensor | None = None,
    token_indices: torch.Tensor | None = None,
    topk_indices: torch.Tensor | None = None,
    num_tokens: int | None = None,
) -> dict[str, Any]:
    """Build the final routed-output combine reference.

    The default index mode mirrors the all-gather token-combine surface, where
    ``expanded_row_idx`` addresses flattened ``[token, top_k]`` slots and
    ``topk_weights`` supplies the per-route probabilities.
    """
    if routed_output.ndim != 2:
        raise ValueError(f"routed_output must be rank-2 [rows, hidden], got {tuple(routed_output.shape)}.")
    if topk_weights.ndim != 2:
        raise ValueError(f"topk_weights must be rank-2 [tokens, top_k], got {tuple(topk_weights.shape)}.")

    num_rows = int(routed_output.shape[0])
    hidden_size = int(routed_output.shape[1])
    weights_num_tokens = int(topk_weights.shape[0])
    top_k = int(topk_weights.shape[1])
    if top_k <= 0:
        raise ValueError("topk_weights must have a positive top_k dimension.")
    if num_tokens is None:
        num_tokens = weights_num_tokens
    else:
        num_tokens = int(num_tokens)
        if num_tokens != weights_num_tokens:
            raise ValueError(f"num_tokens {num_tokens} must match topk_weights tokens {weights_num_tokens}.")

    resolved_token_indices, resolved_topk_indices = _resolve_final_combine_indices(
        num_rows=num_rows,
        num_tokens=num_tokens,
        top_k=top_k,
        expanded_row_idx=expanded_row_idx,
        token_indices=token_indices,
        topk_indices=topk_indices,
    )

    routed_output_fp32 = routed_output.detach().float().cpu()
    topk_weights_fp32 = topk_weights.detach().float().cpu()
    row_weights = topk_weights_fp32[resolved_token_indices, resolved_topk_indices]
    weighted_expert_output = routed_output_fp32 * row_weights[:, None]
    combined_output = torch.zeros((num_tokens, hidden_size), dtype=torch.float32)
    if num_rows:
        combined_output.index_add_(0, resolved_token_indices, weighted_expert_output)

    stages = {
        "row_weights": row_weights,
        "weighted_expert_output": weighted_expert_output,
        "combined_output": combined_output,
    }
    return {
        "stages": stages,
        "stage_shapes": {name: list(stages[name].shape) for name in SVDQ_FINAL_COMBINE_STAGE_NAMES},
        "all_finite": all(bool(torch.isfinite(tensor).all().item()) for tensor in stages.values()),
        "num_tokens": num_tokens,
        "top_k": top_k,
        "resolved_token_indices": resolved_token_indices,
        "resolved_topk_indices": resolved_topk_indices,
    }


def _branch_isolation_errors(
    *,
    fused_rank: torch.Tensor,
    gate_l2: torch.Tensor,
    up_l2: torch.Tensor,
    gate_rank: int,
    up_rank: int,
    gate_offset: int,
    up_offset: int,
) -> dict[str, dict[str, float]]:
    gate_rank_state = fused_rank[:, gate_offset : gate_offset + gate_rank]
    up_rank_state = fused_rank[:, up_offset : up_offset + up_rank]
    gate_base = gate_rank_state @ gate_l2.T
    up_base = up_rank_state @ up_l2.T

    gate_perturbed = fused_rank.clone()
    gate_perturbed[:, gate_offset : gate_offset + gate_rank] += 1.0
    up_after_gate_perturb = gate_perturbed[:, up_offset : up_offset + up_rank] @ up_l2.T

    up_perturbed = fused_rank.clone()
    up_perturbed[:, up_offset : up_offset + up_rank] += 1.0
    gate_after_up_perturb = up_perturbed[:, gate_offset : gate_offset + gate_rank] @ gate_l2.T

    return {
        "gate_rank_perturb_does_not_change_up_l2": _branch_error(up_after_gate_perturb, up_base),
        "up_rank_perturb_does_not_change_gate_l2": _branch_error(gate_after_up_perturb, gate_base),
    }


def build_svdq_bf16_stage_reference(
    layer: torch.nn.Module,
    expert: int,
    *,
    x: torch.Tensor | None = None,
    hidden: torch.Tensor | None = None,
    num_tokens: int = 3,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Build E2-E7 BF16 low-rank stage references for one post-loaded expert."""
    gate_up_l1 = _as_tensor(layer.gate_up_svdq_l1)
    gate_l2 = _as_tensor(layer.gate_svdq_l2)
    up_l2 = _as_tensor(layer.up_svdq_l2)
    down_l1 = _as_tensor(layer.down_svdq_l1)
    down_l2 = _as_tensor(layer.down_svdq_l2)
    final_tensors = {
        "gate_up_svdq_l1": gate_up_l1,
        "gate_svdq_l2": gate_l2,
        "up_svdq_l2": up_l2,
        "down_svdq_l1": down_l1,
        "down_svdq_l2": down_l2,
    }
    for name, tensor in final_tensors.items():
        _require_bf16(name, tensor)

    local_experts = int(gate_up_l1.shape[0])
    if expert < 0 or expert >= local_experts:
        raise ValueError(f"expert {expert} is outside local expert range [0, {local_experts}).")

    gate_rank = int(layer.svdq_gate_rank)
    up_rank = int(layer.svdq_up_rank)
    down_rank = int(layer.svdq_down_rank)
    gate_offset = int(layer.svdq_gate_rank_offset)
    up_offset = int(layer.svdq_up_rank_offset)
    if gate_offset != 0 or up_offset != gate_rank:
        raise ValueError(
            f"invalid SVDQ rank offsets: gate={gate_offset}, up={up_offset}, gate_rank={gate_rank}."
        )

    hidden_size = int(gate_up_l1.shape[2])
    intermediate_size = int(gate_l2.shape[1])
    if x is None:
        x = torch.randn(num_tokens, hidden_size, generator=generator)
    else:
        x = x.detach().float().cpu()
        num_tokens = int(x.shape[0])
    if hidden is None:
        hidden = torch.randn(num_tokens, intermediate_size, generator=generator)
    else:
        hidden = hidden.detach().float().cpu()
    if tuple(x.shape) != (num_tokens, hidden_size):
        raise ValueError(f"x expected shape {(num_tokens, hidden_size)}, got {tuple(x.shape)}.")
    if tuple(hidden.shape) != (num_tokens, intermediate_size):
        raise ValueError(f"hidden expected shape {(num_tokens, intermediate_size)}, got {tuple(hidden.shape)}.")

    raw_gate_l1 = _as_tensor(layer.gate_svd_l1_raw)[expert].detach().float().cpu()
    raw_gate_l2 = _as_tensor(layer.gate_svd_l2_raw)[expert].detach().float().cpu()
    raw_up_l1 = _as_tensor(layer.up_svd_l1_raw)[expert].detach().float().cpu()
    raw_up_l2 = _as_tensor(layer.up_svd_l2_raw)[expert].detach().float().cpu()
    raw_down_l1 = _as_tensor(layer.down_svd_l1_raw)[expert].detach().float().cpu()
    raw_down_l2 = _as_tensor(layer.down_svd_l2_raw)[expert].detach().float().cpu()

    final_gate_up_l1 = gate_up_l1[expert].detach().float().cpu()
    final_gate_l2 = gate_l2[expert].detach().float().cpu()
    final_up_l2 = up_l2[expert].detach().float().cpu()
    final_down_l1 = down_l1[expert].detach().float().cpu()
    final_down_l2 = down_l2[expert].detach().float().cpu()

    stages = _evaluate_svdq_bf16_debug_stages(
        x=x,
        hidden=hidden,
        gate_up_l1=final_gate_up_l1,
        gate_l2=final_gate_l2,
        up_l2=final_up_l2,
        down_l1=final_down_l1,
        down_l2=final_down_l2,
        gate_rank=gate_rank,
        up_rank=up_rank,
        gate_offset=gate_offset,
        up_offset=up_offset,
    )

    gate_rank_ref = x @ raw_gate_l1.T
    up_rank_ref = x @ raw_up_l1.T
    down_rank_ref = hidden @ raw_down_l1.T
    references = {
        "routing_input": x,
        "gate_up_l1_rank": torch.cat((gate_rank_ref, up_rank_ref), dim=1),
        "gate_rank_split": gate_rank_ref,
        "up_rank_split": up_rank_ref,
        "gate_l2_output": gate_rank_ref @ raw_gate_l2.T,
        "up_l2_output": up_rank_ref @ raw_up_l2.T,
        "down_l1_rank": down_rank_ref,
        "down_l2_output": down_rank_ref @ raw_down_l2.T,
    }

    return {
        "expert": expert,
        "actual": stages,
        "reference": references,
        "stage_shapes": _stage_shape_metadata(stages),
        "stage_errors": _stage_error_metadata(stages, references),
        "branch_isolation": _branch_isolation_errors(
            fused_rank=stages["gate_up_l1_rank"],
            gate_l2=final_gate_l2,
            up_l2=final_up_l2,
            gate_rank=gate_rank,
            up_rank=up_rank,
            gate_offset=gate_offset,
            up_offset=up_offset,
        ),
        "rank_metadata": {
            "gate_rank": gate_rank,
            "up_rank": up_rank,
            "down_rank": down_rank,
            "gate_rank_offset": gate_offset,
            "up_rank_offset": up_offset,
        },
    }


def audit_svdq_operator_factors(
    layer: torch.nn.Module,
    *,
    max_experts: int = 2,
    num_tokens: int = 3,
) -> dict[str, Any]:
    """Run a sampled numerical audit from raw factors to final operator factors."""
    gate_up_l1 = _as_tensor(layer.gate_up_svdq_l1)
    gate_l2 = _as_tensor(layer.gate_svdq_l2)
    up_l2 = _as_tensor(layer.up_svdq_l2)
    down_l1 = _as_tensor(layer.down_svdq_l1)
    down_l2 = _as_tensor(layer.down_svdq_l2)
    final_tensors = {
        "gate_up_svdq_l1": gate_up_l1,
        "gate_svdq_l2": gate_l2,
        "up_svdq_l2": up_l2,
        "down_svdq_l1": down_l1,
        "down_svdq_l2": down_l2,
    }
    for name, tensor in final_tensors.items():
        _require_bf16(name, tensor)

    gate_rank = int(layer.svdq_gate_rank)
    up_rank = int(layer.svdq_up_rank)
    down_rank = int(layer.svdq_down_rank)
    gate_offset = int(layer.svdq_gate_rank_offset)
    up_offset = int(layer.svdq_up_rank_offset)
    local_experts = int(gate_up_l1.shape[0])
    hidden_size = int(gate_up_l1.shape[2])
    intermediate_size = int(gate_l2.shape[1])
    operator_contract = _svdq_operator_contract_metadata(layer)

    expected_shapes = {
        "gate_up_svdq_l1": (local_experts, gate_rank + up_rank, hidden_size),
        "gate_svdq_l2": (local_experts, intermediate_size, gate_rank),
        "up_svdq_l2": (local_experts, intermediate_size, up_rank),
        "down_svdq_l1": (local_experts, down_rank, intermediate_size),
        "down_svdq_l2": (local_experts, hidden_size, down_rank),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(final_tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} expected shape {expected}, got {actual}.")
    if gate_offset != 0 or up_offset != gate_rank:
        raise ValueError(
            f"invalid SVDQ rank offsets: gate={gate_offset}, up={up_offset}, gate_rank={gate_rank}."
        )

    generator = torch.Generator(device="cpu").manual_seed(20260623)
    sampled_experts = list(range(min(local_experts, max_experts)))
    branch_errors: list[dict[str, Any]] = []
    for expert in sampled_experts:
        stage_audit = build_svdq_bf16_stage_reference(
            layer,
            expert,
            num_tokens=num_tokens,
            generator=generator,
        )
        stages = stage_audit["actual"]
        stage_references = stage_audit["reference"]

        branch_errors.append(
            {
                "expert": expert,
                "gate": _branch_error(stages["gate_l2_output"], stage_references["gate_l2_output"]),
                "up": _branch_error(stages["up_l2_output"], stage_references["up_l2_output"]),
                "down": _branch_error(stages["down_l2_output"], stage_references["down_l2_output"]),
                "stage_shapes": stage_audit["stage_shapes"],
                "stage_errors": stage_audit["stage_errors"],
                "branch_isolation": stage_audit["branch_isolation"],
            }
        )

    max_abs = 0.0
    bf16_stage_max_abs = 0.0
    all_finite = True
    bf16_stage_all_finite = True
    for entry in branch_errors:
        max_abs = max(max_abs, entry["gate"]["max_abs"], entry["up"]["max_abs"], entry["down"]["max_abs"])
        for stage_error in entry["stage_errors"].values():
            max_abs = max(max_abs, float(stage_error["max_abs"]))
            bf16_stage_max_abs = max(bf16_stage_max_abs, float(stage_error["max_abs"]))
            all_finite = all_finite and bool(stage_error["actual_finite"])
            all_finite = all_finite and bool(stage_error["expected_finite"])
            all_finite = all_finite and bool(stage_error["diff_finite"])
            bf16_stage_all_finite = bf16_stage_all_finite and bool(stage_error["actual_finite"])
            bf16_stage_all_finite = bf16_stage_all_finite and bool(stage_error["expected_finite"])
            bf16_stage_all_finite = bf16_stage_all_finite and bool(stage_error["diff_finite"])
        for isolation_error in entry["branch_isolation"].values():
            max_abs = max(max_abs, isolation_error["max_abs"])

    tp_local_dimensions = _factor_tp_local_dimensions(
        local_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_rank=gate_rank,
        up_rank=up_rank,
        down_rank=down_rank,
    )

    return {
        "passed": max_abs == 0.0 and all_finite,
        "max_abs": max_abs,
        "all_finite": all_finite,
        "bf16_stage_names": list(SVDQ_BF16_DEBUG_STAGE_NAMES),
        "bf16_stage_max_abs": bf16_stage_max_abs,
        "bf16_stage_all_finite": bf16_stage_all_finite,
        "operator_contract": operator_contract,
        "sampled_experts": sampled_experts,
        "branch_errors": branch_errors,
        "factor_metadata": {
            name: _tensor_metadata(
                name,
                tensor,
                expert_dimension=0,
                tp_local_dimensions=tp_local_dimensions[name],
            )
            for name, tensor in final_tensors.items()
        },
        "rank_metadata": {
            "gate_rank": gate_rank,
            "up_rank": up_rank,
            "down_rank": down_rank,
            "gate_rank_offset": gate_offset,
            "up_rank_offset": up_offset,
        },
    }


def emit_svdq_operator_audit(layer: torch.nn.Module, evidence_dir: str) -> str:
    os.makedirs(evidence_dir, exist_ok=True)
    audit = audit_svdq_operator_factors(layer)
    layer_name = getattr(layer, "layer_name", "unknown").replace(".", "_")
    path = os.path.join(evidence_dir, f"{layer_name}_svdq_operator_factor_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"layer_name": getattr(layer, "layer_name", "unknown"), **audit}, f, indent=2)
    return path
