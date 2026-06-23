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
        raise ValueError(f"up_svd_l1_raw shape {tuple(up_l1.shape)} is incompatible with gate L1 {tuple(gate_l1.shape)}.")
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
