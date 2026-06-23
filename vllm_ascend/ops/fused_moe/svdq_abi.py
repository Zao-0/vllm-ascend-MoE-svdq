#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Dedicated W4A8-SVDQ MoE operator ABI checks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoESVDQWeights

SVDQ_OPERATOR_NAME = "dispatch_ffn_combine_w4a8_svdq"

SVDQ_OPERATOR_ARG_ORDER = (
    "x",
    "weight1",
    "weight2",
    "expert_idx",
    "scale1",
    "scale2",
    "bias1",
    "bias2",
    "probs",
    "gate_up_svdq_l1",
    "gate_svdq_l2",
    "up_svdq_l2",
    "down_svdq_l1",
    "down_svdq_l2",
    "gate_rank",
    "up_rank",
    "down_rank",
    "gate_rank_offset",
    "up_rank_offset",
    "group",
    "max_output_size",
    "swiglu_limit",
    "x_active_mask",
    "out",
    "expert_token_nums",
)


def _first_tensor(name: str, value: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if len(value) == 0:
        raise ValueError(f"{name} must not be an empty tensor list.")
    return value[0]


def _require_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.ndim != rank:
        raise ValueError(f"{name} must be rank-{rank}, got shape {tuple(tensor.shape)}.")


def _require_dtype(name: str, tensor: torch.Tensor, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}, got {tensor.dtype}.")


def validate_svdq_operator_abi(
    *,
    x: torch.Tensor,
    weight1: torch.Tensor | Sequence[torch.Tensor],
    weight2: torch.Tensor | Sequence[torch.Tensor],
    expert_idx: torch.Tensor,
    scale1: torch.Tensor | Sequence[torch.Tensor],
    scale2: torch.Tensor | Sequence[torch.Tensor],
    bias1: torch.Tensor | Sequence[torch.Tensor],
    bias2: torch.Tensor | Sequence[torch.Tensor],
    probs: torch.Tensor,
    svdq: MoESVDQWeights,
    group: str,
    max_output_size: int,
    swiglu_limit: int,
    x_active_mask: torch.Tensor | None,
    out: torch.Tensor,
    expert_token_nums: torch.Tensor,
) -> None:
    """Validate the Python-visible ABI before a future SVDQ kernel call."""
    _require_rank("x", x, 2)
    _require_rank("expert_idx", expert_idx, 2)
    _require_rank("probs", probs, 2)
    _require_rank("out", out, 2)
    if x.shape != out.shape:
        raise ValueError(f"out shape {tuple(out.shape)} must match x shape {tuple(x.shape)}.")
    if expert_idx.shape != probs.shape:
        raise ValueError(f"expert_idx shape {tuple(expert_idx.shape)} must match probs shape {tuple(probs.shape)}.")
    if expert_idx.shape[0] != x.shape[0]:
        raise ValueError("expert_idx/probs token dimension must match x.")
    if x_active_mask is not None and x_active_mask.shape[0] != x.shape[0]:
        raise ValueError("x_active_mask token dimension must match x.")
    if not isinstance(group, str) or not group:
        raise ValueError("group must be a non-empty string.")
    if max_output_size <= 0:
        raise ValueError("max_output_size must be positive.")
    if swiglu_limit < 0:
        raise ValueError("swiglu_limit must be non-negative.")

    _require_dtype("gate_up_svdq_l1", svdq.gate_up_svdq_l1, torch.bfloat16)
    _require_dtype("gate_svdq_l2", svdq.gate_svdq_l2, torch.bfloat16)
    _require_dtype("up_svdq_l2", svdq.up_svdq_l2, torch.bfloat16)
    _require_dtype("down_svdq_l1", svdq.down_svdq_l1, torch.bfloat16)
    _require_dtype("down_svdq_l2", svdq.down_svdq_l2, torch.bfloat16)
    for name, tensor in (
        ("gate_up_svdq_l1", svdq.gate_up_svdq_l1),
        ("gate_svdq_l2", svdq.gate_svdq_l2),
        ("up_svdq_l2", svdq.up_svdq_l2),
        ("down_svdq_l1", svdq.down_svdq_l1),
        ("down_svdq_l2", svdq.down_svdq_l2),
    ):
        _require_rank(name, tensor, 3)

    if min(svdq.gate_rank, svdq.up_rank, svdq.down_rank) <= 0:
        raise ValueError("SVDQ ranks must be positive.")
    if svdq.gate_rank_offset != 0:
        raise ValueError(f"gate_rank_offset must be 0, got {svdq.gate_rank_offset}.")
    if svdq.up_rank_offset != svdq.gate_rank:
        raise ValueError(
            f"up_rank_offset must equal gate_rank ({svdq.gate_rank}), got {svdq.up_rank_offset}."
        )

    num_experts, total_gate_up_rank, hidden_size = svdq.gate_up_svdq_l1.shape
    if hidden_size != x.shape[1]:
        raise ValueError(f"gate_up_svdq_l1 hidden dim {hidden_size} must match x hidden dim {x.shape[1]}.")
    if total_gate_up_rank != svdq.gate_rank + svdq.up_rank:
        raise ValueError(
            "gate_up_svdq_l1 rank dim must equal gate_rank + up_rank, "
            f"got {total_gate_up_rank} vs {svdq.gate_rank + svdq.up_rank}."
        )
    if svdq.gate_svdq_l2.shape[0] != num_experts or svdq.gate_svdq_l2.shape[2] != svdq.gate_rank:
        raise ValueError("gate_svdq_l2 shape is inconsistent with gate_rank.")
    intermediate_size = svdq.gate_svdq_l2.shape[1]
    expected_shapes = {
        "up_svdq_l2": (num_experts, intermediate_size, svdq.up_rank),
        "down_svdq_l1": (num_experts, svdq.down_rank, intermediate_size),
        "down_svdq_l2": (num_experts, hidden_size, svdq.down_rank),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(getattr(svdq, name).shape)
        if actual != expected:
            raise ValueError(f"{name} expected shape {expected}, got {actual}.")
    if svdq.gate_svdq_l2 is svdq.up_svdq_l2:
        raise ValueError("gate_svdq_l2 and up_svdq_l2 must be separate tensors.")
    try:
        same_storage = svdq.gate_svdq_l2.data_ptr() == svdq.up_svdq_l2.data_ptr()
    except RuntimeError:
        same_storage = False
    if same_storage:
        raise ValueError("gate_svdq_l2 and up_svdq_l2 must not share storage.")

    residual_w1 = _first_tensor("weight1", weight1)
    residual_w2 = _first_tensor("weight2", weight2)
    _require_rank("weight1", residual_w1, 3)
    _require_rank("weight2", residual_w2, 3)
    if residual_w1.shape[0] != num_experts or residual_w2.shape[0] != num_experts:
        raise ValueError("residual W4A8 expert count must match SVDQ factor expert count.")
    _first_tensor("scale1", scale1)
    _first_tensor("scale2", scale2)
    _first_tensor("bias1", bias1)
    _first_tensor("bias2", bias2)
    _require_rank("expert_token_nums", expert_token_nums, 1)


def make_svdq_weights_from_operator_args(
    *,
    gate_up_svdq_l1: torch.Tensor,
    gate_svdq_l2: torch.Tensor,
    up_svdq_l2: torch.Tensor,
    down_svdq_l1: torch.Tensor,
    down_svdq_l2: torch.Tensor,
    gate_rank: int,
    up_rank: int,
    down_rank: int,
    gate_rank_offset: int,
    up_rank_offset: int,
) -> MoESVDQWeights:
    return MoESVDQWeights(
        gate_up_svdq_l1=gate_up_svdq_l1,
        gate_svdq_l2=gate_svdq_l2,
        up_svdq_l2=up_svdq_l2,
        down_svdq_l1=down_svdq_l1,
        down_svdq_l2=down_svdq_l2,
        gate_rank=gate_rank,
        up_rank=up_rank,
        down_rank=down_rank,
        gate_rank_offset=gate_rank_offset,
        up_rank_offset=up_rank_offset,
    )
