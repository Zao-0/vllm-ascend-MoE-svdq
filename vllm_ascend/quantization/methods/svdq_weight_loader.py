#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""SVDQ factor loader hooks for FusedMoE parameters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm_ascend.quantization.svdq_spec import SVDQ_FACTOR_SPECS


def _get_tp_rank(layer: torch.nn.Module) -> int:
    tp_rank = getattr(layer, "tp_rank", None)
    if tp_rank is not None:
        return int(tp_rank)
    moe_parallel_config = getattr(layer, "moe_parallel_config", None)
    return int(getattr(moe_parallel_config, "tp_rank", 0))


def _map_global_expert_id_to_local_expert_id(layer: torch.nn.Module, expert_id: int) -> int:
    if hasattr(layer, "_map_global_expert_id_to_local_expert_id"):
        return int(layer._map_global_expert_id_to_local_expert_id(expert_id))
    expert_map = getattr(layer, "_expert_map", None)
    if expert_map is None:
        expert_map = getattr(layer, "expert_map", None)
    if expert_map is None:
        return expert_id
    return int(expert_map[expert_id].item())


def _load_svdq_factor(
    layer: torch.nn.Module,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    global_expert_id: int,
    return_success: bool,
) -> bool | None:
    if shard_id not in SVDQ_FACTOR_SPECS:
        raise ValueError(f"Unsupported SVDQ shard_id={shard_id}.")
    expected_shard_id = getattr(param, "svdq_shard_id", None)
    if expected_shard_id is not None and expected_shard_id != shard_id:
        raise ValueError(
            f"SVDQ shard mismatch for {weight_name}: parameter expects {expected_shard_id}, loader got {shard_id}."
        )

    expert_id = _map_global_expert_id_to_local_expert_id(layer, global_expert_id)
    if expert_id == -1:
        return False if return_success else None
    if loaded_weight.ndim != 2:
        raise ValueError(f"SVDQ factor {weight_name} must be rank-2, got shape {tuple(loaded_weight.shape)}.")

    expected_dtype = getattr(param, "svdq_dtype", torch.bfloat16)
    if loaded_weight.dtype != expected_dtype:
        raise ValueError(f"SVDQ factor {weight_name} must use dtype {expected_dtype}, got {loaded_weight.dtype}.")

    rank_dim = getattr(param, "svdq_rank_dim", None)
    expected_rank = getattr(param, "svdq_rank", None)
    if expected_rank is not None and rank_dim is not None and loaded_weight.shape[rank_dim] != expected_rank:
        raise ValueError(
            f"SVDQ factor {weight_name} rank mismatch on dim {rank_dim}: "
            f"expected {expected_rank}, got {loaded_weight.shape[rank_dim]}."
        )

    loaded_experts = getattr(param, "svdq_loaded_experts", None)
    if loaded_experts is not None and expert_id in loaded_experts:
        raise ValueError(f"Duplicate SVDQ factor load for {weight_name}, local expert {expert_id}.")

    expert_data = param.data[expert_id]
    _, _, shard_dim, _ = SVDQ_FACTOR_SPECS[shard_id]
    if shard_dim is not None and loaded_weight.shape[shard_dim] != expert_data.shape[shard_dim]:
        start_offset = expert_data.shape[shard_dim] * _get_tp_rank(layer)
        available = loaded_weight.shape[shard_dim] - start_offset
        if available <= 0:
            return False if return_success else None
        narrow_size = min(expert_data.shape[shard_dim], available)
        loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)

    target = expert_data
    for dim, loaded_dim in enumerate(loaded_weight.shape):
        target_dim = target.shape[dim]
        if target_dim == loaded_dim:
            continue
        if target_dim < loaded_dim:
            raise ValueError(
                f"SVDQ target for {weight_name} is smaller than loaded tensor on dim {dim}: "
                f"target={tuple(expert_data.shape)}, loaded={tuple(loaded_weight.shape)}."
            )
        target = target.narrow(dim, 0, loaded_dim)
    target.copy_(loaded_weight)
    if loaded_experts is not None:
        loaded_experts.add(expert_id)
    return True if return_success else None


def make_svdq_factor_weight_loader(layer: torch.nn.Module) -> Callable[..., bool | None]:
    """Return a loader for SVDQ factors allocated by the Ascend quant scheme.

    The semantics intentionally match the upstream SVDQ branch in vLLM's
    FusedMoE loader, but are implemented here because vLLM-Ascend cannot rely
    on that branch being present in the installed vLLM package.
    """

    def load_factor(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
        **_: Any,
    ) -> bool | None:
        return _load_svdq_factor(
            layer=layer,
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            global_expert_id=expert_id,
            return_success=return_success,
        )

    return load_factor
