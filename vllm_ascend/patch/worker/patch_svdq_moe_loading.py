#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Patch installed vLLM MoE expert mappings for ModelSlim SVDQ factors."""

from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from vllm_ascend.quantization.svdq_spec import SVDQ_FACTOR_SPECS


def _make_svdq_mappings(
    *,
    ckpt_gate_proj_name: str,
    ckpt_down_proj_name: str,
    ckpt_up_proj_name: str,
    num_experts: int,
    num_redundant_experts: int,
    base_layer: str,
) -> list[tuple[str, str, int, str]]:
    from vllm.model_executor.layers.fused_moe.layer import EplbState

    physical_to_logical_map = EplbState.build_initial_global_physical_to_logical_map(
        num_experts,
        num_redundant_experts,
    )
    projection_to_ckpt_name = {
        "gate_proj": ckpt_gate_proj_name,
        "up_proj": ckpt_up_proj_name,
        "down_proj": ckpt_down_proj_name,
    }

    mappings = []
    for expert_id in range(num_experts + num_redundant_experts):
        logical_expert_id = physical_to_logical_map[expert_id]
        for param_name, (projection, factor_kind, _, _) in SVDQ_FACTOR_SPECS.items():
            ckpt_projection = projection_to_ckpt_name[projection]
            mappings.append(
                (
                    f"experts.{base_layer}{param_name}",
                    f"experts.{logical_expert_id}.{ckpt_projection}.{factor_kind}",
                    expert_id,
                    param_name,
                )
            )
    return mappings


def _patch_fused_moe_expert_mapping() -> None:
    original = FusedMoE.make_expert_params_mapping
    if getattr(original, "_vllm_ascend_svdq_patched", False):
        return

    def patched_make_expert_params_mapping(
        cls: type[FusedMoE],
        model: Any,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        mappings = original.__func__(
            cls,
            model,
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            num_experts,
            num_redundant_experts,
        )
        if any(shard_id in SVDQ_FACTOR_SPECS for _, _, _, shard_id in mappings):
            return mappings

        base_layer = "base_layer." if any(".base_layer." in name for name, _ in model.named_parameters()) else ""
        svdq_mappings = _make_svdq_mappings(
            ckpt_gate_proj_name=ckpt_gate_proj_name,
            ckpt_down_proj_name=ckpt_down_proj_name,
            ckpt_up_proj_name=ckpt_up_proj_name,
            num_experts=num_experts,
            num_redundant_experts=num_redundant_experts,
            base_layer=base_layer,
        )
        return svdq_mappings + mappings

    patched_make_expert_params_mapping._vllm_ascend_svdq_patched = True  # type: ignore[attr-defined]
    FusedMoE.make_expert_params_mapping = classmethod(patched_make_expert_params_mapping)


_patch_fused_moe_expert_mapping()
