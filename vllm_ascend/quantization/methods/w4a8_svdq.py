#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""W4A8 residual plus BF16 low-rank SVDQ MoE scheme."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

import torch
from vllm.config import get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.svdq_spec import SVDQ_FACTOR_SPECS, SVDQ_FACTOR_TO_RAW_NAME, SVDQMoELayerSpec

from .base import get_moe_num_logical_experts
from .svdq_post_load import build_svdq_operator_factors, emit_svdq_operator_audit
from .svdq_weight_loader import make_svdq_factor_weight_loader
from .w4a8 import AscendW4A8DynamicFusedMoEMethod


class AscendW4A8SVDQFusedMoEMethod(AscendW4A8DynamicFusedMoEMethod):
    """Composite ModelSlim W4A8-SVDQ MoE scheme.

    The W4A8 residual allocation, loading, and post-load processing are
    inherited from the official W4A8 implementation. This class adds only the
    BF16 low-rank staging factors and their loader metadata.
    """

    quant_type: QuantType = QuantType.W4A8_SVDQ

    def __init__(self, spec: SVDQMoELayerSpec):
        self.svdq_spec = spec
        super().__init__()
        if self.dynamic_eplb:
            raise NotImplementedError("W4A8-SVDQ does not support dynamic EPLB until factors are reordered too.")
        vllm_config = get_current_vllm_config()
        offload_config = getattr(vllm_config, "offload_config", None)
        uva_config = getattr(offload_config, "uva", None)
        prefetch_config = getattr(offload_config, "prefetch", None)
        if uva_config is not None and getattr(uva_config, "cpu_offload_gb", 0) > 0:
            raise NotImplementedError("W4A8-SVDQ does not support UVA/CPU weight offload.")
        if prefetch_config is not None and getattr(prefetch_config, "offload_group_size", 0) > 0:
            raise NotImplementedError("W4A8-SVDQ does not support prefetch weight offload.")
        if get_ascend_config().eplb_config.dynamic_eplb:
            raise NotImplementedError("W4A8-SVDQ does not support dynamic EPLB.")

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        param_dict = super().get_weight(num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype)
        spec = self.svdq_spec
        if hidden_sizes != spec.hidden_size:
            raise ValueError(f"SVDQ hidden size mismatch: expected {spec.hidden_size}, got {hidden_sizes}.")
        dtype = spec.lowrank_dtype
        param_dict.update(
            {
                "svdq_w1_l1": torch.empty(num_experts, spec.gate_rank, hidden_sizes, dtype=dtype),
                "svdq_w1_l2": torch.empty(num_experts, intermediate_size_per_partition, spec.gate_rank, dtype=dtype),
                "svdq_w3_l1": torch.empty(num_experts, spec.up_rank, hidden_sizes, dtype=dtype),
                "svdq_w3_l2": torch.empty(num_experts, intermediate_size_per_partition, spec.up_rank, dtype=dtype),
                "svdq_w2_l1": torch.empty(num_experts, spec.down_rank, intermediate_size_per_partition, dtype=dtype),
                "svdq_w2_l2": torch.empty(num_experts, hidden_sizes, spec.down_rank, dtype=dtype),
            }
        )
        return param_dict

    def get_weight_attrs(
        self,
        param_key: str,
        default_attrs: dict[str, Any],
        layer: torch.nn.Module,
    ) -> dict[str, Any]:
        if param_key not in SVDQ_FACTOR_SPECS:
            return default_attrs

        _, _, shard_dim, rank_dim = SVDQ_FACTOR_SPECS[param_key]
        attrs = dict(default_attrs)
        attrs["weight_loader"] = make_svdq_factor_weight_loader(layer)
        attrs["svdq_shard_id"] = param_key
        attrs["svdq_raw_name"] = SVDQ_FACTOR_TO_RAW_NAME[param_key]
        attrs["svdq_sharded_dim"] = shard_dim
        attrs["svdq_rank_dim"] = rank_dim
        attrs["svdq_rank"] = self.svdq_spec.rank_by_loader_name[param_key]
        attrs["svdq_dtype"] = self.svdq_spec.lowrank_dtype
        attrs["svdq_loaded_experts"] = set()
        return attrs

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._validate_raw_factor_loads(layer)
        self._emit_raw_load_manifest(layer)
        super().process_weights_after_loading(layer)
        self._attach_raw_factor_aliases(layer)
        build_svdq_operator_factors(layer)
        self._emit_operator_factor_audit(layer)

    def _attach_raw_factor_aliases(self, layer: torch.nn.Module) -> None:
        for loader_name, raw_name in SVDQ_FACTOR_TO_RAW_NAME.items():
            if not hasattr(layer, raw_name):
                object.__setattr__(layer, raw_name, getattr(layer, loader_name))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, (
            "Number of global experts mismatch (excluding redundancy): "
            f"router_logits.shape[1]={router_logits.shape[1]}, num_logical_experts={num_logical_experts}"
        )

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
            tid2eid=tid2eid,
        )

        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        w1 = [layer.w13_weight]
        w1_scale = [layer.w13_weight_scale]
        w2 = [layer.w2_weight]
        w2_scale = [layer.w2_weight_scale]
        w1_scale_bias = [layer.w13_scale_bias.detach()] if hasattr(layer, "w13_scale_bias") else None
        w2_scale_bias = [layer.w2_scale_bias.detach()] if hasattr(layer, "w2_scale_bias") else None

        return _EXTRA_CTX.moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                is_per_channel_weight=self.is_per_channel_weight,
                gate_up_svdq_l1=layer.gate_up_svdq_l1,
                gate_svdq_l2=layer.gate_svdq_l2,
                up_svdq_l2=layer.up_svdq_l2,
                down_svdq_l1=layer.down_svdq_l1,
                down_svdq_l2=layer.down_svdq_l2,
                gate_rank=layer.svdq_gate_rank,
                up_rank=layer.svdq_up_rank,
                down_rank=layer.svdq_down_rank,
                gate_rank_offset=layer.svdq_gate_rank_offset,
                up_rank_offset=layer.svdq_up_rank_offset,
                swiglu_limit=layer.swiglu_limit,
            )
        )

    def _validate_raw_factor_loads(self, layer: torch.nn.Module) -> None:
        expected = set(range(layer.local_num_experts))
        for param_name in SVDQ_FACTOR_SPECS:
            param = getattr(layer, param_name)
            loaded = getattr(param, "svdq_loaded_experts", None)
            if loaded is None:
                raise ValueError(f"SVDQ parameter {param_name} is missing load tracking metadata.")
            if loaded != expected:
                missing = sorted(expected - loaded)
                unexpected = sorted(loaded - expected)
                raise ValueError(
                    f"SVDQ factor {param_name} load completeness failed: "
                    f"missing local experts={missing}, unexpected local experts={unexpected}."
                )

    @staticmethod
    def _tensor_sha256(tensor: torch.Tensor) -> str:
        cpu = tensor.detach().contiguous().cpu()
        if cpu.dtype == torch.bfloat16:
            data = cpu.view(torch.uint16).numpy().tobytes()
        else:
            data = cpu.numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def _emit_raw_load_manifest(self, layer: torch.nn.Module) -> None:
        evidence_dir = os.environ.get("VLLM_ASCEND_SVDQ_EVIDENCE_DIR")
        if not evidence_dir:
            return
        os.makedirs(evidence_dir, exist_ok=True)
        entries = []
        for param_name, (projection, factor_kind, shard_dim, _) in SVDQ_FACTOR_SPECS.items():
            param = getattr(layer, param_name)
            raw_name = SVDQ_FACTOR_TO_RAW_NAME[param_name]
            for local_expert in range(layer.local_num_experts):
                global_expert = local_expert
                expert_map = getattr(layer, "expert_map", None)
                if expert_map is not None:
                    matches = (expert_map.detach().cpu() == local_expert).nonzero()
                    if matches.numel() > 0:
                        global_expert = int(matches[0].item())
                entries.append(
                    {
                        "checkpoint_key": (
                            f"{self.svdq_spec.checkpoint_prefix}.{global_expert}.{projection}.{factor_kind}"
                        ),
                        "global_expert_id": global_expert,
                        "local_expert_id": local_expert,
                        "destination_parameter": param_name,
                        "raw_role": raw_name,
                        "destination_shape": list(param.data[local_expert].shape),
                        "destination_dtype": str(param.dtype),
                        "tp_policy": "replicate" if shard_dim is None else f"shard_dim_{shard_dim}",
                        "checksum": self._tensor_sha256(param.data[local_expert]),
                        "load_status": "loaded_once",
                    }
                )
        layer_name = getattr(layer, "layer_name", "unknown").replace(".", "_")
        path = os.path.join(evidence_dir, f"{layer_name}_svdq_raw_load_manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"layer_name": getattr(layer, "layer_name", "unknown"), "entries": entries}, f, indent=2)

    @staticmethod
    def _emit_operator_factor_audit(layer: torch.nn.Module) -> None:
        evidence_dir = os.environ.get("VLLM_ASCEND_SVDQ_EVIDENCE_DIR")
        if evidence_dir:
            emit_svdq_operator_audit(layer, evidence_dir)
