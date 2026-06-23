#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate Qwen3.5 ModelSlim W4A8-SVDQ factor loading before kernel entry.

This command intentionally stops before any production SVDQ operator call. It
uses the real checkpoint, the active vLLM expert-parameter mapping, and the
vLLM-Ascend SVDQ factor weight loaders to prove raw BF16 factor placement and
the five final operator-facing factors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

import vllm_ascend.patch.worker.patch_svdq_moe_loading  # noqa: F401
from vllm_ascend.quantization.methods.svdq_post_load import (
    audit_svdq_operator_factors,
    build_svdq_operator_factors,
)
from vllm_ascend.quantization.methods.svdq_weight_loader import make_svdq_factor_weight_loader
from vllm_ascend.quantization.svdq_spec import (
    SVDQ_FACTOR_SPECS,
    SVDQ_FACTOR_TO_RAW_NAME,
    build_svdq_moe_layer_spec,
)

DEFAULT_MODEL_PATH = "/root/workspace/lza/LLM/Qwen3.5-35B-A3B-W4A8-svdq-r64-mtp-canonical"
DEFAULT_EVIDENCE_DIR = "/root/workspace/lza/svdq_clean_evidence"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 39])
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--audit-max-experts", type=int, default=2)
    parser.add_argument("--audit-num-tokens", type=int, default=3)
    return parser.parse_args()


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _weight_map(model_path: str) -> dict[str, str]:
    return _read_json(os.path.join(model_path, "quant_model_weights.safetensors.index.json"))["weight_map"]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        data = cpu.view(torch.uint16).numpy().tobytes()
    else:
        data = cpu.numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _make_mapping_probe_model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.register_parameter("svdq_w1_l1", torch.nn.Parameter(torch.empty(1), requires_grad=False))
    return model


def _make_validation_layer(spec: Any, *, tp_size: int, tp_rank: int) -> torch.nn.Module:
    if spec.intermediate_size % tp_size != 0:
        raise ValueError(f"intermediate size {spec.intermediate_size} is not divisible by tp_size={tp_size}.")
    intermediate_local = spec.intermediate_size // tp_size
    layer = torch.nn.Module()
    layer.layer_name = spec.prefix
    layer.local_num_experts = spec.num_experts
    layer.expert_map = None
    layer._expert_map = None
    layer.moe_parallel_config = SimpleNamespace(tp_rank=tp_rank)

    shapes = {
        "svdq_w1_l1": (spec.num_experts, spec.gate_rank, spec.hidden_size),
        "svdq_w1_l2": (spec.num_experts, intermediate_local, spec.gate_rank),
        "svdq_w3_l1": (spec.num_experts, spec.up_rank, spec.hidden_size),
        "svdq_w3_l2": (spec.num_experts, intermediate_local, spec.up_rank),
        "svdq_w2_l1": (spec.num_experts, spec.down_rank, intermediate_local),
        "svdq_w2_l2": (spec.num_experts, spec.hidden_size, spec.down_rank),
    }
    for param_name, shape in shapes.items():
        _, _, shard_dim, rank_dim = SVDQ_FACTOR_SPECS[param_name]
        param = torch.nn.Parameter(torch.empty(shape, dtype=spec.lowrank_dtype), requires_grad=False)
        param.weight_loader = make_svdq_factor_weight_loader(layer)
        param.svdq_shard_id = param_name
        param.svdq_raw_name = SVDQ_FACTOR_TO_RAW_NAME[param_name]
        param.svdq_sharded_dim = shard_dim
        param.svdq_rank_dim = rank_dim
        param.svdq_rank = spec.rank_by_loader_name[param_name]
        param.svdq_dtype = spec.lowrank_dtype
        param.svdq_loaded_experts = set()
        layer.register_parameter(param_name, param)
    return layer


def _factor_checkpoint_keys(spec: Any) -> list[str]:
    keys = []
    for expert_id in range(spec.num_experts):
        for param_name, (projection, factor_kind, _, _) in SVDQ_FACTOR_SPECS.items():
            keys.append(f"{spec.checkpoint_prefix}.{expert_id}.{projection}.{factor_kind}")
    return keys


def _group_keys_by_shard(keys: Iterable[str], weight_map: dict[str, str]) -> dict[str, list[str]]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    return by_shard


def _load_factor_through_qwen_mapping(
    *,
    key: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, torch.nn.Parameter],
    expert_params_mapping: list[tuple[str, str, int, str]],
) -> dict[str, Any]:
    for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
        if weight_name not in key:
            continue
        name_mapped = key.replace(weight_name, param_name)
        if name_mapped not in params_dict:
            raise KeyError(f"mapped parameter {name_mapped!r} for checkpoint key {key!r} is not registered.")
        param = params_dict[name_mapped]
        success = param.weight_loader(
            param,
            loaded_weight,
            name_mapped,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        return {
            "checkpoint_key": key,
            "mapped_parameter": name_mapped,
            "shard_id": shard_id,
            "global_expert_id": expert_id,
            "success": bool(success),
        }
    raise KeyError(f"no expert mapping matched checkpoint key {key!r}.")


def _attach_raw_aliases(layer: torch.nn.Module) -> None:
    for loader_name, raw_name in SVDQ_FACTOR_TO_RAW_NAME.items():
        object.__setattr__(layer, raw_name, getattr(layer, loader_name))


def _emit_raw_manifest(
    *,
    layer: torch.nn.Module,
    spec: Any,
    evidence_dir: str,
    load_records: list[dict[str, Any]],
) -> str:
    records_by_key = {record["checkpoint_key"]: record for record in load_records}
    entries = []
    for param_name, (projection, factor_kind, shard_dim, _) in SVDQ_FACTOR_SPECS.items():
        param = getattr(layer, param_name)
        raw_name = SVDQ_FACTOR_TO_RAW_NAME[param_name]
        for expert_id in range(spec.num_experts):
            checkpoint_key = f"{spec.checkpoint_prefix}.{expert_id}.{projection}.{factor_kind}"
            record = records_by_key[checkpoint_key]
            entries.append(
                {
                    "checkpoint_key": checkpoint_key,
                    "global_expert_id": expert_id,
                    "local_expert_id": expert_id,
                    "destination_parameter": param_name,
                    "mapped_parameter": record["mapped_parameter"],
                    "raw_role": raw_name,
                    "destination_shape": list(param.data[expert_id].shape),
                    "destination_dtype": str(param.dtype),
                    "tp_policy": "replicate" if shard_dim is None else f"shard_dim_{shard_dim}",
                    "checksum": _tensor_sha256(param.data[expert_id]),
                    "load_status": "loaded_once" if record["success"] else "not_loaded",
                }
            )

    layer_name = spec.prefix.replace(".", "_")
    path = os.path.join(evidence_dir, f"{layer_name}_svdq_raw_load_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"layer_name": spec.prefix, "entries": entries}, f, indent=2)
    return path


def _validate_loaded_sets(layer: torch.nn.Module, spec: Any) -> dict[str, list[int]]:
    expected = set(range(spec.num_experts))
    loaded_summary = {}
    for param_name in SVDQ_FACTOR_SPECS:
        loaded = getattr(getattr(layer, param_name), "svdq_loaded_experts")
        if loaded != expected:
            raise ValueError(
                f"{spec.prefix}.{param_name} incomplete: "
                f"missing={sorted(expected - loaded)}, unexpected={sorted(loaded - expected)}"
            )
        loaded_summary[param_name] = sorted(loaded)
    return loaded_summary


def _validate_layer(
    *,
    model_path: str,
    evidence_dir: str,
    quant_description: dict[str, Any],
    weight_map: dict[str, str],
    layer_index: int,
    tp_size: int,
    tp_rank: int,
    audit_max_experts: int,
    audit_num_tokens: int,
) -> dict[str, Any]:
    prefix = f"model.language_model.layers.{layer_index}.mlp.experts"
    spec = build_svdq_moe_layer_spec(
        quant_description=quant_description,
        prefix=prefix,
        model_path=model_path,
        num_experts=256,
        hidden_size=2048,
        intermediate_size=512,
    )
    layer = _make_validation_layer(spec, tp_size=tp_size, tp_rank=tp_rank)
    params_dict = {f"{spec.prefix}.{name}": param for name, param in layer.named_parameters()}

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        _make_mapping_probe_model(),
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=spec.num_experts,
    )
    factor_keys = _factor_checkpoint_keys(spec)
    by_shard = _group_keys_by_shard(factor_keys, weight_map)
    load_records = []
    for shard, shard_keys in by_shard.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                load_records.append(
                    _load_factor_through_qwen_mapping(
                        key=key,
                        loaded_weight=f.get_tensor(key),
                        params_dict=params_dict,
                        expert_params_mapping=expert_params_mapping,
                    )
                )

    loaded_summary = _validate_loaded_sets(layer, spec)
    _attach_raw_aliases(layer)
    build_svdq_operator_factors(layer)
    audit = audit_svdq_operator_factors(
        layer,
        max_experts=audit_max_experts,
        num_tokens=audit_num_tokens,
    )
    raw_manifest_path = _emit_raw_manifest(
        layer=layer,
        spec=spec,
        evidence_dir=evidence_dir,
        load_records=load_records,
    )

    layer_name = spec.prefix.replace(".", "_")
    operator_audit_path = os.path.join(evidence_dir, f"{layer_name}_svdq_operator_factor_audit.json")
    with open(operator_audit_path, "w", encoding="utf-8") as f:
        json.dump({"layer_name": spec.prefix, **audit}, f, indent=2)

    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "raw_manifest_path": raw_manifest_path,
        "operator_audit_path": operator_audit_path,
        "num_factor_keys": len(factor_keys),
        "num_load_records": len(load_records),
        "all_loads_successful": all(record["success"] for record in load_records),
        "loaded_expert_counts": {name: len(experts) for name, experts in loaded_summary.items()},
        "audit_passed": bool(audit["passed"]),
        "audit_max_abs": audit["max_abs"],
        "rank_metadata": audit["rank_metadata"],
        "mapping_first_factor_entries": [
            mapping for mapping in expert_params_mapping[:6] if mapping[3] in SVDQ_FACTOR_SPECS
        ],
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.evidence_dir, exist_ok=True)
    quant_description = _read_json(os.path.join(args.model_path, "quant_model_description.json"))
    weight_map = _weight_map(args.model_path)

    results = [
        _validate_layer(
            model_path=args.model_path,
            evidence_dir=args.evidence_dir,
            quant_description=quant_description,
            weight_map=weight_map,
            layer_index=layer_index,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            audit_max_experts=args.audit_max_experts,
            audit_num_tokens=args.audit_num_tokens,
        )
        for layer_index in args.layers
    ]

    summary = {
        "model_path": args.model_path,
        "evidence_dir": args.evidence_dir,
        "layers": args.layers,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "passed": all(
            result["all_loads_successful"] and result["audit_passed"] for result in results
        ),
        "results": results,
    }
    summary_path = os.path.join(args.evidence_dir, "phase_bc_real_loader_pre_kernel_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
