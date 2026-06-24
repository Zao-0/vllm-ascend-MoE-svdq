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
from vllm_ascend.quantization.methods.w4a8 import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.svdq_spec import (
    SVDQ_FACTOR_SPECS,
    SVDQ_FACTOR_TO_RAW_NAME,
    build_svdq_moe_layer_spec,
)

DEFAULT_MODEL_PATH = "/root/workspace/lza/LLM/Qwen3.5-35B-A3B-W4A8-svdq-r64-mtp-canonical"
DEFAULT_EVIDENCE_DIR = "/root/workspace/lza/svdq_clean_evidence"

RESIDUAL_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RESIDUAL_KINDS = (
    "weight",
    "weight_scale",
    "weight_offset",
    "weight_scale_second",
    "weight_offset_second",
    "scale_bias",
)
RESIDUAL_OPERATOR_TENSORS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_scale_bias",
    "w2_scale_bias",
)


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


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().contiguous().cpu()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy().tobytes()
    return cpu.numpy().tobytes()


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


def _sampled_bytes_hex(data: bytes, *, sample_bytes: int = 64) -> dict[str, str]:
    if len(data) <= sample_bytes * 3:
        return {"all": data.hex()}
    midpoint = max((len(data) // 2) - (sample_bytes // 2), 0)
    return {
        "head": data[:sample_bytes].hex(),
        "middle": data[midpoint : midpoint + sample_bytes].hex(),
        "tail": data[-sample_bytes:].hex(),
    }


def _tensor_audit_metadata(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    data = _tensor_bytes(tensor)
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "logical_shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_size_bytes": _storage_nbytes(tensor),
        "storage_offset": int(tensor.storage_offset()),
        "element_size_bytes": int(tensor.element_size()),
        "numel": int(tensor.numel()),
        "npu_format": _tensor_npu_format(tensor),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sampled_bytes_hex": _sampled_bytes_hex(data),
    }


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


def _make_official_w4a8_method(
    quant_description: dict[str, Any],
    *,
    tp_size: int,
) -> AscendW4A8DynamicFusedMoEMethod:
    method = object.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.group_size = int(quant_description.get("group_size", 256))
    method.is_per_channel_weight = method.group_size == 0
    method.new_quant_version = quant_description.get("version", "0") == "1.0.0"
    method.quant_method = quant_description.get("ascend_quant_method", "") or ""
    if "weight_strategy" in quant_description:
        method.weight_strategy = quant_description.get("weight_strategy", "group")
    method.tp_size = int(tp_size)
    method.dynamic_eplb = False
    return method


def _ensure_minimal_ascend_config_for_official_postload() -> None:
    from vllm_ascend import ascend_config as ascend_config_module
    from vllm_ascend import envs as ascend_envs

    try:
        ascend_config_module.get_ascend_config()
        return
    except RuntimeError:
        pass

    ascend_config_module._ASCEND_CONFIG = SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(),
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        weight_nz_mode=ascend_envs.VLLM_ASCEND_ENABLE_NZ,
    )


def _require_npu_for_residual_parity() -> None:
    try:
        import torch_npu  # noqa: F401  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError("official W4A8 residual parity requires torch_npu.") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("official W4A8 residual parity requires an available NPU.")


def _make_residual_validation_layer(
    *,
    method: AscendW4A8DynamicFusedMoEMethod,
    spec: Any,
    tp_size: int,
    tp_rank: int,
) -> torch.nn.Module:
    if spec.intermediate_size % tp_size != 0:
        raise ValueError(f"intermediate size {spec.intermediate_size} is not divisible by tp_size={tp_size}.")
    intermediate_local = spec.intermediate_size // tp_size
    layer = torch.nn.Module()
    layer.layer_name = spec.prefix
    layer.local_num_experts = spec.num_experts
    layer.expert_map = None
    layer._expert_map = None
    layer.moe_parallel_config = SimpleNamespace(tp_rank=tp_rank)
    layer.swiglu_limit = 0.0

    param_dict = {}
    param_dict.update(method.get_weight(spec.num_experts, intermediate_local, spec.hidden_size, torch.float32))
    param_dict.update(
        method.get_dynamic_quant_param(spec.num_experts, intermediate_local, spec.hidden_size, torch.float32)
    )
    for param_name, tensor in param_dict.items():
        layer.register_parameter(param_name, torch.nn.Parameter(tensor.npu(), requires_grad=False))
    return layer


def _factor_checkpoint_keys(spec: Any) -> list[str]:
    keys = []
    for expert_id in range(spec.num_experts):
        for param_name, (projection, factor_kind, _, _) in SVDQ_FACTOR_SPECS.items():
            keys.append(f"{spec.checkpoint_prefix}.{expert_id}.{projection}.{factor_kind}")
    return keys


def _residual_checkpoint_keys(spec: Any, weight_map: dict[str, str]) -> list[str]:
    keys = []
    for expert_id in range(spec.num_experts):
        for projection in RESIDUAL_PROJECTIONS:
            for kind in RESIDUAL_KINDS:
                key = f"{spec.checkpoint_prefix}.{expert_id}.{projection}.{kind}"
                if key in weight_map:
                    keys.append(key)
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


def _residual_destination(
    *,
    projection: str,
    kind: str,
) -> tuple[str, int | None]:
    if projection == "down_proj":
        return f"w2_{kind}", None
    if projection == "gate_proj":
        return f"w13_{kind}", 0
    if projection == "up_proj":
        return f"w13_{kind}", 1
    raise ValueError(f"unsupported residual projection {projection!r}.")


def _load_residual_checkpoint_tensor(
    *,
    layer: torch.nn.Module,
    spec: Any,
    key: str,
    loaded_weight: torch.Tensor,
) -> dict[str, Any]:
    suffix = key.removeprefix(f"{spec.checkpoint_prefix}.")
    expert_text, projection, kind = suffix.split(".", 2)
    expert_id = int(expert_text)
    param_name, fused_slot = _residual_destination(projection=projection, kind=kind)
    if not hasattr(layer, param_name):
        raise KeyError(f"layer is missing residual parameter {param_name!r} for checkpoint key {key!r}.")

    target = getattr(layer, param_name).data[expert_id]
    if fused_slot is None:
        if tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"{key} shape mismatch for {param_name}: loaded={tuple(loaded_weight.shape)}, "
                f"target={tuple(target.shape)}."
            )
        target.copy_(loaded_weight)
        destination_slice = "full"
    else:
        rows = int(loaded_weight.shape[0])
        start = fused_slot * rows
        end = start + rows
        target_slice = target[start:end]
        if tuple(target_slice.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"{key} shape mismatch for {param_name}[{start}:{end}]: "
                f"loaded={tuple(loaded_weight.shape)}, target={tuple(target_slice.shape)}."
            )
        target_slice.copy_(loaded_weight)
        destination_slice = f"{start}:{end}"

    return {
        "checkpoint_key": key,
        "projection": projection,
        "kind": kind,
        "global_expert_id": expert_id,
        "destination_parameter": param_name,
        "destination_slice": destination_slice,
        "loaded_shape": list(loaded_weight.shape),
        "loaded_dtype": str(loaded_weight.dtype),
        "loaded_stride": list(loaded_weight.stride()),
        "loaded_checksum": _tensor_sha256(loaded_weight),
    }


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


def _emit_residual_parity_manifest(
    *,
    layer: torch.nn.Module,
    spec: Any,
    evidence_dir: str,
    official_audit: dict[str, Any],
) -> str:
    layer_name = spec.prefix.replace(".", "_")
    path = os.path.join(evidence_dir, f"{layer_name}_svdq_official_w4a8_residual_parity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"layer_name": spec.prefix, **official_audit}, f, indent=2)
    return path


def _compare_residual_operator_tensors(
    *,
    svdq_layer: torch.nn.Module,
    official_layer: torch.nn.Module,
) -> dict[str, Any]:
    entries = []
    for name in RESIDUAL_OPERATOR_TENSORS:
        if not hasattr(svdq_layer, name) or not hasattr(official_layer, name):
            entries.append(
                {
                    "name": name,
                    "present_in_svdq": hasattr(svdq_layer, name),
                    "present_in_official": hasattr(official_layer, name),
                    "exact_match": False,
                }
            )
            continue
        svdq_tensor = getattr(svdq_layer, name).detach()
        official_tensor = getattr(official_layer, name).detach()
        svdq_meta = _tensor_audit_metadata(f"svdq.{name}", svdq_tensor)
        official_meta = _tensor_audit_metadata(f"official_w4a8.{name}", official_tensor)
        metadata_equal = {
            field: svdq_meta[field] == official_meta[field]
            for field in (
                "dtype",
                "device",
                "logical_shape",
                "stride",
                "storage_offset",
                "element_size_bytes",
                "numel",
                "npu_format",
                "sha256",
                "sampled_bytes_hex",
            )
        }
        exact_match = bool(torch.equal(svdq_tensor.cpu(), official_tensor.cpu()))
        entries.append(
            {
                "name": name,
                "present_in_svdq": True,
                "present_in_official": True,
                "exact_match": exact_match,
                "metadata_equal": metadata_equal,
                "svdq": svdq_meta,
                "official_w4a8": official_meta,
            }
        )
    return {
        "passed": all(entry.get("exact_match", False) for entry in entries),
        "entries": entries,
    }


def _load_and_audit_residual_parity(
    *,
    model_path: str,
    evidence_dir: str,
    quant_description: dict[str, Any],
    weight_map: dict[str, str],
    spec: Any,
    tp_size: int,
    tp_rank: int,
) -> dict[str, Any]:
    _require_npu_for_residual_parity()
    method = _make_official_w4a8_method(quant_description, tp_size=tp_size)
    svdq_residual_layer = _make_residual_validation_layer(
        method=method,
        spec=spec,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    official_layer = _make_residual_validation_layer(
        method=method,
        spec=spec,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )

    residual_keys = _residual_checkpoint_keys(spec, weight_map)
    by_shard = _group_keys_by_shard(residual_keys, weight_map)
    load_records = []
    for shard, shard_keys in by_shard.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                loaded_weight = f.get_tensor(key)
                svdq_record = _load_residual_checkpoint_tensor(
                    layer=svdq_residual_layer,
                    spec=spec,
                    key=key,
                    loaded_weight=loaded_weight,
                )
                official_record = _load_residual_checkpoint_tensor(
                    layer=official_layer,
                    spec=spec,
                    key=key,
                    loaded_weight=loaded_weight,
                )
                if svdq_record != official_record:
                    raise ValueError(f"residual load record diverged for {key}.")
                load_records.append(svdq_record)

    pre_postload = _compare_residual_operator_tensors(
        svdq_layer=svdq_residual_layer,
        official_layer=official_layer,
    )
    _ensure_minimal_ascend_config_for_official_postload()
    method.process_weights_after_loading(svdq_residual_layer)
    method.process_weights_after_loading(official_layer)
    post_postload = _compare_residual_operator_tensors(
        svdq_layer=svdq_residual_layer,
        official_layer=official_layer,
    )
    audit = {
        "official_method": "vllm_ascend.quantization.methods.w4a8.AscendW4A8DynamicFusedMoEMethod",
        "official_post_load": (
            "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim"
            if method.quant_method == ""
            else "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_compressed_tensors"
        ),
        "svdq_residual_source": (
            "AscendW4A8SVDQFusedMoEMethod inherits official W4A8 residual "
            "allocation/loading/post-load; this audit constructs the residual "
            "subset with the official W4A8 method and compares it with an "
            "isolated official W4A8 layer after the same real checkpoint loads."
        ),
        "quant_description": {
            "version": quant_description.get("version"),
            "group_size": quant_description.get("group_size"),
            "ascend_quant_method": quant_description.get("ascend_quant_method"),
        },
        "num_residual_checkpoint_keys": len(residual_keys),
        "num_residual_load_records": len(load_records),
        "load_records_sample": load_records[:12],
        "pre_postload_parity": pre_postload,
        "post_postload_parity": post_postload,
    }
    audit["passed"] = (
        len(residual_keys) > 0
        and len(load_records) == len(residual_keys)
        and bool(pre_postload["passed"])
        and bool(post_postload["passed"])
    )
    manifest_path = _emit_residual_parity_manifest(
        layer=svdq_residual_layer,
        spec=spec,
        evidence_dir=evidence_dir,
        official_audit=audit,
    )
    audit["manifest_path"] = manifest_path
    return audit


def _validate_loaded_sets(layer: torch.nn.Module, spec: Any) -> dict[str, list[int]]:
    expected = set(range(spec.num_experts))
    loaded_summary = {}
    for param_name in SVDQ_FACTOR_SPECS:
        loaded = getattr(layer, param_name).svdq_loaded_experts
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
    residual_parity = _load_and_audit_residual_parity(
        model_path=model_path,
        evidence_dir=evidence_dir,
        quant_description=quant_description,
        weight_map=weight_map,
        spec=spec,
        tp_size=tp_size,
        tp_rank=tp_rank,
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
        "official_w4a8_residual_parity_passed": bool(residual_parity["passed"]),
        "official_w4a8_residual_parity_path": residual_parity["manifest_path"],
        "official_w4a8_residual_operator_tensors": [
            entry["name"] for entry in residual_parity["post_postload_parity"]["entries"]
        ],
        "audit_max_abs": audit["max_abs"],
        "bf16_stage_names": audit["bf16_stage_names"],
        "bf16_stage_max_abs": audit["bf16_stage_max_abs"],
        "bf16_stage_all_finite": audit["bf16_stage_all_finite"],
        "operator_contract": audit["operator_contract"],
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
            result["all_loads_successful"]
            and result["audit_passed"]
            and result["official_w4a8_residual_parity_passed"]
            for result in results
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
