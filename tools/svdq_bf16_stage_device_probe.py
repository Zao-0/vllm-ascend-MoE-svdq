#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Compare NPU BF16 SVDQ stage outputs against real-checkpoint CPU references."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
    _attach_raw_aliases,
    _factor_checkpoint_keys,
    _group_keys_by_shard,
    _load_factor_through_qwen_mapping,
    _make_mapping_probe_model,
    _make_validation_layer,
    _read_json,
    _validate_loaded_sets,
    _weight_map,
)

import vllm_ascend.patch.worker.patch_svdq_moe_loading  # noqa: E402,F401
from vllm_ascend.quantization.methods.svdq_post_load import (  # noqa: E402
    SVDQ_BF16_DEBUG_STAGE_NAMES,
    build_svdq_bf16_stage_reference,
    build_svdq_operator_factors,
)
from vllm_ascend.quantization.svdq_spec import (  # noqa: E402
    SVDQ_FACTOR_SPECS,
    build_svdq_moe_layer_spec,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default="phase_r_bf16_stage_device_probe_summary.json")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 39])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--input-scale", type=float, default=0.03125)
    parser.add_argument("--max-abs-tol", type=float, default=0.5)
    parser.add_argument("--mean-abs-tol", type=float, default=0.02)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _npu_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "has_torch_npu_attr": hasattr(torch, "npu"),
        "torch_npu_imported": False,
        "torch_npu_version": None,
        "npu_available": False,
        "npu_device_count": 0,
    }
    try:
        import torch_npu  # type: ignore[import-untyped]

        info["torch_npu_imported"] = True
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
    except Exception as exc:
        info["torch_npu_import_error"] = f"{type(exc).__name__}: {exc}"

    if hasattr(torch, "npu"):
        try:
            info["npu_available"] = bool(torch.npu.is_available())
            info["npu_device_count"] = int(torch.npu.device_count())
        except Exception as exc:
            info["npu_query_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _load_validation_layer(
    *,
    model_path: str,
    quant_description: dict[str, Any],
    weight_map: dict[str, str],
    layer_index: int,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.nn.Module, Any, int]:
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

    _validate_loaded_sets(layer, spec)
    _attach_raw_aliases(layer)
    build_svdq_operator_factors(layer)
    if not all(record["success"] for record in load_records):
        failed = [record["checkpoint_key"] for record in load_records if not record["success"]]
        raise ValueError(f"{spec.prefix} has failed SVDQ factor loads: {failed[:8]}.")
    return layer, spec, len(load_records)


def _make_inputs(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
    seed: int,
    layer_index: int,
    expert: int,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + layer_index * 1000 + expert)
    x = torch.randn(num_tokens, hidden_size, generator=generator) * input_scale
    hidden = torch.randn(num_tokens, intermediate_size, generator=generator) * input_scale
    return x.to(torch.bfloat16), hidden.to(torch.bfloat16)


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
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "actual_finite": actual_finite,
        "expected_finite": expected_finite,
        "diff_finite": diff_finite,
        "numel": int(actual.numel()),
    }


def _compute_npu_stages(
    *,
    layer: torch.nn.Module,
    expert: int,
    x: torch.Tensor,
    hidden: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    gate_rank = int(layer.svdq_gate_rank)
    up_rank = int(layer.svdq_up_rank)
    gate_offset = int(layer.svdq_gate_rank_offset)
    up_offset = int(layer.svdq_up_rank_offset)

    x_npu = x.to(device=device, dtype=torch.bfloat16)
    hidden_npu = hidden.to(device=device, dtype=torch.bfloat16)
    gate_up_l1 = layer.gate_up_svdq_l1[expert].to(device=device, dtype=torch.bfloat16)
    gate_l2 = layer.gate_svdq_l2[expert].to(device=device, dtype=torch.bfloat16)
    up_l2 = layer.up_svdq_l2[expert].to(device=device, dtype=torch.bfloat16)
    down_l1 = layer.down_svdq_l1[expert].to(device=device, dtype=torch.bfloat16)
    down_l2 = layer.down_svdq_l2[expert].to(device=device, dtype=torch.bfloat16)

    fused_rank = x_npu @ gate_up_l1.transpose(0, 1)
    gate_rank_state = fused_rank[:, gate_offset : gate_offset + gate_rank]
    up_rank_state = fused_rank[:, up_offset : up_offset + up_rank]
    gate_output = gate_rank_state @ gate_l2.transpose(0, 1)
    up_output = up_rank_state @ up_l2.transpose(0, 1)
    down_rank_state = hidden_npu @ down_l1.transpose(0, 1)
    down_output = down_rank_state @ down_l2.transpose(0, 1)
    torch.npu.synchronize()

    stages = {
        "routing_input": x_npu,
        "gate_up_l1_rank": fused_rank,
        "gate_rank_split": gate_rank_state,
        "up_rank_split": up_rank_state,
        "gate_l2_output": gate_output,
        "up_l2_output": up_output,
        "down_l1_rank": down_rank_state,
        "down_l2_output": down_output,
    }
    return {name: stages[name].detach().float().cpu() for name in SVDQ_BF16_DEBUG_STAGE_NAMES}


def _probe_expert(
    *,
    layer: torch.nn.Module,
    spec: Any,
    layer_index: int,
    expert: int,
    num_tokens: int,
    seed: int,
    input_scale: float,
    device: torch.device,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> dict[str, Any]:
    intermediate_size = int(layer.gate_svdq_l2.shape[1])
    x, hidden = _make_inputs(
        hidden_size=int(spec.hidden_size),
        intermediate_size=intermediate_size,
        num_tokens=num_tokens,
        seed=seed,
        layer_index=layer_index,
        expert=expert,
        input_scale=input_scale,
    )

    reference = build_svdq_bf16_stage_reference(layer, expert, x=x, hidden=hidden, num_tokens=num_tokens)
    actual = _compute_npu_stages(layer=layer, expert=expert, x=x, hidden=hidden, device=device)

    stage_errors = {
        name: _stage_error(actual[name], reference["reference"][name].detach().float().cpu())
        for name in SVDQ_BF16_DEBUG_STAGE_NAMES
    }
    stage_passed = {
        name: (
            bool(error["actual_finite"])
            and bool(error["expected_finite"])
            and bool(error["diff_finite"])
            and float(error["max_abs"]) <= max_abs_tol
            and float(error["mean_abs"]) <= mean_abs_tol
        )
        for name, error in stage_errors.items()
    }
    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "expert": expert,
        "stage_shapes": reference["stage_shapes"],
        "rank_metadata": reference["rank_metadata"],
        "stage_errors": stage_errors,
        "stage_passed": stage_passed,
        "passed": all(stage_passed.values()),
    }


def _probe_layer(
    *,
    model_path: str,
    quant_description: dict[str, Any],
    weight_map: dict[str, str],
    layer_index: int,
    experts: list[int],
    num_tokens: int,
    tp_size: int,
    tp_rank: int,
    seed: int,
    input_scale: float,
    device: torch.device,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> dict[str, Any]:
    layer, spec, load_count = _load_validation_layer(
        model_path=model_path,
        quant_description=quant_description,
        weight_map=weight_map,
        layer_index=layer_index,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    expert_results = [
        _probe_expert(
            layer=layer,
            spec=spec,
            layer_index=layer_index,
            expert=expert,
            num_tokens=num_tokens,
            seed=seed,
            input_scale=input_scale,
            device=device,
            max_abs_tol=max_abs_tol,
            mean_abs_tol=mean_abs_tol,
        )
        for expert in experts
    ]
    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "num_load_records": load_count,
        "factor_families": list(SVDQ_FACTOR_SPECS),
        "experts": expert_results,
        "passed": all(result["passed"] for result in expert_results),
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.evidence_dir, exist_ok=True)
    npu_env = _npu_environment()
    device_count = int(npu_env.get("npu_device_count") or 0)
    can_run = bool(npu_env.get("npu_available")) and args.device_id < device_count
    if not can_run:
        summary = {
            "model_path": args.model_path,
            "evidence_dir": args.evidence_dir,
            "passed": not args.require_npu,
            "skipped": True,
            "skip_reason": f"NPU device {args.device_id} unavailable; device_count={device_count}.",
            "npu_environment": npu_env,
        }
        summary_path = os.path.join(args.evidence_dir, args.summary_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        if args.require_npu:
            raise SystemExit(1)
        return

    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    quant_description = _read_json(os.path.join(args.model_path, "quant_model_description.json"))
    weights = _weight_map(args.model_path)
    layer_results = [
        _probe_layer(
            model_path=args.model_path,
            quant_description=quant_description,
            weight_map=weights,
            layer_index=layer_index,
            experts=args.experts,
            num_tokens=args.num_tokens,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            seed=args.seed,
            input_scale=args.input_scale,
            device=device,
            max_abs_tol=args.max_abs_tol,
            mean_abs_tol=args.mean_abs_tol,
        )
        for layer_index in args.layers
    ]
    summary = {
        "model_path": args.model_path,
        "evidence_dir": args.evidence_dir,
        "layers": args.layers,
        "experts": args.experts,
        "num_tokens": args.num_tokens,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "max_abs_tol": args.max_abs_tol,
        "mean_abs_tol": args.mean_abs_tol,
        "device_id": args.device_id,
        "npu_environment": npu_env,
        "skipped": False,
        "passed": all(result["passed"] for result in layer_results),
        "results": layer_results,
    }
    summary_path = os.path.join(args.evidence_dir, args.summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
