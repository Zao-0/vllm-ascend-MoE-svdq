#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate official BF16 routing before SVDQ BF16 stage comparisons."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from svdq_bf16_stage_device_probe import (  # noqa: E402
    _compute_npu_stages,
    _load_validation_layer,
    _npu_environment,
    _stage_error,
)
from svdq_loader_pre_kernel_validate import (  # noqa: E402
    DEFAULT_EVIDENCE_DIR,
    DEFAULT_MODEL_PATH,
    _read_json,
    _weight_map,
)

from vllm_ascend.quantization.methods.svdq_post_load import (  # noqa: E402
    SVDQ_BF16_DEBUG_STAGE_NAMES,
    build_svdq_bf16_stage_reference,
)
from vllm_ascend.utils import enable_custom_op  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default="phase_t_bf16_routing_stage_probe_summary.json")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 39])
    parser.add_argument("--experts", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--input-scale", type=float, default=0.03125)
    parser.add_argument("--max-abs-tol", type=float, default=0.5)
    parser.add_argument("--mean-abs-tol", type=float, default=0.02)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _make_routing_inputs(
    *,
    hidden_size: int,
    num_tokens: int,
    top_k: int,
    experts: list[int],
    seed: int,
    layer_index: int,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + layer_index * 1000)
    x = (torch.randn(num_tokens, hidden_size, generator=generator) * input_scale).to(torch.bfloat16)
    flattened = [experts[index % len(experts)] for index in range(num_tokens * top_k)]
    expert_idx = torch.tensor(flattened, dtype=torch.int32).reshape(num_tokens, top_k)
    return x, expert_idx


def _cpu_routing_golden(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    expert_num: int,
    active_expert_range: tuple[int, int],
) -> dict[str, torch.Tensor]:
    expert_start, expert_end = active_expert_range
    flat = expert_idx.detach().cpu().reshape(-1).to(torch.int64)
    valid = (flat >= expert_start) & (flat < expert_end)
    sortable = flat.clone()
    sortable[~valid] = torch.iinfo(torch.int64).max
    sorted_indices = torch.argsort(sortable, stable=True)
    active = int(valid.sum().item())
    sorted_active = sorted_indices[:active]
    counts = torch.bincount(flat[sorted_active] - expert_start, minlength=expert_end - expert_start).to(torch.int64)
    expanded_x = x.detach().cpu()[torch.div(sorted_active, expert_idx.shape[1], rounding_mode="floor")]
    expanded_row_idx = sorted_active.to(torch.int32)
    return {
        "expanded_x": expanded_x,
        "expanded_row_idx": expanded_row_idx,
        "expert_tokens": counts,
        "expanded_scale": torch.empty(active, dtype=torch.float32),
        "expert_start": torch.tensor(expert_start, dtype=torch.int64),
        "expert_end": torch.tensor(expert_end, dtype=torch.int64),
        "expert_num": torch.tensor(expert_num, dtype=torch.int64),
    }


def _run_official_routing(
    *,
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    expert_num: int,
    active_expert_range: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    enable_custom_op()
    x_npu = x.to(device)
    expert_idx_npu = expert_idx.to(device)
    expanded_x, expanded_row_idx, expert_tokens, expanded_scale = torch.ops._C_ascend.npu_moe_init_routing_custom(
        x_npu,
        expert_idx_npu,
        scale=None,
        offset=None,
        active_num=int(expert_idx.numel()),
        expert_capacity=-1,
        expert_num=expert_num,
        drop_pad_mode=0,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        quant_mode=-1,
        active_expert_range=list(active_expert_range),
        row_idx_type=1,
    )
    torch.npu.synchronize()
    return {
        "expanded_x": expanded_x.detach().cpu(),
        "expanded_row_idx": expanded_row_idx.detach().cpu(),
        "expert_tokens": expert_tokens.detach().cpu(),
        "expanded_scale": expanded_scale.detach().cpu(),
    }


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool | int]:
    return _stage_error(actual.detach().float(), expected.detach().float())


def _routing_errors(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    active = int(expected["expanded_x"].shape[0])
    expert_count = int(expected["expert_tokens"].shape[0])
    return {
        "expanded_x": _tensor_error(actual["expanded_x"][:active], expected["expanded_x"]),
        "expanded_row_idx_exact": bool(torch.equal(actual["expanded_row_idx"][:active], expected["expanded_row_idx"])),
        "expert_tokens_exact": bool(torch.equal(actual["expert_tokens"][:expert_count], expected["expert_tokens"])),
        "actual_expanded_row_idx": actual["expanded_row_idx"][:active].to(torch.int64).tolist(),
        "expected_expanded_row_idx": expected["expanded_row_idx"].to(torch.int64).tolist(),
        "actual_expert_tokens": actual["expert_tokens"][:expert_count].to(torch.int64).tolist(),
        "expected_expert_tokens": expected["expert_tokens"].to(torch.int64).tolist(),
    }


def _make_hidden_for_down(
    *,
    token_count: int,
    intermediate_size: int,
    seed: int,
    layer_index: int,
    expert: int,
    input_scale: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + layer_index * 1000 + expert)
    return (torch.randn(token_count, intermediate_size, generator=generator) * input_scale).to(torch.bfloat16)


def _compare_stages_for_routed_expert(
    *,
    layer: torch.nn.Module,
    layer_index: int,
    expert: int,
    routed_x: torch.Tensor,
    expert_offset: int,
    token_count: int,
    seed: int,
    input_scale: float,
    device: torch.device,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> dict[str, Any]:
    x_slice = routed_x[expert_offset : expert_offset + token_count].detach().cpu().to(torch.bfloat16)
    intermediate_size = int(layer.gate_svdq_l2.shape[1])
    hidden = _make_hidden_for_down(
        token_count=token_count,
        intermediate_size=intermediate_size,
        seed=seed,
        layer_index=layer_index,
        expert=expert,
        input_scale=input_scale,
    )
    reference = build_svdq_bf16_stage_reference(layer, expert, x=x_slice, hidden=hidden, num_tokens=token_count)
    actual = _compute_npu_stages(layer=layer, expert=expert, x=x_slice, hidden=hidden, device=device)
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
        "expert": expert,
        "expert_offset": expert_offset,
        "token_count": token_count,
        "rank_metadata": reference["rank_metadata"],
        "stage_shapes": reference["stage_shapes"],
        "stage_errors": stage_errors,
        "stage_passed": stage_passed,
        "passed": all(stage_passed.values()),
    }


def _probe_layer(
    *,
    model_path: str,
    quant_description: dict[str, Any],
    weights: dict[str, str],
    layer_index: int,
    experts: list[int],
    num_tokens: int,
    top_k: int,
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
        weight_map=weights,
        layer_index=layer_index,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    x, expert_idx = _make_routing_inputs(
        hidden_size=int(spec.hidden_size),
        num_tokens=num_tokens,
        top_k=top_k,
        experts=experts,
        seed=seed,
        layer_index=layer_index,
        input_scale=input_scale,
    )
    active_range = (0, int(spec.num_experts))
    expected_routing = _cpu_routing_golden(
        x,
        expert_idx,
        expert_num=int(spec.num_experts),
        active_expert_range=active_range,
    )
    actual_routing = _run_official_routing(
        x=x,
        expert_idx=expert_idx,
        expert_num=int(spec.num_experts),
        active_expert_range=active_range,
        device=device,
    )
    routing_errors = _routing_errors(actual_routing, expected_routing)
    routing_passed = (
        bool(routing_errors["expanded_x"]["actual_finite"])
        and bool(routing_errors["expanded_x"]["expected_finite"])
        and bool(routing_errors["expanded_x"]["diff_finite"])
        and float(routing_errors["expanded_x"]["max_abs"]) == 0.0
        and bool(routing_errors["expanded_row_idx_exact"])
        and bool(routing_errors["expert_tokens_exact"])
    )

    counts = expected_routing["expert_tokens"].tolist()
    offsets = torch.cumsum(torch.tensor([0] + counts[:-1], dtype=torch.int64), dim=0).tolist()
    stage_results = []
    for expert in experts:
        token_count = int(counts[expert])
        if token_count == 0:
            continue
        stage_results.append(
            _compare_stages_for_routed_expert(
                layer=layer,
                layer_index=layer_index,
                expert=expert,
                routed_x=actual_routing["expanded_x"],
                expert_offset=int(offsets[expert]),
                token_count=token_count,
                seed=seed,
                input_scale=input_scale,
                device=device,
                max_abs_tol=max_abs_tol,
                mean_abs_tol=mean_abs_tol,
            )
        )

    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "num_load_records": load_count,
        "routing": {
            "top_k": top_k,
            "num_tokens": num_tokens,
            "expert_idx": expert_idx.to(torch.int64).tolist(),
            "active_expert_range": list(active_range),
            "errors": routing_errors,
            "passed": routing_passed,
        },
        "stage_results": stage_results,
        "passed": routing_passed and all(result["passed"] for result in stage_results),
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.evidence_dir, exist_ok=True)
    npu_env = _npu_environment()
    device_count = int(npu_env.get("npu_device_count") or 0)
    can_run = bool(npu_env.get("npu_available")) and args.device_id < device_count
    summary_path = os.path.join(args.evidence_dir, args.summary_name)
    if not can_run:
        summary = {
            "model_path": args.model_path,
            "evidence_dir": args.evidence_dir,
            "passed": not args.require_npu,
            "skipped": True,
            "skip_reason": f"NPU device {args.device_id} unavailable; device_count={device_count}.",
            "npu_environment": npu_env,
        }
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
    results = [
        _probe_layer(
            model_path=args.model_path,
            quant_description=quant_description,
            weights=weights,
            layer_index=layer_index,
            experts=args.experts,
            num_tokens=args.num_tokens,
            top_k=args.top_k,
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
        "top_k": args.top_k,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "seed": args.seed,
        "input_scale": args.input_scale,
        "max_abs_tol": args.max_abs_tol,
        "mean_abs_tol": args.mean_abs_tol,
        "device_id": args.device_id,
        "npu_environment": npu_env,
        "skipped": False,
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
