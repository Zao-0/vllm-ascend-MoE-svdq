#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Launch SVDQLowRankDebugReadback and compare real-checkpoint BF16 stages."""

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
    _load_validation_layer,
    _make_inputs,
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
    build_svdq_bf16_stage_reference,
)
from vllm_ascend.utils import enable_custom_op  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default="phase_z_lowrank_debug_readback_probe_summary.json")
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
    parser.add_argument(
        "--require-accumulator-readback",
        action="store_true",
        help="Require FP32 accumulator buffers to match references. Use with a package built with "
        "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK=ON.",
    )
    return parser.parse_args()


def _stage_passed(error: dict[str, float | bool | int], *, max_abs_tol: float, mean_abs_tol: float) -> bool:
    return (
        bool(error["actual_finite"])
        and bool(error["expected_finite"])
        and bool(error["diff_finite"])
        and float(error["max_abs"]) <= max_abs_tol
        and float(error["mean_abs"]) <= mean_abs_tol
    )


def _make_expert_contiguous_inputs(
    *,
    layer: torch.nn.Module,
    spec: Any,
    layer_index: int,
    experts: list[int],
    num_tokens: int,
    seed: int,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, tuple[int, int]], dict[int, dict[str, torch.Tensor]]]:
    intermediate_size = int(layer.gate_svdq_l2.shape[1])
    expert_token_nums = torch.zeros(int(spec.num_experts), dtype=torch.int32)
    x_segments = []
    hidden_segments = []
    references: dict[int, dict[str, torch.Tensor]] = {}
    spans: dict[int, tuple[int, int]] = {}
    offset = 0
    for expert in sorted(experts):
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
        x_segments.append(x)
        hidden_segments.append(hidden)
        references[expert] = reference["reference"]
        spans[expert] = (offset, offset + num_tokens)
        expert_token_nums[expert] = num_tokens
        offset += num_tokens
    return (
        torch.cat(x_segments, dim=0),
        torch.cat(hidden_segments, dim=0),
        expert_token_nums,
        spans,
        references,
    )


def _launch_debug_readback(
    *,
    layer: torch.nn.Module,
    routed_x: torch.Tensor,
    hidden: torch.Tensor,
    expert_token_nums: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    enable_custom_op()
    op = getattr(torch.ops._C_ascend, "svdq_low_rank_debug_readback", None)
    if op is None:
        raise RuntimeError(
            "torch.ops._C_ascend.svdq_low_rank_debug_readback is not registered. "
            "Rebuild/install vllm-ascend after adding the debug binding."
        )

    outputs = op(
        routed_x.to(device=device, dtype=torch.bfloat16),
        hidden.to(device=device, dtype=torch.bfloat16),
        layer.gate_up_svdq_l1.to(device=device, dtype=torch.bfloat16),
        layer.gate_svdq_l2.to(device=device, dtype=torch.bfloat16),
        layer.up_svdq_l2.to(device=device, dtype=torch.bfloat16),
        layer.down_svdq_l1.to(device=device, dtype=torch.bfloat16),
        layer.down_svdq_l2.to(device=device, dtype=torch.bfloat16),
        expert_token_nums.to(device=device, dtype=torch.int32),
        int(layer.svdq_gate_rank),
        int(layer.svdq_up_rank),
        int(layer.svdq_down_rank),
        int(layer.svdq_gate_rank_offset),
        int(layer.svdq_up_rank_offset),
    )
    torch.npu.synchronize()
    gate_up_output, down_output, gate_up_accumulator, down_accumulator = outputs
    return {
        "gate_up_output": gate_up_output.detach().float().cpu(),
        "down_output": down_output.detach().float().cpu(),
        "gate_up_accumulator": gate_up_accumulator.detach().float().cpu(),
        "down_accumulator": down_accumulator.detach().float().cpu(),
    }


def _compare_expert(
    *,
    layer: torch.nn.Module,
    expert: int,
    span: tuple[int, int],
    actual: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    max_abs_tol: float,
    mean_abs_tol: float,
    require_accumulator_readback: bool,
) -> dict[str, Any]:
    start, end = span
    intermediate_size = int(layer.gate_svdq_l2.shape[1])
    expected_gate_up = torch.cat(
        (
            reference["gate_l2_output"].detach().float().cpu(),
            reference["up_l2_output"].detach().float().cpu(),
        ),
        dim=1,
    )
    expected_down = reference["down_l2_output"].detach().float().cpu()

    comparisons = {
        "gate_l2_output": _stage_error(
            actual["gate_up_output"][start:end, :intermediate_size],
            expected_gate_up[:, :intermediate_size],
        ),
        "up_l2_output": _stage_error(
            actual["gate_up_output"][start:end, intermediate_size:],
            expected_gate_up[:, intermediate_size:],
        ),
        "down_l2_output": _stage_error(actual["down_output"][start:end], expected_down),
    }
    if require_accumulator_readback:
        comparisons["gate_up_accumulator"] = _stage_error(
            actual["gate_up_accumulator"][start:end],
            expected_gate_up,
        )
        comparisons["down_accumulator"] = _stage_error(
            actual["down_accumulator"][start:end],
            expected_down,
        )
    else:
        comparisons["gate_up_accumulator_finite"] = {
            "actual_finite": bool(torch.isfinite(actual["gate_up_accumulator"][start:end]).all().item()),
            "numel": int(actual["gate_up_accumulator"][start:end].numel()),
        }
        comparisons["down_accumulator_finite"] = {
            "actual_finite": bool(torch.isfinite(actual["down_accumulator"][start:end]).all().item()),
            "numel": int(actual["down_accumulator"][start:end].numel()),
        }

    required_names = (
        "gate_l2_output",
        "up_l2_output",
        "down_l2_output",
        "gate_up_accumulator",
        "down_accumulator",
    ) if require_accumulator_readback else ("gate_l2_output", "up_l2_output", "down_l2_output")
    passed_by_stage = {
        name: _stage_passed(comparisons[name], max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol)
        for name in required_names
    }
    return {
        "expert": expert,
        "span": list(span),
        "stage_errors": comparisons,
        "stage_passed": passed_by_stage,
        "passed": all(passed_by_stage.values()),
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
    require_accumulator_readback: bool,
) -> dict[str, Any]:
    layer, spec, load_count = _load_validation_layer(
        model_path=model_path,
        quant_description=quant_description,
        weight_map=weight_map,
        layer_index=layer_index,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    routed_x, hidden, expert_token_nums, spans, references = _make_expert_contiguous_inputs(
        layer=layer,
        spec=spec,
        layer_index=layer_index,
        experts=experts,
        num_tokens=num_tokens,
        seed=seed,
        input_scale=input_scale,
    )
    actual = _launch_debug_readback(
        layer=layer,
        routed_x=routed_x,
        hidden=hidden,
        expert_token_nums=expert_token_nums,
        device=device,
    )
    expert_results = [
        _compare_expert(
            layer=layer,
            expert=expert,
            span=spans[expert],
            actual=actual,
            reference=references[expert],
            max_abs_tol=max_abs_tol,
            mean_abs_tol=mean_abs_tol,
            require_accumulator_readback=require_accumulator_readback,
        )
        for expert in sorted(experts)
    ]
    return {
        "layer_index": layer_index,
        "layer_name": spec.prefix,
        "num_load_records": load_count,
        "num_tokens_per_expert": num_tokens,
        "expert_token_nums_nonzero": {
            str(expert): int(expert_token_nums[expert].item()) for expert in sorted(experts)
        },
        "rank_metadata": {
            "gate_rank": int(layer.svdq_gate_rank),
            "up_rank": int(layer.svdq_up_rank),
            "down_rank": int(layer.svdq_down_rank),
            "gate_rank_offset": int(layer.svdq_gate_rank_offset),
            "up_rank_offset": int(layer.svdq_up_rank_offset),
        },
        "output_shapes": {name: list(tensor.shape) for name, tensor in actual.items()},
        "experts": expert_results,
        "passed": all(result["passed"] for result in expert_results),
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.evidence_dir, exist_ok=True)
    summary_path = os.path.join(args.evidence_dir, args.summary_name)
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
            require_accumulator_readback=args.require_accumulator_readback,
        )
        for layer_index in args.layers
    ]
    summary = {
        "model_path": args.model_path,
        "evidence_dir": args.evidence_dir,
        "layers": args.layers,
        "experts": args.experts,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "device": str(device),
        "max_abs_tol": args.max_abs_tol,
        "mean_abs_tol": args.mean_abs_tol,
        "require_accumulator_readback": args.require_accumulator_readback,
        "npu_environment": npu_env,
        "results": results,
        "passed": all(result["passed"] for result in results),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
