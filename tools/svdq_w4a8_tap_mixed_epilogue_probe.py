#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate SVDQ mixed epilogues using official W4A8 debug taps.

This probe is intentionally isolated from production
``DispatchFFNCombineW4A8SVDQ``. It uses the official
``dispatch_ffn_combine_w4_a8`` debug readback as the W4A8 source of truth,
loads real checkpoint SVDQ factors, computes BF16 low-rank outputs through the
SVDQ low-rank debug op, and feeds official residual taps plus BF16 low-rank
outputs into the mixed-epilogue debug op.
"""

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

from svdq_bf16_stage_device_probe import _load_validation_layer  # noqa: E402
from svdq_loader_pre_kernel_validate import DEFAULT_EVIDENCE_DIR, DEFAULT_MODEL_PATH, _read_json, _weight_map  # noqa: E402
from svdq_lowrank_debug_readback_probe import _launch_debug_readback as _launch_lowrank_debug  # noqa: E402
from svdq_mixed_epilogue_device_probe import _run_npu_mixed_epilogue  # noqa: E402
from svdq_w4a8_debug_readback_probe import _destroy_hccl_if_needed, _init_single_rank_hccl  # noqa: E402
from svdq_w4a8_debug_readback_real_checkpoint_probe import (  # noqa: E402
    _has_registered_debug_op,
    _load_real_residual_layer,
    _local_num_experts,
    _make_expert_idx,
    _make_input,
    _npu_environment,
    _official_debug_symbol_status,
    _official_gmm1_unfused_reference,
    _official_gmm2_unfused_reference,
    _routed_experts,
)

from vllm_ascend.quantization.methods.svdq_post_load import build_svdq_mixed_epilogue_reference  # noqa: E402

DEFAULT_SUMMARY_NAME = "phase_w4a8_tap_mixed_epilogue_summary.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=Path(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--route-experts", type=int, nargs="*", default=None)
    parser.add_argument("--local-num-experts", type=int, default=None)
    parser.add_argument("--max-output-size", type=int, default=64)
    parser.add_argument("--input-mode", choices=("linear",), default="linear")
    parser.add_argument("--linear-input-scale", type=float, default=0.01)
    parser.add_argument("--mixed-max-abs-tol", type=float, default=0.02)
    parser.add_argument("--mixed-mean-abs-tol", type=float, default=0.002)
    parser.add_argument("--scale-tol", type=float, default=1e-7)
    parser.add_argument("--hidden-q-mismatch-count-tol", type=int, default=2)
    parser.add_argument("--hidden-q-max-abs-diff-tol", type=int, default=1)
    parser.add_argument("--w4a8-reference-max-rows", type=int, default=64)
    parser.add_argument("--gmm1-reference-max-abs-tol", type=float, default=2e-2)
    parser.add_argument("--gmm1-reference-mean-abs-tol", type=float, default=2e-3)
    parser.add_argument("--gmm2-reference-max-abs-tol", type=float, default=2e-4)
    parser.add_argument("--gmm2-reference-mean-abs-tol", type=float, default=2e-5)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    diff = (actual_cpu.float() - expected_cpu.float()).abs()
    diff_finite = bool(torch.isfinite(diff).all().item()) if diff.numel() else True
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual_cpu.dtype),
        "expected_dtype": str(expected_cpu.dtype),
        "actual_finite": bool(torch.isfinite(actual_cpu.float()).all().item()) if actual_cpu.numel() else True,
        "expected_finite": bool(torch.isfinite(expected_cpu.float()).all().item()) if expected_cpu.numel() else True,
        "diff_finite": diff_finite,
        "max_abs": float(diff.max().item()) if diff.numel() and diff_finite else float("inf"),
        "mean_abs": float(diff.mean().item()) if diff.numel() and diff_finite else float("inf"),
        "numel": int(diff.numel()),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().float()
    finite = bool(torch.isfinite(cpu).all().item()) if cpu.numel() else True
    abs_cpu = torch.nan_to_num(cpu).abs()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "max_abs": float(abs_cpu.max().item()) if abs_cpu.numel() else 0.0,
        "mean_abs": float(abs_cpu.mean().item()) if abs_cpu.numel() else 0.0,
        "nan_count": int(torch.isnan(cpu).sum().item()),
        "inf_count": int(torch.isinf(cpu).sum().item()),
    }


def _nonzero_finite(stats: dict[str, Any]) -> bool:
    return bool(stats["finite"] and float(stats["max_abs"]) > 0.0)


def _stage_passed(error: dict[str, Any], *, max_abs_tol: float, mean_abs_tol: float) -> bool:
    return (
        bool(error["actual_finite"])
        and bool(error["expected_finite"])
        and bool(error["diff_finite"])
        and float(error["max_abs"]) <= max_abs_tol
        and float(error["mean_abs"]) <= mean_abs_tol
    )


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _active_expert_counts(expert_token_nums: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    counts = expert_token_nums.detach().cpu().reshape(-1).to(torch.int64)
    active_experts = [int(index) for index, count in enumerate(counts.tolist()) if int(count) > 0]
    return active_experts, counts


def _group_routed_x_by_expert(x: torch.Tensor, counts: torch.Tensor, active_experts: list[int]) -> torch.Tensor:
    rows = []
    x_cpu = x.detach().cpu().to(torch.bfloat16)
    for expert in active_experts:
        count = int(counts[expert].item())
        if count > int(x_cpu.shape[0]):
            raise ValueError(f"expert {expert} requested {count} rows, but input has {x_cpu.shape[0]}.")
        rows.append(x_cpu[:count])
    if not rows:
        return torch.empty((0, x_cpu.shape[1]), dtype=torch.bfloat16)
    return torch.cat(rows, dim=0).contiguous()


def _expand_counts_for_svdq(counts: torch.Tensor, num_experts: int) -> torch.Tensor:
    expanded = torch.zeros(num_experts, dtype=torch.int32)
    limit = min(num_experts, int(counts.numel()))
    expanded[:limit] = counts[:limit].to(torch.int32)
    return expanded


def _run_w4a8_debug(
    *,
    args: argparse.Namespace,
    group: str,
    residual_layer: torch.nn.Module,
    spec: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x = _make_input(args, spec.hidden_size, device=device)
    expert_idx = _make_expert_idx(_routed_experts(args), args.num_tokens, device=device)
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool, device=device)
    op = torch.ops._C_ascend.svdq_w4a8_debug_readback
    (
        out,
        expert_token_nums,
        routed_x_int8,
        routed_x_scale,
        gmm1_post_dequant,
        gmm1_hidden_prequant,
        hidden_x_int4_packed,
        hidden_x_scale,
        gmm2_post_dequant,
    ) = op(
        x,
        [residual_layer.w13_weight],
        [residual_layer.w2_weight],
        expert_idx,
        [residual_layer.w13_weight_scale],
        [residual_layer.w2_weight_scale],
        [residual_layer.w13_scale_bias],
        [residual_layer.w2_scale_bias],
        probs,
        group,
        args.max_output_size,
        x_active_mask,
    )
    torch.npu.synchronize()
    return {
        "x": x.detach().cpu(),
        "out": out.detach().cpu(),
        "expert_token_nums": expert_token_nums.detach().cpu(),
        "routed_x_int8": routed_x_int8.detach().cpu(),
        "routed_x_scale": routed_x_scale.detach().cpu(),
        "gmm1_post_dequant": gmm1_post_dequant.detach().cpu(),
        "gmm1_hidden_prequant": gmm1_hidden_prequant.detach().cpu(),
        "hidden_x_int4_packed": hidden_x_int4_packed.detach().cpu(),
        "hidden_x_scale": hidden_x_scale.detach().cpu(),
        "gmm2_post_dequant": gmm2_post_dequant.detach().cpu(),
    }


def _w4a8_reference_checks(
    *,
    args: argparse.Namespace,
    residual_layer: torch.nn.Module,
    spec: Any,
    taps: dict[str, torch.Tensor],
    active_rows: int,
) -> dict[str, Any]:
    reference1, contract1 = _official_gmm1_unfused_reference(
        routed_x_int8=taps["routed_x_int8"][:active_rows],
        routed_x_scale=taps["routed_x_scale"][:active_rows],
        weight=residual_layer.w13_weight,
        weight_scale=residual_layer.w13_weight_scale,
        scale_bias=residual_layer.w13_scale_bias,
        expert_token_nums=taps["expert_token_nums"],
        output_columns=2 * spec.intermediate_size,
        max_rows=args.w4a8_reference_max_rows,
    )
    error1 = _tensor_error(taps["gmm1_post_dequant"][: reference1.shape[0], : reference1.shape[1]], reference1)
    reference2, contract2 = _official_gmm2_unfused_reference(
        hidden_x_int4_packed=taps["hidden_x_int4_packed"][:active_rows],
        hidden_x_scale=taps["hidden_x_scale"][:active_rows],
        weight=residual_layer.w2_weight,
        weight_scale=residual_layer.w2_weight_scale,
        scale_bias=residual_layer.w2_scale_bias,
        expert_token_nums=taps["expert_token_nums"],
        output_columns=spec.hidden_size,
        max_rows=args.w4a8_reference_max_rows,
    )
    error2 = _tensor_error(taps["gmm2_post_dequant"][: reference2.shape[0], : reference2.shape[1]], reference2)
    gmm1_passed = _stage_passed(
        error1,
        max_abs_tol=args.gmm1_reference_max_abs_tol,
        mean_abs_tol=args.gmm1_reference_mean_abs_tol,
    )
    gmm2_passed = _stage_passed(
        error2,
        max_abs_tol=args.gmm2_reference_max_abs_tol,
        mean_abs_tol=args.gmm2_reference_mean_abs_tol,
    )
    return {
        "gmm1": {
            "passed": gmm1_passed,
            "contract": contract1,
            "error": error1,
            "max_abs_tolerance": args.gmm1_reference_max_abs_tol,
            "mean_abs_tolerance": args.gmm1_reference_mean_abs_tol,
        },
        "gmm2": {
            "passed": gmm2_passed,
            "contract": contract2,
            "error": error2,
            "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
            "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
        },
        "passed": bool(gmm1_passed and gmm2_passed),
    }


def _run_combined_probe(args: argparse.Namespace, group: str) -> dict[str, Any]:
    device = torch.device(f"npu:{args.device_id}")
    routed_experts = _routed_experts(args)
    local_num_experts = _local_num_experts(args, routed_experts)
    quant_description = _read_json(os.path.join(args.model_path, "quant_model_description.json"))
    weights = _weight_map(args.model_path)
    residual_layer, residual_spec, residual_key_count = _load_real_residual_layer(
        model_path=args.model_path,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
        routed_experts=set(routed_experts),
        local_num_experts=local_num_experts,
    )
    svdq_layer, svdq_spec, factor_load_count = _load_validation_layer(
        model_path=args.model_path,
        quant_description=quant_description,
        weight_map=weights,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
    )
    if residual_spec.hidden_size != svdq_spec.hidden_size or residual_spec.intermediate_size != svdq_spec.intermediate_size:
        raise ValueError("residual and SVDQ specs disagree on hidden/intermediate size.")

    taps = _run_w4a8_debug(
        args=args,
        group=group,
        residual_layer=residual_layer,
        spec=residual_spec,
        device=device,
    )
    active_rows = args.num_tokens * args.top_k
    active_experts, counts = _active_expert_counts(taps["expert_token_nums"])
    routed_x_grouped = _group_routed_x_by_expert(taps["x"], counts, active_experts)
    expert_token_nums_svdq = _expand_counts_for_svdq(counts, int(svdq_spec.num_experts))
    hidden_placeholder = torch.zeros((active_rows, svdq_spec.intermediate_size), dtype=torch.bfloat16)
    lowrank_gate = _launch_lowrank_debug(
        layer=svdq_layer,
        routed_x=routed_x_grouped,
        hidden=hidden_placeholder,
        expert_token_nums=expert_token_nums_svdq,
        device=device,
    )
    gate_up_output = lowrank_gate["gate_up_output"][:active_rows]
    gate_up_accumulator = lowrank_gate["gate_up_accumulator"][:active_rows]
    gate_lowrank = gate_up_output[:, : svdq_spec.intermediate_size]
    up_lowrank = gate_up_output[:, svdq_spec.intermediate_size :]
    first_reference = build_svdq_mixed_epilogue_reference(
        residual_gate_up=taps["gmm1_post_dequant"][:active_rows].float(),
        gate_lowrank=gate_lowrank.to(torch.bfloat16),
        up_lowrank=up_lowrank.to(torch.bfloat16),
        residual_down=torch.zeros((active_rows, svdq_spec.hidden_size), dtype=torch.float32),
        down_lowrank=torch.zeros((active_rows, svdq_spec.hidden_size), dtype=torch.bfloat16),
        swiglu_limit=args.swiglu_limit,
    )
    hidden_bf16 = first_reference["stages"]["hidden_bf16"]
    lowrank_down = _launch_lowrank_debug(
        layer=svdq_layer,
        routed_x=routed_x_grouped,
        hidden=hidden_bf16,
        expert_token_nums=expert_token_nums_svdq,
        device=device,
    )
    down_output = lowrank_down["down_output"][:active_rows]
    down_accumulator = lowrank_down["down_accumulator"][:active_rows]
    down_lowrank = down_output
    mixed_actual = _run_npu_mixed_epilogue(
        inputs={
            "residual_gate_up": taps["gmm1_post_dequant"][:active_rows].float(),
            "gate_lowrank": gate_lowrank.to(torch.bfloat16),
            "up_lowrank": up_lowrank.to(torch.bfloat16),
            "residual_down": taps["gmm2_post_dequant"][:active_rows].float(),
            "down_lowrank": down_lowrank.to(torch.bfloat16),
        },
        device=device,
        swiglu_limit=args.swiglu_limit,
    )
    mixed_reference = build_svdq_mixed_epilogue_reference(
        residual_gate_up=taps["gmm1_post_dequant"][:active_rows].float(),
        gate_lowrank=gate_lowrank.to(torch.bfloat16),
        up_lowrank=up_lowrank.to(torch.bfloat16),
        residual_down=taps["gmm2_post_dequant"][:active_rows].float(),
        down_lowrank=down_lowrank.to(torch.bfloat16),
        swiglu_limit=args.swiglu_limit,
    )
    stage_errors = {
        "gate_mixed": _tensor_error(mixed_actual["gate_mixed"], mixed_reference["stages"]["gate_mixed"]),
        "up_mixed": _tensor_error(mixed_actual["up_mixed"], mixed_reference["stages"]["up_mixed"]),
        "hidden_bf16": _tensor_error(mixed_actual["hidden_bf16"], mixed_reference["stages"]["hidden_bf16"]),
        "hidden_scale": _tensor_error(mixed_actual["hidden_scale"], mixed_reference["stages"]["hidden_scale"]),
        "down_mixed": _tensor_error(mixed_actual["down_mixed"], mixed_reference["stages"]["down_mixed"]),
        "out_bf16": _tensor_error(mixed_actual["out_bf16"], mixed_reference["stages"]["down_mixed"].to(torch.bfloat16)),
    }
    q_diff = (mixed_actual["hidden_q"].to(torch.int16) - mixed_reference["stages"]["hidden_q"].to(torch.int16)).abs()
    hidden_q_mismatch_count = int((q_diff != 0).sum().item())
    hidden_q_max_abs_diff = int(q_diff.max().item()) if q_diff.numel() else 0
    hidden_q_passed = (
        hidden_q_mismatch_count <= args.hidden_q_mismatch_count_tol
        and hidden_q_max_abs_diff <= args.hidden_q_max_abs_diff_tol
    )
    stage_passed = {
        "gate_mixed": _stage_passed(
            stage_errors["gate_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "up_mixed": _stage_passed(
            stage_errors["up_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "hidden_bf16": _stage_passed(
            stage_errors["hidden_bf16"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "hidden_scale": (
            bool(stage_errors["hidden_scale"]["actual_finite"])
            and bool(stage_errors["hidden_scale"]["expected_finite"])
            and bool(stage_errors["hidden_scale"]["diff_finite"])
            and float(stage_errors["hidden_scale"]["max_abs"]) <= args.scale_tol
        ),
        "hidden_q": hidden_q_passed,
        "down_mixed": _stage_passed(
            stage_errors["down_mixed"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
        "out_bf16": _stage_passed(
            stage_errors["out_bf16"],
            max_abs_tol=args.mixed_max_abs_tol,
            mean_abs_tol=args.mixed_mean_abs_tol,
        ),
    }
    w4a8_reference = _w4a8_reference_checks(
        args=args,
        residual_layer=residual_layer,
        spec=residual_spec,
        taps=taps,
        active_rows=active_rows,
    )
    lowrank_stats = {
        "source_for_mixed_epilogue": "bf16_lowrank_output_readback",
        "bf16_output_used_for_mixed_epilogue": True,
        "gate_up_output": _tensor_stats(gate_up_output),
        "gate_up_accumulator": _tensor_stats(gate_up_accumulator),
        "down_output": _tensor_stats(down_output),
        "down_accumulator": _tensor_stats(down_accumulator),
    }
    lowrank_output_health_passed = bool(
        _nonzero_finite(lowrank_stats["gate_up_output"]) and _nonzero_finite(lowrank_stats["down_output"])
    )
    return {
        "stage": "official_w4a8_tap_svdq_mixed_epilogue",
        "official_w4a8_debug_op": "torch.ops._C_ascend.svdq_w4a8_debug_readback",
        "svdq_lowrank_debug_op": "torch.ops._C_ascend.svdq_low_rank_debug_readback",
        "svdq_mixed_epilogue_debug_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "public_grouped_matmul_used": False,
        "production_svdq_host_tiling_fail_closed": True,
        "limitation": (
            "GMM2 residual tap comes from the official W4A8 debug path for its internally quantized hidden. "
            "This probe validates mixed residual-plus-SVDQ epilogues with official W4A8 taps; it does not yet "
            "relaunch official GMM2 from the SVDQ-modified hidden activation."
        ),
        "layer_index": args.layer,
        "routed_experts": routed_experts,
        "active_experts_from_w4a8": active_experts,
        "residual_checkpoint_key_count": residual_key_count,
        "factor_load_count": factor_load_count,
        "shape": {
            "num_tokens": args.num_tokens,
            "top_k": args.top_k,
            "active_rows": active_rows,
            "hidden_size": int(svdq_spec.hidden_size),
            "intermediate_size": int(svdq_spec.intermediate_size),
        },
        "rank_metadata": {
            "gate_rank": int(svdq_layer.svdq_gate_rank),
            "up_rank": int(svdq_layer.svdq_up_rank),
            "down_rank": int(svdq_layer.svdq_down_rank),
            "gate_rank_offset": int(svdq_layer.svdq_gate_rank_offset),
            "up_rank_offset": int(svdq_layer.svdq_up_rank_offset),
        },
        "w4a8_reference": w4a8_reference,
        "lowrank_readback": lowrank_stats,
        "lowrank_output_health_passed": lowrank_output_health_passed,
        "stage_errors": stage_errors,
        "stage_passed": stage_passed,
        "hidden_q_exact_match": bool(torch.equal(mixed_actual["hidden_q"], mixed_reference["stages"]["hidden_q"])),
        "hidden_q_mismatch_count": hidden_q_mismatch_count,
        "hidden_q_max_abs_diff": hidden_q_max_abs_diff,
        "hidden_q_mismatch_count_tolerance": args.hidden_q_mismatch_count_tol,
        "hidden_q_max_abs_diff_tolerance": args.hidden_q_max_abs_diff_tol,
        "passed": bool(w4a8_reference["passed"] and lowrank_output_health_passed and all(stage_passed.values())),
    }


def main() -> int:
    args = _parse_args()
    summary_path = args.evidence_dir / args.summary_name
    env = _npu_environment(args.device_id)
    symbol_status = _official_debug_symbol_status()
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_tap_mixed_epilogue_probe",
        "passed": False,
        "skipped": False,
        "preflight_failed": False,
        "environment": env,
        "official_debug_symbol_status": symbol_status,
    }
    if not (env["torch_npu_imported"] and env["npu_available"] and int(env["npu_device_count"]) > args.device_id):
        summary["skipped"] = not args.require_npu
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch_npu import and an available selected NPU are required."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False, "skipped": summary["skipped"]}, indent=2))
        return 2 if args.require_npu else 0
    if args.max_output_size < args.num_tokens * args.top_k:
        summary["failure_reason"] = "max-output-size must cover num_tokens * top_k."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1

    torch.npu.set_device(args.device_id)
    try:
        registered = _has_registered_debug_op()
    except Exception as exc:
        summary["preflight_failed"] = True
        summary["failure_reason"] = f"failed to enable custom ops: {type(exc).__name__}: {exc}"
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1
    summary["torch_op_registered"] = registered
    if not registered:
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch.ops._C_ascend.svdq_w4a8_debug_readback is not registered."
        _write_summary(summary_path, summary)
        print(json.dumps({"summary_path": str(summary_path), "passed": False}, indent=2))
        return 1

    group_info: dict[str, Any] = {}
    try:
        group_info = _init_single_rank_hccl(args.device_id)
        summary["hccl_group"] = group_info
        stage = _run_combined_probe(args, str(group_info["group"]))
        summary["stage"] = stage
        summary["passed"] = bool(stage["passed"])
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["passed"] = False
    finally:
        _destroy_hccl_if_needed(group_info)

    _write_summary(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path), "passed": summary["passed"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
