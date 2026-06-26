#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate official W4A8 GMM2 from SVDQ mixed-epilogue packed hidden.

This Stage 2.2 probe keeps the production SVDQ operator fail-closed. It uses
the already isolated mixed-epilogue debug op to produce canonical SVDQ-modified
hidden, plain INT8 hidden, official high/low packed INT4 hidden, and hidden
scale. It then relaunches the official W4A8 GMM2 consumer path through
``torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback`` with real post-loaded W2
checkpoint weights and compares the official GMM2 post-dequant readback against
the existing unfused reference derived from the official W4A8 contract.
"""

from __future__ import annotations

import argparse
import ctypes
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

from svdq_loader_pre_kernel_validate import DEFAULT_EVIDENCE_DIR, DEFAULT_MODEL_PATH  # noqa: E402
from svdq_mixed_epilogue_device_probe import (  # noqa: E402
    _make_inputs as _make_mixed_epilogue_inputs,
    _npu_environment,
    _run_npu_mixed_epilogue,
)
from svdq_w4a8_debug_readback_probe import (  # noqa: E402
    _destroy_hccl_if_needed,
    _init_single_rank_hccl,
)
from svdq_w4a8_debug_readback_real_checkpoint_probe import (  # noqa: E402
    _float_stats,
    _load_real_residual_layer,
    _official_gmm2_unfused_reference,
    _postload_metadata,
    _tensor_error,
)
from vllm_ascend.quantization.methods.svdq_post_load import pack_official_hidden_i4_reference  # noqa: E402
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op  # noqa: E402

CUSTOM_OPAPI_LIB = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
DEFAULT_SUMMARY_NAME = "phase_stage2_gmm2_from_mixed_hidden_summary.json"
_PRELOADED_CUSTOM_OPAPI_GLOBAL = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--evidence-dir", type=Path, default=Path(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--route-experts", type=int, nargs="*", default=None)
    parser.add_argument("--local-num-experts", type=int, default=None)
    parser.add_argument("--max-output-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--gmm2-reference-max-rows", type=int, default=64)
    parser.add_argument("--gmm2-reference-max-abs-tol", type=float, default=2e-4)
    parser.add_argument("--gmm2-reference-mean-abs-tol", type=float, default=2e-5)
    parser.add_argument("--require-npu", action="store_true")
    return parser.parse_args()


def _preload_custom_opapi() -> bool:
    global _PRELOADED_CUSTOM_OPAPI_GLOBAL
    if not CUSTOM_OPAPI_LIB.exists():
        return False
    ctypes.CDLL(str(CUSTOM_OPAPI_LIB), mode=ctypes.RTLD_GLOBAL)
    _PRELOADED_CUSTOM_OPAPI_GLOBAL = True
    return True


def _enable_custom_ops() -> None:
    bootstrap_custom_op_env(include_vendor_lib=True)
    _preload_custom_opapi()
    enable_custom_op()


def _has_registered_gmm2_debug_op() -> bool:
    _enable_custom_ops()
    return getattr(torch.ops._C_ascend, "svdq_w4a8_gmm2_debug_readback", None) is not None


def _routed_experts(args: argparse.Namespace) -> list[int]:
    experts = list(range(args.top_k)) if args.route_experts is None else list(args.route_experts)
    if len(experts) != args.top_k:
        raise ValueError("route-experts length must match top-k.")
    if len(set(experts)) != len(experts):
        raise ValueError("route-experts must not contain duplicates.")
    if min(experts) < 0:
        raise ValueError("route-experts must be non-negative.")
    return experts


def _local_num_experts(args: argparse.Namespace, routed_experts: list[int]) -> int:
    local_num_experts = max(routed_experts) + 1 if args.local_num_experts is None else int(args.local_num_experts)
    if local_num_experts <= 0:
        raise ValueError("local-num-experts must be positive.")
    if max(routed_experts) >= local_num_experts:
        raise ValueError("local-num-experts must cover every routed expert id.")
    return local_num_experts


def _make_expert_idx(routed_experts: list[int], num_tokens: int, *, device: torch.device) -> torch.Tensor:
    route = torch.tensor(routed_experts, dtype=torch.int32).reshape(1, len(routed_experts)).expand(num_tokens, -1)
    return route.contiguous().to(device=device)


def _make_expert_token_nums(
    *, routed_experts: list[int], num_tokens: int, local_num_experts: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros((1, local_num_experts), dtype=torch.int32)
    for expert_id in routed_experts:
        counts[0, int(expert_id)] = int(num_tokens)
    return counts.to(device=device)


def _tensor_int_exact(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    diff = (actual_cpu.to(torch.int16) - expected_cpu.to(torch.int16)).abs()
    mismatch = diff != 0
    first_mismatch_coords = mismatch.nonzero()[:16]
    first_mismatches: list[dict[str, Any]] = []
    for coord in first_mismatch_coords:
        index = tuple(int(v) for v in coord.tolist())
        first_mismatches.append(
            {
                "index": list(index),
                "actual": int(actual_cpu[index].item()),
                "expected": int(expected_cpu[index].item()),
                "abs_diff": int(diff[index].item()),
            }
        )
    mismatch_rows = mismatch.any(dim=1) if mismatch.dim() >= 2 else torch.empty((0,), dtype=torch.bool)
    mismatch_cols = mismatch.any(dim=0) if mismatch.dim() >= 2 else torch.empty((0,), dtype=torch.bool)
    return {
        "actual_shape": list(actual_cpu.shape),
        "expected_shape": list(expected_cpu.shape),
        "actual_dtype": str(actual_cpu.dtype),
        "expected_dtype": str(expected_cpu.dtype),
        "exact_match": bool(torch.equal(actual_cpu, expected_cpu)),
        "mismatch_count": int((diff != 0).sum().item()),
        "max_abs_diff": int(diff.max().item()) if diff.numel() else 0,
        "mismatch_row_count": int(mismatch_rows.sum().item()) if mismatch.dim() >= 2 else None,
        "mismatch_col_count": int(mismatch_cols.sum().item()) if mismatch.dim() >= 2 else None,
        "mismatch_rows_first32": mismatch_rows.nonzero().flatten()[:32].tolist() if mismatch.dim() >= 2 else [],
        "mismatch_cols_first32": mismatch_cols.nonzero().flatten()[:32].tolist() if mismatch.dim() >= 2 else [],
        "first_mismatches": first_mismatches,
        "actual_sample": actual_cpu.flatten()[:16].tolist(),
        "expected_sample": expected_cpu.flatten()[:16].tolist(),
    }


def _tensor_float_exact_locations(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_cpu = actual.detach().cpu().float()
    expected_cpu = expected.detach().cpu().float()
    diff = (actual_cpu - expected_cpu).abs()
    mismatch = diff != 0
    first_mismatch_coords = mismatch.nonzero()[:16]
    first_mismatches: list[dict[str, Any]] = []
    for coord in first_mismatch_coords:
        index = tuple(int(v) for v in coord.tolist())
        first_mismatches.append(
            {
                "index": list(index),
                "actual": float(actual_cpu[index].item()),
                "expected": float(expected_cpu[index].item()),
                "abs_diff": float(diff[index].item()),
            }
        )
    max_abs_index: list[int] = []
    if diff.numel():
        flat_index = int(diff.flatten().argmax().item())
        max_abs_index = list(torch.unravel_index(torch.tensor(flat_index), diff.shape))
        max_abs_index = [int(v) for v in max_abs_index]
    return {
        "exact_mismatch_count": int(mismatch.sum().item()),
        "exact_mismatch_indices_first32": mismatch.nonzero().flatten()[:32].tolist()
        if mismatch.dim() == 1
        else mismatch.nonzero()[:32].tolist(),
        "max_abs_index": max_abs_index,
        "first_exact_mismatches": first_mismatches,
    }


def _pad_hidden_boundary(
    *,
    hidden_x_int4_packed: torch.Tensor,
    hidden_x_scale: torch.Tensor,
    max_output_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_rows = int(hidden_x_int4_packed.shape[0])
    if active_rows > max_output_size:
        raise ValueError("active routed rows exceed max-output-size.")
    packed = torch.zeros((max_output_size, intermediate_size), dtype=torch.int8, device=device)
    scale = torch.zeros((max_output_size,), dtype=torch.float32, device=device)
    packed[:active_rows].copy_(hidden_x_int4_packed.to(device=device, dtype=torch.int8))
    scale[:active_rows].copy_(hidden_x_scale.to(device=device, dtype=torch.float32))
    return packed, scale


def _is_gmm2_loop_stats_debug(swiglu_limit: float) -> bool:
    return 430000.0 < float(swiglu_limit) < 440000.0


def _parse_gmm2_loop_stats(tensor: torch.Tensor, *, expert_per_rank: int) -> dict[str, Any]:
    values = tensor.detach().cpu().flatten()[:512].tolist()
    group_count = min(int(values[1]) if len(values) > 1 else 0, int(expert_per_rank), 48)
    groups = []
    for group_idx in range(group_count):
        base = 16 + group_idx * 5
        groups.append(
            {
                "group_idx": group_idx,
                "raw_current_m": int(values[base + 0]),
                "clipped_current_m": int(values[base + 1]),
                "doubled_current_m": int(values[base + 2]),
                "core_loops": int(values[base + 3]),
                "pre_current_m_sum": int(values[base + 4]),
            }
        )
    state_count = min(int(expert_per_rank), 32)
    state_probe = {
        "magic": float(values[256]) if len(values) > 256 else 0.0,
        "valid_magic": bool(len(values) > 256 and abs(float(values[256]) - 434344.0) < 0.5),
        "ep": int(values[257]) if len(values) > 257 else 0,
        "rank": int(values[258]) if len(values) > 258 else 0,
        "cumsum_base": int(values[259]) if len(values) > 259 else 0,
        "layout_base": int(values[260]) if len(values) > 260 else 0,
        "layout_base_equals_cumsum_base": bool(len(values) > 261 and int(values[261]) != 0),
        "external_expert_token_nums_first32": [int(values[272 + idx]) for idx in range(state_count)],
        "token_per_expert_cumsum_base_first32": [int(values[304 + idx]) for idx in range(state_count)],
        "token_per_expert_layout_base_first32": [int(values[336 + idx]) for idx in range(state_count)],
        "cumsum_mm_last_rank_first32": [int(values[368 + idx]) for idx in range(state_count)],
    }
    return {
        "enabled": True,
        "magic": float(values[0]) if values else 0.0,
        "expert_per_rank": int(values[1]) if len(values) > 1 else 0,
        "max_output_size": int(values[2]) if len(values) > 2 else 0,
        "total_active_rows": int(values[3]) if len(values) > 3 else 0,
        "total_doubled_rows": int(values[4]) if len(values) > 4 else 0,
        "total_core_loops": int(values[5]) if len(values) > 5 else 0,
        "groups_with_work": int(values[6]) if len(values) > 6 else 0,
        "final_pre_current_m_sum": int(values[7]) if len(values) > 7 else 0,
        "n2": int(values[8]) if len(values) > 8 else 0,
        "k2": int(values[9]) if len(values) > 9 else 0,
        "core_num": int(values[10]) if len(values) > 10 else 0,
        "groups": groups,
        "state_probe": state_probe,
        "valid_magic": bool(values and abs(float(values[0]) - 434343.0) < 0.5),
        "active_tile_count_nonzero": bool(len(values) > 5 and int(values[5]) > 0),
    }


def _run_stage(args: argparse.Namespace, group: str) -> dict[str, Any]:
    device = torch.device(f"npu:{args.device_id}")
    routed_experts = _routed_experts(args)
    local_num_experts = _local_num_experts(args, routed_experts)
    active_rows = int(args.num_tokens * args.top_k)
    if args.max_output_size < active_rows:
        raise ValueError("max-output-size must cover num_tokens * top_k routed rows.")

    layer, spec, residual_key_count = _load_real_residual_layer(
        model_path=args.model_path,
        layer_index=args.layer,
        tp_size=1,
        tp_rank=0,
        routed_experts=set(routed_experts),
        local_num_experts=local_num_experts,
    )
    if max(routed_experts) >= spec.num_experts:
        raise ValueError(f"route-experts={routed_experts} must be within [0, {spec.num_experts}).")

    mixed_inputs = _make_mixed_epilogue_inputs(
        num_tokens=active_rows,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        seed=args.seed,
    )
    mixed = _run_npu_mixed_epilogue(inputs=mixed_inputs, device=device, swiglu_limit=args.swiglu_limit)
    expected_packed = pack_official_hidden_i4_reference(mixed["hidden_q"])
    packed_exact = _tensor_int_exact(mixed["hidden_q_packed"], expected_packed)

    hidden_x_int4_packed, hidden_x_scale = _pad_hidden_boundary(
        hidden_x_int4_packed=mixed["hidden_q_packed"],
        hidden_x_scale=mixed["hidden_scale"],
        max_output_size=args.max_output_size,
        intermediate_size=spec.intermediate_size,
        device=device,
    )
    external_expert_token_nums = _make_expert_token_nums(
        routed_experts=routed_experts,
        num_tokens=args.num_tokens,
        local_num_experts=local_num_experts,
        device=device,
    )

    x = torch.zeros((args.num_tokens, spec.hidden_size), dtype=torch.bfloat16, device=device)
    expert_idx = _make_expert_idx(routed_experts, args.num_tokens, device=device)
    probs = torch.full((args.num_tokens, args.top_k), 1.0 / args.top_k, dtype=torch.float32, device=device)
    x_active_mask = torch.ones((args.num_tokens,), dtype=torch.bool, device=device)

    _enable_custom_ops()
    op = getattr(torch.ops._C_ascend, "svdq_w4a8_gmm2_debug_readback", None)
    if op is None:
        raise RuntimeError("torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback is not registered.")

    debug_outputs = op(
        x,
        [layer.w13_weight],
        [layer.w2_weight],
        expert_idx,
        [layer.w13_weight_scale],
        [layer.w2_weight_scale],
        [layer.w13_scale_bias],
        [layer.w2_scale_bias],
        probs,
        hidden_x_int4_packed,
        hidden_x_scale,
        external_expert_token_nums,
        group,
        args.max_output_size,
        x_active_mask,
        args.swiglu_limit,
    )
    if isinstance(debug_outputs, tuple):
        gmm2_post_dequant, hidden_x_readback, hidden_scale_readback = debug_outputs
    else:
        # Backward-compatible fallback for stale installs; the ABI test requires
        # the tuple-returning debug op after this diagnostic patch is installed.
        gmm2_post_dequant = debug_outputs
        hidden_x_readback = None
        hidden_scale_readback = None
    torch.npu.synchronize()
    hidden_x_readback_exact = (
        _tensor_int_exact(hidden_x_readback[:active_rows], hidden_x_int4_packed[:active_rows])
        if hidden_x_readback is not None
        else None
    )
    hidden_scale_readback_error = (
        _tensor_error(hidden_scale_readback[:active_rows], hidden_x_scale[:active_rows])
        if hidden_scale_readback is not None
        else None
    )
    if hidden_scale_readback_error is not None:
        hidden_scale_readback_error.update(
            _tensor_float_exact_locations(hidden_scale_readback[:active_rows], hidden_x_scale[:active_rows])
        )
    hidden_scale_readback_exact = (
        hidden_scale_readback_error is not None
        and hidden_scale_readback_error["actual_finite"]
        and hidden_scale_readback_error["expected_finite"]
        and hidden_scale_readback_error["diff_finite"]
        and hidden_scale_readback_error["max_abs"] == 0.0
    )

    loop_stats_debug = _is_gmm2_loop_stats_debug(args.swiglu_limit)
    reference, reference_contract = _official_gmm2_unfused_reference(
        hidden_x_int4_packed=hidden_x_int4_packed[:active_rows],
        hidden_x_scale=hidden_x_scale[:active_rows],
        weight=layer.w2_weight,
        weight_scale=layer.w2_weight_scale,
        scale_bias=layer.w2_scale_bias,
        expert_token_nums=external_expert_token_nums,
        output_columns=spec.hidden_size,
        max_rows=args.gmm2_reference_max_rows,
    )
    if loop_stats_debug:
        hidden_scale_active = hidden_x_scale[:active_rows].detach().cpu()
        loop_stats = _parse_gmm2_loop_stats(gmm2_post_dequant, expert_per_rank=local_num_experts)
        return {
            "stage": "stage2_modified_hidden_official_w4a8_gmm2_loop_stats",
            "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback -> aclnnSVDQW4A8GMM2DebugReadback",
            "mixed_hidden_source_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
            "official_source_of_truth": "dispatch_ffn_combine_w4_a8 GMM2 AIC scheduling path",
            "public_grouped_matmul_used": False,
            "real_checkpoint_validation": True,
            "production_svdq_host_tiling_fail_closed": True,
            "diagnostic_mode": "gmm2_loop_stats_only",
            "diagnostic_swiglu_limit": args.swiglu_limit,
            "layer_index": args.layer,
            "layer_name": spec.prefix,
            "residual_checkpoint_key_count": residual_key_count,
            "routed_experts": routed_experts,
            "shape": {
                "num_experts": spec.num_experts,
                "local_num_experts": local_num_experts,
                "hidden_size": spec.hidden_size,
                "intermediate_size": spec.intermediate_size,
                "num_tokens": args.num_tokens,
                "top_k": args.top_k,
                "max_output_size": args.max_output_size,
                "active_rows": active_rows,
            },
            "routing_identity": {
                "expert_token_nums_shape": list(external_expert_token_nums.shape),
                "expert_token_nums": external_expert_token_nums.detach().cpu().tolist(),
                "expert_token_total": int(external_expert_token_nums.detach().cpu().sum().item()),
                "expert_contiguous_rows": True,
                "row_source": "mixed epilogue rows are generated in the same expert-contiguous order described by expert_token_nums",
                "reference_group_counts": reference_contract["group_counts"],
            },
            "official_postload": {
                "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
                "metadata": _postload_metadata(layer),
            },
            "hidden_boundary": {
                "canonical_hidden_bf16": _float_stats(mixed["hidden_bf16"]),
                "hidden_int8": _float_stats(mixed["hidden_q"]),
                "hidden_int4_packed": _float_stats(hidden_x_int4_packed[:active_rows]),
                "hidden_scale": _float_stats(hidden_scale_active),
                "hidden_q_packed_exact_reference": packed_exact,
                "hidden_q_post_override_readback_exact_reference": hidden_x_readback_exact,
                "hidden_scale_post_override_readback_error": hidden_scale_readback_error,
                "hidden_scale_post_override_exact_match": hidden_scale_readback_exact,
                "hidden_q_exact_match_from_stage2_1_probe": None,
                "hidden_q_packed_exact_match": packed_exact["exact_match"],
                "hidden_q_packed_mismatch_count": packed_exact["mismatch_count"],
            },
            "gmm2": {
                "loop_stats": loop_stats,
                "unfused_reference": {
                    "enabled": True,
                    "contract": reference_contract,
                    "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
                    "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
                },
            },
            "checks": {
                "official_gmm2_entry_reached": bool(loop_stats["valid_magic"]),
                "official_gmm2_loop_count": int(loop_stats["total_core_loops"]),
                "official_gmm2_active_tile_count": int(loop_stats["total_core_loops"]),
                "official_gmm2_active_tile_count_nonzero": bool(loop_stats["active_tile_count_nonzero"]),
                "hidden_packed_exact": packed_exact["exact_match"],
                "hidden_packed_mismatch_count_zero": packed_exact["mismatch_count"] == 0,
                "hidden_scale_finite": bool(torch.isfinite(hidden_scale_active).all().item()),
                "hidden_scale_nonzero": bool(torch.any(hidden_scale_active.abs() > 0).item()),
                "canonical_hidden_finite": bool(torch.isfinite(mixed["hidden_bf16"].float()).all().item()),
                "canonical_hidden_nonzero": bool(torch.any(mixed["hidden_bf16"].float().abs() > 0).item()),
                "official_gmm2_numerical_gate_passed": False,
            },
            "passed": False,
        }

    actual = gmm2_post_dequant[: reference.shape[0], : reference.shape[1]]
    error = _tensor_error(actual, reference)
    gmm2_reference_passed = (
        error["actual_finite"]
        and error["expected_finite"]
        and error["diff_finite"]
        and error["max_abs"] <= args.gmm2_reference_max_abs_tol
        and error["mean_abs"] <= args.gmm2_reference_mean_abs_tol
    )

    hidden_scale_active = hidden_x_scale[:active_rows].detach().cpu()
    gmm2_active = gmm2_post_dequant[:active_rows].detach().cpu()
    return {
        "stage": "stage2_modified_hidden_official_w4a8_gmm2",
        "official_debug_op": "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback -> aclnnSVDQW4A8GMM2DebugReadback",
        "mixed_hidden_source_op": "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "official_source_of_truth": "dispatch_ffn_combine_w4_a8 GMM2 AIC path and W4A8_DEBUG AIV post-dequant tap",
        "public_grouped_matmul_used": False,
        "real_checkpoint_validation": True,
        "production_svdq_host_tiling_fail_closed": True,
        "layer_index": args.layer,
        "layer_name": spec.prefix,
        "residual_checkpoint_key_count": residual_key_count,
        "routed_experts": routed_experts,
        "shape": {
            "num_experts": spec.num_experts,
            "local_num_experts": local_num_experts,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "num_tokens": args.num_tokens,
            "top_k": args.top_k,
            "max_output_size": args.max_output_size,
            "active_rows": active_rows,
        },
        "routing_identity": {
            "expert_token_nums_shape": list(external_expert_token_nums.shape),
            "expert_token_nums": external_expert_token_nums.detach().cpu().tolist(),
            "expert_token_total": int(external_expert_token_nums.detach().cpu().sum().item()),
            "expert_contiguous_rows": True,
            "row_source": "mixed epilogue rows are generated in the same expert-contiguous order described by expert_token_nums",
        },
        "official_postload": {
            "loader": "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
            "metadata": _postload_metadata(layer),
        },
        "hidden_boundary": {
            "canonical_hidden_bf16": _float_stats(mixed["hidden_bf16"]),
            "hidden_int8": _float_stats(mixed["hidden_q"]),
            "hidden_int4_packed": _float_stats(hidden_x_int4_packed[:active_rows]),
            "hidden_scale": _float_stats(hidden_scale_active),
            "hidden_q_packed_exact_reference": packed_exact,
            "hidden_q_post_override_readback_exact_reference": hidden_x_readback_exact,
            "hidden_scale_post_override_readback_error": hidden_scale_readback_error,
            "hidden_scale_post_override_exact_match": hidden_scale_readback_exact,
            "hidden_q_exact_match_from_stage2_1_probe": None,
            "hidden_q_packed_exact_match": packed_exact["exact_match"],
            "hidden_q_packed_mismatch_count": packed_exact["mismatch_count"],
        },
        "gmm2": {
            "post_dequant_active": _float_stats(gmm2_active),
            "unfused_reference": {
                "enabled": True,
                "passed": bool(gmm2_reference_passed),
                "contract": reference_contract,
                "error": error,
                "max_abs_tolerance": args.gmm2_reference_max_abs_tol,
                "mean_abs_tolerance": args.gmm2_reference_mean_abs_tol,
            },
        },
        "checks": {
            "official_gmm2_kernel_launched": True,
            "hidden_packed_exact": packed_exact["exact_match"],
            "hidden_packed_mismatch_count_zero": packed_exact["mismatch_count"] == 0,
            "hidden_post_override_readback_exact": (
                hidden_x_readback_exact is not None and hidden_x_readback_exact["exact_match"]
            ),
            "hidden_scale_post_override_readback_exact": bool(hidden_scale_readback_exact),
            "hidden_scale_finite": bool(torch.isfinite(hidden_scale_active).all().item()),
            "hidden_scale_nonzero": bool(torch.any(hidden_scale_active.abs() > 0).item()),
            "canonical_hidden_finite": bool(torch.isfinite(mixed["hidden_bf16"].float()).all().item()),
            "canonical_hidden_nonzero": bool(torch.any(mixed["hidden_bf16"].float().abs() > 0).item()),
            "gmm2_reference_passed": bool(gmm2_reference_passed),
        },
        "passed": bool(
            packed_exact["exact_match"]
            and packed_exact["mismatch_count"] == 0
            and hidden_x_readback_exact is not None
            and hidden_x_readback_exact["exact_match"]
            and hidden_scale_readback_exact
            and torch.isfinite(hidden_scale_active).all().item()
            and torch.any(hidden_scale_active.abs() > 0).item()
            and torch.isfinite(mixed["hidden_bf16"].float()).all().item()
            and torch.any(mixed["hidden_bf16"].float().abs() > 0).item()
            and gmm2_reference_passed
        ),
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _finish_probe(payload: dict[str, Any], exit_code: int) -> int:
    print(json.dumps(payload, indent=2), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if _PRELOADED_CUSTOM_OPAPI_GLOBAL and os.environ.get("SVDQ_GMM2_DEBUG_ALLOW_CANN_TEARDOWN") != "1":
        os._exit(exit_code)
    return exit_code


def main() -> int:
    args = _parse_args()
    summary_path = args.evidence_dir / args.summary_name
    env = _npu_environment(args.device_id)
    summary: dict[str, Any] = {
        "probe": "svdq_w4a8_gmm2_from_mixed_hidden_probe",
        "passed": False,
        "skipped": False,
        "preflight_failed": False,
        "environment": env,
        "production_svdq_host_tiling_fail_closed": True,
    }
    if not (env["torch_npu_imported"] and env["npu_available"] and int(env["npu_device_count"]) > args.device_id):
        summary["skipped"] = not args.require_npu
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch_npu import and an available selected NPU are required."
        _write_summary(summary_path, summary)
        return _finish_probe(
            {"summary_path": str(summary_path), "passed": False, "skipped": summary["skipped"]},
            2 if args.require_npu else 0,
        )

    torch.npu.set_device(args.device_id)
    try:
        registered = _has_registered_gmm2_debug_op()
    except Exception as exc:
        summary["preflight_failed"] = True
        summary["failure_reason"] = f"failed to enable custom ops: {type(exc).__name__}: {exc}"
        _write_summary(summary_path, summary)
        return _finish_probe({"summary_path": str(summary_path), "passed": False, "skipped": False}, 1)
    summary["torch_op_registered"] = registered
    if not registered:
        summary["preflight_failed"] = True
        summary["failure_reason"] = "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback is not registered."
        _write_summary(summary_path, summary)
        return _finish_probe({"summary_path": str(summary_path), "passed": False, "skipped": False}, 1)

    group_info: dict[str, Any] = {}
    try:
        group_info = _init_single_rank_hccl(args.device_id)
        summary["hccl_group"] = group_info
        stage = _run_stage(args, str(group_info["group"]))
        summary["stage"] = stage
        summary["passed"] = bool(stage["passed"])
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        summary["passed"] = False
    finally:
        _destroy_hccl_if_needed(group_info)

    _write_summary(summary_path, summary)
    return _finish_probe(
        {"summary_path": str(summary_path), "passed": summary["passed"], "skipped": False},
        0 if summary["passed"] else 1,
    )


if __name__ == "__main__":
    raise SystemExit(main())
